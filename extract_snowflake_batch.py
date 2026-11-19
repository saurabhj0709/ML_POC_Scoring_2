"""Snowflake -> PySpark customer-level extraction -> S3 Parquet.

Runs on EC2 using the Snowflake Spark connector and Hadoop S3A.

TEST_MODE = True: extract just customer IDs 1..5.
TEST_MODE = False: extract IDs 1..1,000,000 in 50,000-customer batches.

No credentials are stored in this file: reads ~/.env.
Each batch appends one Parquet part file to a unique timestamped S3 prefix.
20 batches x 50,000 customers = 1,000,000 customer records (assuming IDs 1..1,000,000).
If a batch fails, the run is INCOMPLETE; do not use its data for scoring.
"""

import os
from datetime import datetime, timedelta, timezonegit
from pathlib import Path

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Configuration
TEST_MODE = False  # Set True to re-run the 5-customer test.
BATCH_SIZE = 50_000
FIRST_CUSTOMER_ID = 1
LAST_CUSTOMER_ID = 1_000_000  # Matches this POC's synthetic CUSTOMERS table.
BUCKET = "mlops-poc2"
REGION = "ap-southeast-1"
SF_FORMAT = "net.snowflake.spark.snowflake"

OUTPUT_COLUMNS = [
    "CUSTOMER_ID", "AGE", "CUSTOMER_SEGMENT", "JOIN_DATE",
    "ACCOUNT_COUNT", "TOTAL_BALANCE", "TOTAL_CREDIT_LIMIT", "CREDIT_ACCOUNT_COUNT",
    "TXN_COUNT_90D", "TOTAL_SPEND_90D", "DEBIT_COUNT_90D", "UPI_TXN_COUNT_90D",
    "SUPPORT_TICKETS_90D", "UNRESOLVED_TICKETS_90D",
    "LOGINS_30D", "ACTIVE_DAYS_30D", "AVG_SESSION_MIN_30D", "AS_OF_DATE",
]


def snowflake_options():
    load_dotenv(Path.home() / ".env")
    required = (
        "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))

    account = os.environ["SNOWFLAKE_ACCOUNT"].strip()
    url = account if account.endswith(".snowflakecomputing.com") else f"{account}.snowflakecomputing.com"
    return {
        "sfURL": url,
        "sfUser": os.environ["SNOWFLAKE_USER"],
        "sfPassword": os.environ["SNOWFLAKE_PASSWORD"],
        "sfWarehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        "sfDatabase": os.environ["SNOWFLAKE_DATABASE"],
        "sfSchema": os.environ["SNOWFLAKE_SCHEMA"],
        "autopushdown": "on",
    }


def read_table(spark, options, name, columns):
    return (
        spark.read.format(SF_FORMAT)
        .options(**options)
        .option("dbtable", name)
        .load()
        .select(*columns)
    )


def count_when(condition):
    return F.sum(F.when(condition, F.lit(1)).otherwise(F.lit(0)))


def extract_batch(spark, sf_options, first_id, last_id, as_of, start_90d, start_30d):
    """Build one 18-column customer-level Spark DataFrame for this ID range."""
    customer_filter = F.col("CUSTOMER_ID").between(first_id, last_id)

    customers = read_table(
        spark, sf_options, "CUSTOMERS",
        ["CUSTOMER_ID", "AGE", "CUSTOMER_SEGMENT", "JOIN_DATE"],
    ).filter(customer_filter)

    accounts = read_table(
        spark, sf_options, "ACCOUNTS",
        ["ACCOUNT_ID", "CUSTOMER_ID", "ACCOUNT_TYPE", "BALANCE", "CREDIT_LIMIT"],
    ).filter(customer_filter)

    transactions = read_table(
        spark, sf_options, "TRANSACTIONS",
        ["ACCOUNT_ID", "TRANSACTION_DATE", "AMOUNT", "TRANSACTION_TYPE", "CHANNEL"],
    ).filter(F.col("TRANSACTION_DATE").between(F.lit(start_90d), F.lit(as_of)))

    tickets = read_table(
        spark, sf_options, "SUPPORT_TICKETS",
        ["CUSTOMER_ID", "TICKET_DATE", "RESOLUTION_STATUS"],
    ).filter(customer_filter).filter(
        F.col("TICKET_DATE").between(F.lit(start_90d), F.lit(as_of))
    )

    digital = read_table(
        spark, sf_options, "DIGITAL_ACTIVITY",
        ["CUSTOMER_ID", "ACTIVITY_DATE", "LOGIN_COUNT", "SESSION_DURATION"],
    ).filter(customer_filter).filter(
        F.col("ACTIVITY_DATE").between(F.lit(start_30d), F.lit(as_of))
    )

    account_summary = accounts.groupBy("CUSTOMER_ID").agg(
        F.count("ACCOUNT_ID").alias("ACCOUNT_COUNT"),
        F.sum("BALANCE").alias("TOTAL_BALANCE"),
        F.sum("CREDIT_LIMIT").alias("TOTAL_CREDIT_LIMIT"),
        count_when(F.col("ACCOUNT_TYPE") == "CREDIT").alias("CREDIT_ACCOUNT_COUNT"),
    )

    transaction_summary = (
        transactions
        .join(accounts.select("ACCOUNT_ID", "CUSTOMER_ID"), "ACCOUNT_ID", "inner")
        .groupBy("CUSTOMER_ID")
        .agg(
            F.count(F.lit(1)).alias("TXN_COUNT_90D"),
            F.sum(
                F.when(F.col("TRANSACTION_TYPE") == "DEBIT", F.col("AMOUNT"))
                .otherwise(F.lit(0))
            ).alias("TOTAL_SPEND_90D"),
            count_when(F.col("TRANSACTION_TYPE") == "DEBIT").alias("DEBIT_COUNT_90D"),
            count_when(F.col("CHANNEL") == "UPI").alias("UPI_TXN_COUNT_90D"),
        )
    )

    support_summary = tickets.groupBy("CUSTOMER_ID").agg(
        F.count(F.lit(1)).alias("SUPPORT_TICKETS_90D"),
        count_when(F.col("RESOLUTION_STATUS") != "RESOLVED").alias("UNRESOLVED_TICKETS_90D"),
    )

    digital_summary = digital.groupBy("CUSTOMER_ID").agg(
        F.sum("LOGIN_COUNT").alias("LOGINS_30D"),
        F.countDistinct("ACTIVITY_DATE").alias("ACTIVE_DAYS_30D"),
        F.avg("SESSION_DURATION").alias("AVG_SESSION_MIN_30D"),
    )

    result = (
        customers
        .join(account_summary, "CUSTOMER_ID", "left")
        .join(transaction_summary, "CUSTOMER_ID", "left")
        .join(support_summary, "CUSTOMER_ID", "left")
        .join(digital_summary, "CUSTOMER_ID", "left")
        .withColumn("AS_OF_DATE", F.lit(as_of).cast("date"))
    )

    numeric_cols = OUTPUT_COLUMNS[4:-1]
    result = result.fillna(0, subset=numeric_cols)
    return result.select(*OUTPUT_COLUMNS)


def main():
    sf_options = snowflake_options()
    spark = (
        SparkSession.builder
        .appName("POC2_Batched_Snowflake_Extraction")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint.region", REGION)
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider",
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        as_of = datetime.now(timezone.utc).date()
        start_90d = as_of - timedelta(days=89)
        start_30d = as_of - timedelta(days=29)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_kind = "spark_test" if TEST_MODE else "spark_full"
        output_path = f"s3a://{BUCKET}/raw/customer_features/{run_kind}/{run_id}/"

        first_id = FIRST_CUSTOMER_ID
        last_id = 5 if TEST_MODE else LAST_CUSTOMER_ID
        current_batch_size = 5 if TEST_MODE else BATCH_SIZE
        num_batches = (last_id - first_id + current_batch_size) // current_batch_size

        print("TEST_MODE:", TEST_MODE, flush=True)
        print("Customer IDs:", first_id, "to", last_id, flush=True)
        print("Number of batches:", num_batches, flush=True)
        print("Output S3 path:", output_path, flush=True)

        for batch_index, batch_start in enumerate(range(first_id, last_id + 1, current_batch_size), start=1):
            batch_end = min(batch_start + current_batch_size - 1, last_id)
            print(f"BATCH {batch_index}/{num_batches}: IDs {batch_start}..{batch_end}", flush=True)

            batch_df = extract_batch(
                spark, sf_options, batch_start, batch_end,
                as_of, start_90d, start_30d,
            )

            # One output part-file per batch; 50k output rows fit in a Parquet part-file.
            # This does NOT bring the dataset into pandas or collect it to the driver.
            (
                batch_df.coalesce(1)
                .write.mode("append")
                .option("compression", "snappy")
                .parquet(output_path)
            )
            print(f"FINISHED BATCH {batch_index}/{num_batches}", flush=True)

        print("SUCCESS - all batches saved to:", output_path, flush=True)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
