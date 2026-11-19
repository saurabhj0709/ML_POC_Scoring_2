

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# --------------------- CHANGE ONLY THIS FOR FULL RUN ---------------------
TEST_MODE = False            # Change to False AFTER the 5-customer test succeeds
TEST_CUSTOMER_MAX = 5
# ------------------------------------------------------------------------

BUCKET = "mlops-poc2"
REGION = "ap-southeast-1"
SNOWFLAKE_FORMAT = "net.snowflake.spark.snowflake"

OUTPUT_COLUMNS = [
    "CUSTOMER_ID", "AGE", "CUSTOMER_SEGMENT", "JOIN_DATE",
    "ACCOUNT_COUNT", "TOTAL_BALANCE", "TOTAL_CREDIT_LIMIT", "CREDIT_ACCOUNT_COUNT",
    "TXN_COUNT_90D", "TOTAL_SPEND_90D", "DEBIT_COUNT_90D", "UPI_TXN_COUNT_90D",
    "SUPPORT_TICKETS_90D", "UNRESOLVED_TICKETS_90D",
    "LOGINS_30D", "ACTIVE_DAYS_30D", "AVG_SESSION_MIN_30D", "AS_OF_DATE",
]


def snowflake_connection_options():
    # Credentials live on EC2 in /home/ssm-user/.env (or current user's home).
    load_dotenv(Path.home() / ".env")

    names = [
        "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
    ]
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError("Missing .env variables: " + ", ".join(missing))

    account = os.environ["SNOWFLAKE_ACCOUNT"].strip()
    sf_url = (
        account if account.endswith(".snowflakecomputing.com")
        else f"{account}.snowflakecomputing.com"
    )

    return {
        "sfURL": sf_url,
        "sfUser": os.environ["SNOWFLAKE_USER"],
        "sfPassword": os.environ["SNOWFLAKE_PASSWORD"],
        "sfWarehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        "sfDatabase": os.environ["SNOWFLAKE_DATABASE"],
        "sfSchema": os.environ["SNOWFLAKE_SCHEMA"],
        "autopushdown": "on",
    }


def read_snowflake_table(spark, options, table_name, columns):
    # DataFrame API, not a hand-written SQL query.
    return (
        spark.read
        .format(SNOWFLAKE_FORMAT)
        .options(**options)
        .option("dbtable", table_name)
        .load()
        .select(*columns)
    )


def count_if(condition):
    # PySpark equivalent of SQL COUNT_IF(condition).
    return F.sum(F.when(condition, F.lit(1)).otherwise(F.lit(0)))


def main():
    sf_options = snowflake_connection_options()

    spark = (
        SparkSession.builder
        .appName("POC2_Snowflake_PySpark_Extraction")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint.region", REGION)
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider",
        )
        .getOrCreate()
    )

    try:
        print("Spark version:", spark.version)
        print("TEST_MODE:", TEST_MODE)

        # Use one date for all rolling-window calculations and the output snapshot.
        as_of_date = datetime.now(timezone.utc).date()
        start_90d = as_of_date - timedelta(days=89)
        start_30d = as_of_date - timedelta(days=29)

        # 1. Read only required raw columns from each Snowflake table.
        customers = read_snowflake_table(
            spark, sf_options, "CUSTOMERS",
            ["CUSTOMER_ID", "AGE", "CUSTOMER_SEGMENT", "JOIN_DATE"],
        )
        accounts = read_snowflake_table(
            spark, sf_options, "ACCOUNTS",
            ["ACCOUNT_ID", "CUSTOMER_ID", "ACCOUNT_TYPE", "BALANCE", "CREDIT_LIMIT"],
        )
        transactions = read_snowflake_table(
            spark, sf_options, "TRANSACTIONS",
            ["ACCOUNT_ID", "TRANSACTION_DATE", "AMOUNT", "TRANSACTION_TYPE", "CHANNEL"],
        )
        tickets = read_snowflake_table(
            spark, sf_options, "SUPPORT_TICKETS",
            ["CUSTOMER_ID", "TICKET_DATE", "RESOLUTION_STATUS"],
        )
        digital = read_snowflake_table(
            spark, sf_options, "DIGITAL_ACTIVITY",
            ["CUSTOMER_ID", "ACTIVITY_DATE", "LOGIN_COUNT", "SESSION_DURATION"],
        )

        # 2. Restrict the initial test to customer IDs 1 through 5.
        if TEST_MODE:
            customer_range = F.col("CUSTOMER_ID").between(1, TEST_CUSTOMER_MAX)
            customers = customers.filter(customer_range)
            accounts = accounts.filter(customer_range)
            tickets = tickets.filter(customer_range)
            digital = digital.filter(customer_range)

            # TRANSACTIONS has no CUSTOMER_ID. Restrict by the test customers'
            # ACCOUNT_ID values so a small test does not request unrelated rows.
            test_account_ids = [
                row["ACCOUNT_ID"]
                for row in accounts.select("ACCOUNT_ID").distinct().collect()
            ]
            print("Test customer account IDs:", test_account_ids)
            if test_account_ids:
                transactions = transactions.filter(
                    F.col("ACCOUNT_ID").isin(test_account_ids)
                )
            else:
                transactions = transactions.filter(F.lit(False))

        # 3. ACCOUNTS: one aggregated row per customer.
        account_summary = accounts.groupBy("CUSTOMER_ID").agg(
            F.count("ACCOUNT_ID").alias("ACCOUNT_COUNT"),
            F.sum("BALANCE").alias("TOTAL_BALANCE"),
            F.sum("CREDIT_LIMIT").alias("TOTAL_CREDIT_LIMIT"),
            count_if(F.col("ACCOUNT_TYPE") == "CREDIT").alias("CREDIT_ACCOUNT_COUNT"),
        )

        # 4. TRANSACTIONS: join to account IDs, aggregate at customer level.
        txn_90d = transactions.filter(
            F.col("TRANSACTION_DATE").between(F.lit(start_90d), F.lit(as_of_date))
        )
        transaction_summary = (
            txn_90d
            .join(accounts.select("ACCOUNT_ID", "CUSTOMER_ID"), on="ACCOUNT_ID", how="inner")
            .groupBy("CUSTOMER_ID")
            .agg(
                F.count(F.lit(1)).alias("TXN_COUNT_90D"),
                F.sum(
                    F.when(F.col("TRANSACTION_TYPE") == "DEBIT", F.col("AMOUNT"))
                    .otherwise(F.lit(0))
                ).alias("TOTAL_SPEND_90D"),
                count_if(F.col("TRANSACTION_TYPE") == "DEBIT").alias("DEBIT_COUNT_90D"),
                count_if(F.col("CHANNEL") == "UPI").alias("UPI_TXN_COUNT_90D"),
            )
        )

        # 5. SUPPORT_TICKETS: two 90-day customer-level aggregates.
        support_summary = (
            tickets
            .filter(F.col("TICKET_DATE").between(F.lit(start_90d), F.lit(as_of_date)))
            .groupBy("CUSTOMER_ID")
            .agg(
                F.count(F.lit(1)).alias("SUPPORT_TICKETS_90D"),
                count_if(F.col("RESOLUTION_STATUS") != "RESOLVED")
                .alias("UNRESOLVED_TICKETS_90D"),
            )
        )

        # 6. DIGITAL_ACTIVITY: three 30-day customer-level aggregates.
        digital_summary = (
            digital
            .filter(F.col("ACTIVITY_DATE").between(F.lit(start_30d), F.lit(as_of_date)))
            .groupBy("CUSTOMER_ID")
            .agg(
                F.sum("LOGIN_COUNT").alias("LOGINS_30D"),
                F.countDistinct("ACTIVITY_DATE").alias("ACTIVE_DAYS_30D"),
                F.avg("SESSION_DURATION").alias("AVG_SESSION_MIN_30D"),
            )
        )

        # 7. Join all four summaries to CUSTOMERS (one row per customer).
        final_df = (
            customers
            .join(account_summary, on="CUSTOMER_ID", how="left")
            .join(transaction_summary, on="CUSTOMER_ID", how="left")
            .join(support_summary, on="CUSTOMER_ID", how="left")
            .join(digital_summary, on="CUSTOMER_ID", how="left")
            .withColumn("AS_OF_DATE", F.lit(as_of_date).cast("date"))
        )

        # Same NULL -> 0 behavior as the earlier Snowflake SQL query.
        for col_name in OUTPUT_COLUMNS[4:-1]:
            final_df = final_df.withColumn(
                col_name, F.coalesce(F.col(col_name), F.lit(0))
            )

        final_df = final_df.select(*OUTPUT_COLUMNS)

        # 8. One consolidated dataset in S3 (potentially several part-files).
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_type = "spark_test" if TEST_MODE else "spark_full"
        output_path = f"s3a://{BUCKET}/raw/customer_features/{run_type}/{run_id}/"

        if TEST_MODE:
            final_df.show(truncate=False)
            final_df = final_df.coalesce(1)  # Expect one part-file for 5 rows.
        else:
            # Avoid forcing all 1M output rows through one partition.
            final_df = final_df.repartition(20, "CUSTOMER_ID")

        print("Output columns:", final_df.columns)
        print("Saving Parquet to:", output_path)
        (
            final_df.write
            .mode("errorifexists")
            .option("compression", "snappy")
            .option("maxRecordsPerFile", 50000)
            .parquet(output_path)
        )
        print("SUCCESS - extraction saved to:", output_path)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
