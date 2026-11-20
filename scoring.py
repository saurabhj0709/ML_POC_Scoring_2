"""Batch churn scoring on EC2: S3 Parquet -> PySpark features -> sklearn pickle -> S3.

The pickle must be the complete sklearn Pipeline created by
Churn_Prediction_Simple(2).ipynb. The S3 input must be one *completed*
run of extract_snowflake_batch.py, not the parent spark_full/ prefix.

Set the paths in the configuration below and run:
    spark-submit scoring.py

Keep scoring.py and churn_model_simple.pkl in the same GitHub repository
and directory. No command-line arguments are required.
"""

import pickle
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ----------------------------------------------------------------------
# CONFIGURATION: edit these values before running on EC2.
# ----------------------------------------------------------------------
REGION = "ap-southeast-1"

# Use the exact timestamped folder from a SUCCESSFUL, complete extraction.
INPUT_S3_PATH = (
    "s3a://mlops-poc2/raw/customer_features/spark_full/20261119T160605716662Z/"
)

# Change test_001/full_001 on each new run; Spark will not overwrite results.
TEST_MODE = True  # True: 5 customers. False: all 1,000,000 customers.
TEST_OUTPUT_S3_PATH = "s3a://mlops-poc2/predictions/churn/test_001/"
FULL_OUTPUT_S3_PATH = "s3a://mlops-poc2/predictions/churn/full_001/"
OUTPUT_S3_PATH = TEST_OUTPUT_S3_PATH if TEST_MODE else FULL_OUTPUT_S3_PATH

# Resolved relative to this script so it works from any current directory.
MODEL_PATH = Path(__file__).resolve().parent / "churn_model_simple.pkl"

EXPECTED_ROWS = 5 if TEST_MODE else 1000000
# ----------------------------------------------------------------------

# Exact feature names AND order from the training notebook.
NUMERIC_FEATURES = [
    "AGE", "TENURE_MONTHS", "ACCOUNT_COUNT", "TOTAL_BALANCE",
    "TOTAL_CREDIT_LIMIT", "CREDIT_ACCOUNT_COUNT", "BALANCE_TO_LIMIT",
    "TXN_COUNT_90D", "TOTAL_SPEND_90D", "UPI_TXN_COUNT_90D",
    "AVG_DEBIT_AMOUNT_90D", "SUPPORT_TICKETS_90D",
    "UNRESOLVED_TICKETS_90D", "LOGINS_30D", "ACTIVE_DAYS_30D",
    "AVG_SESSION_MIN_30D", "TICKETS_PER_ACCOUNT_90D",
]
MODEL_FEATURES = NUMERIC_FEATURES + ["CUSTOMER_SEGMENT"]

# Columns actually written to Parquet by extract_snowflake_batch.py.
REQUIRED_INPUT = [
    "CUSTOMER_ID", "AGE", "CUSTOMER_SEGMENT", "JOIN_DATE",
    "ACCOUNT_COUNT", "TOTAL_BALANCE", "TOTAL_CREDIT_LIMIT",
    "CREDIT_ACCOUNT_COUNT", "TXN_COUNT_90D", "TOTAL_SPEND_90D",
    "DEBIT_COUNT_90D", "UPI_TXN_COUNT_90D", "SUPPORT_TICKETS_90D",
    "UNRESOLVED_TICKETS_90D", "LOGINS_30D", "ACTIVE_DAYS_30D",
    "AVG_SESSION_MIN_30D", "AS_OF_DATE",
]

OUTPUT_SCHEMA = T.StructType([
    T.StructField("CUSTOMER_ID", T.LongType(), False),
    T.StructField("AS_OF_DATE", T.DateType(), False),
    T.StructField("PREDICTED_CHURN", T.IntegerType(), False),
    T.StructField("CHURN_PROBABILITY", T.DoubleType(), False),
])


def safe_ratio(numerator, denominator):
    """Reproduce notebook's division + fillna(0), without dividing by zero."""
    ratio = F.when(
        F.col(denominator).isNotNull() & (F.col(denominator) != 0),
        F.col(numerator) / F.col(denominator),
    ).otherwise(F.lit(0.0))
    return F.coalesce(ratio, F.lit(0.0)).cast("double")


def create_features(df):
    """Rebuild four engineered features exactly as in the notebook."""
    # Ensure dates and numeric types have the expected representation.
    df = (df.withColumn("JOIN_DATE", F.to_date("JOIN_DATE"))
            .withColumn("AS_OF_DATE", F.to_date("AS_OF_DATE"))
            .withColumn("CUSTOMER_ID", F.col("CUSTOMER_ID").cast("long"))
            .withColumn("CUSTOMER_SEGMENT", F.col("CUSTOMER_SEGMENT").cast("string")))

    for col in set(REQUIRED_INPUT) - {
        "CUSTOMER_ID", "CUSTOMER_SEGMENT", "JOIN_DATE", "AS_OF_DATE"
    }:
        df = df.withColumn(col, F.col(col).cast("double"))

    # Training fills missing aggregated activity with zero, while missing AGE
    # and TENURE_MONTHS go to the sklearn pipeline's median imputer.
    activity_cols = [
        "TXN_COUNT_90D", "TOTAL_SPEND_90D", "DEBIT_COUNT_90D",
        "UPI_TXN_COUNT_90D", "SUPPORT_TICKETS_90D", "UNRESOLVED_TICKETS_90D",
        "LOGINS_30D", "ACTIVE_DAYS_30D", "AVG_SESSION_MIN_30D",
    ]
    df = df.fillna(0.0, subset=activity_cols)

    days_since_join = F.datediff(F.col("AS_OF_DATE"), F.col("JOIN_DATE"))
    df = (
        df.withColumn(
            "TENURE_MONTHS",
            F.when(F.col("JOIN_DATE").isNull(), F.lit(None).cast("double"))
             .otherwise(F.greatest(days_since_join.cast("double") / F.lit(30.44), F.lit(0.0)))
        )
        .withColumn("BALANCE_TO_LIMIT", safe_ratio("TOTAL_BALANCE", "TOTAL_CREDIT_LIMIT"))
        .withColumn("AVG_DEBIT_AMOUNT_90D", safe_ratio("TOTAL_SPEND_90D", "DEBIT_COUNT_90D"))
        .withColumn("TICKETS_PER_ACCOUNT_90D", safe_ratio("SUPPORT_TICKETS_90D", "ACCOUNT_COUNT"))
    )

    # All numeric predictors are sent to pandas as float/double.
    for feature in NUMERIC_FEATURES:
        df = df.withColumn(feature, F.col(feature).cast("double"))
    return df.select("CUSTOMER_ID", "AS_OF_DATE", *MODEL_FEATURES)


def check_pickle(model_bytes):
    """Fail before S3 processing if model is not the saved sklearn pipeline."""
    model = pickle.loads(model_bytes)  # Only load a pickle you created/trust.
    if not (hasattr(model, "predict") and hasattr(model, "predict_proba")):
        raise ValueError("Pickle must contain a classifier pipeline with predict_proba")
    if not (hasattr(model, "named_steps") and "preprocessing" in model.named_steps):
        raise ValueError("Pickle must contain the training preprocessing pipeline")
    recorded = getattr(model, "feature_names_in_", None)
    if recorded is not None and list(recorded) != MODEL_FEATURES:
        raise ValueError("Model feature names/order do not match scoring.py")
    return model


def validate_input(df, expected_rows):
    """Check for incomplete runs, duplicate customer IDs and missing snapshot dates."""
    stats = df.agg(
        F.count("*").alias("rows"),
        F.countDistinct("CUSTOMER_ID").alias("unique_customers"),
        F.count("AS_OF_DATE").alias("dated_rows"),
    ).first()
    print(f"Input: {stats['rows']} rows; {stats['unique_customers']} unique customers", flush=True)

    if stats["rows"] == 0:
        raise ValueError("Input S3 directory has no customer records")
    if stats["unique_customers"] != stats["rows"]:
        raise ValueError("Found missing/duplicate CUSTOMER_ID values")
    if stats["dated_rows"] != stats["rows"]:
        raise ValueError("Some rows have null AS_OF_DATE")
    if expected_rows is not None and stats["rows"] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} customers, found {stats['rows']}; "
            "the extraction may be incomplete."
        )
    snapshot_dates = df.select("AS_OF_DATE").distinct().limit(2).collect()
    if len(snapshot_dates) != 1:
        raise ValueError("A single run should have one AS_OF_DATE")
    print(f"Snapshot date: {snapshot_dates[0]['AS_OF_DATE']}", flush=True)


def main():
    if "REPLACE_WITH_RUN_ID" in INPUT_S3_PATH:
        raise ValueError("Set INPUT_S3_PATH to your actual completed extraction run folder")
    if not INPUT_S3_PATH.startswith("s3a://") or not OUTPUT_S3_PATH.startswith("s3a://"):
        raise ValueError("INPUT_S3_PATH and OUTPUT_S3_PATH must start with s3a://")
    if (OUTPUT_S3_PATH.rstrip("/").startswith(INPUT_S3_PATH.rstrip("/") + "/")
            or INPUT_S3_PATH.rstrip("/") == OUTPUT_S3_PATH.rstrip("/")):
        raise ValueError("Output must not be inside the input directory")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found next to scoring.py: {MODEL_PATH}")

    model_bytes = MODEL_PATH.read_bytes()
    check_pickle(model_bytes)
    print(f"Loaded and checked model: {MODEL_PATH}", flush=True)
    print(f"TEST_MODE: {TEST_MODE}", flush=True)

    spark = (
        SparkSession.builder
        .appName("Churn_Batch_Scoring")
        .master("local[2]")  # Same two-core EC2 configuration as extraction script.
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "2000")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint.region", REGION)
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        print("Reading:", INPUT_S3_PATH, flush=True)
        raw = spark.read.parquet(INPUT_S3_PATH)  # Loads ALL Parquet part files under one run.
        missing = sorted(set(REQUIRED_INPUT) - set(raw.columns))
        if missing:
            raise ValueError("Required S3 columns are missing: " + ", ".join(missing))

        if TEST_MODE:
            raw = raw.limit(5)
        validate_input(raw, EXPECTED_ROWS)
        features = create_features(raw)
        broadcast_model = spark.sparkContext.broadcast(model_bytes)

        def predict_batches(pandas_iterator):
            # One model load per Spark task, rather than one model load per row.
            local_model = pickle.loads(broadcast_model.value)
            for pdf in pandas_iterator:
                if pdf.empty:
                    continue
                inputs = pdf[MODEL_FEATURES]  # Keep training order and column names.
                probabilities = local_model.predict_proba(inputs)[:, 1]
                predictions = local_model.predict(inputs)
                yield pd.DataFrame({
                    "CUSTOMER_ID": pdf["CUSTOMER_ID"].astype("int64"),
                    "AS_OF_DATE": pd.to_datetime(pdf["AS_OF_DATE"]).dt.date,
                    "PREDICTED_CHURN": predictions.astype("int32"),
                    "CHURN_PROBABILITY": probabilities.astype("float64"),
                })

        scored = features.mapInPandas(predict_batches, schema=OUTPUT_SCHEMA)
        print("Writing:", OUTPUT_S3_PATH, flush=True)
        (
            scored.write.mode("errorifexists")
            .option("header", "true")
            .csv(OUTPUT_S3_PATH)
        )
        print("SUCCESS: predictions written to", OUTPUT_S3_PATH, flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
