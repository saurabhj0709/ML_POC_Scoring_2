
from pyspark.sql import SparkSession

# Create SparkSession
spark = (
    SparkSession.builder
    .appName("Validate_S3_Data")
    .master("local[2]")
    .getOrCreate()
)

# S3 path containing the Parquet file
s3_path = "s3a://mlops-poc2/raw/customer_features/spark_test/20261119T110513609495Z/"

# Read Parquet data
df = spark.read.parquet(s3_path)

# Print validation results
print("Number of rows:", df.count())
print("Number of columns:", len(df.columns))
print("Column names:", df.columns)
print(df.show(5))

spark.stop()
