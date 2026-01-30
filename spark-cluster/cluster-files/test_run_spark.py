'''
Simple Spark cluster sanity test
FILE: test_run_spark.py
'''

from pyspark.sql import SparkSession
import os

MASTER_URL = f"spark://{os.environ.get('MASTER_PRIVATE_IP', '172.31.95.19')}:7077"

spark = (
    SparkSession.builder
    .appName("Reddit_S3_Test")
    .master(MASTER_URL)
    .config(
        "spark.jars.packages",
        ",".join([
            # Spark NLP
            "com.johnsnowlabs.nlp:spark-nlp_2.12:6.2.0",
            # S3 connector versions compatible with Spark 3.5.x + Hadoop 3.3.x
            "org.apache.hadoop:hadoop-aws:3.3.4",
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        ])
    )
    # S3A filesystem + IAM role from EC2 instance
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider",
    )
    .getOrCreate()
)


# 1) Basic dataframe test
df = spark.range(10)
df.show()

# 2) Test S3A if you’re using S3 (replace bucket/key)
df_s3 = spark.read.parquet("s3a://ea973-reddit-datasets/project/reddit/parquet/comments/part-00000-2534334c-8a86-4056-bebc-0f0d9b8e7074-c000.snappy.parquet")
df_s3.show(5)

spark.stop()
