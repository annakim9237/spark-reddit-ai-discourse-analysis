#!/usr/bin/env python3
"""
Reddit Data Filtering Script (Master Node Version)

Run this directly on your Spark master node using:
  spark-submit reddit_data_filter_master.py
"""

import logging
import os
import time
from typing import List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_unixtime, to_date

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Customize your NET_ID here or set via environment variable
NET_ID = os.getenv("NET_ID", "yl2035")

# Spark master URL (adjust if needed)
MASTER_URL = "spark://localhost:7077"

# Subreddits of interest
EXAMPLE_SUBREDDITS: List[str] = [
    "ChatGPT", "OpenAI", "GPT4", "ClaudeAI", "PerplexityAI",
    "StableDiffusion", "MidJourney", "Sora", "AIArt",
    "GenerativeAI", "ArtificialIntelligence", "MachineLearning", "computerscience", "datascience", "programming",
    "Futurology", "singularity", "neoliberal",
    "LocalLLaMA", "OpenAI_Dev",
    "AskReddit",
]

COMMENT_COLUMNS = [
    "id", "subreddit", "author", "body", "score", "created_utc",
    "parent_id", "link_id", "controversiality", "gilded",
]

SUBMISSION_COLUMNS = [
    "id", "subreddit", "author", "title", "selftext", "score",
    "created_utc", "num_comments", "url", "over_18",
]

def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("Reddit_Data_Filter_Master")
        .master(MASTER_URL)
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    logger.info("Spark session created")
    return spark

def read_data(spark: SparkSession, data_type: str) -> DataFrame:
    path = f"s3a://{NET_ID}-dsan6000-datasets/reddit/parquet/{data_type}/"
    logger.info(f"Reading {data_type} from {path}")
    df = spark.read.parquet(path)
    print(f"Loaded {df.count():,} rows of {data_type}")
    return df

def filter_subreddits(df: DataFrame, subreddits: List[str]) -> DataFrame:
    return df.filter(col("subreddit").isin(subreddits))

def select_columns(df: DataFrame, columns: List[str]) -> DataFrame:
    selected = [c for c in columns if c in df.columns]
    df = df.select(*selected)
    return df.withColumn("date", to_date(from_unixtime(col("created_utc"))))

def save_data(df: DataFrame, data_type: str) -> None:
    path = f"s3a://{NET_ID}-dsan6000-datasets/project_filter/reddit/parquet/{data_type}/"
    df.write.mode("overwrite").parquet(path)
    print(f"Saved filtered {data_type} to {path}")

def show_sample(df: DataFrame, data_type: str, n: int = 5) -> None:
    print(f"\nSample {data_type} data:")
    df.show(n, truncate=50)

def process(spark: SparkSession, data_type: str, subreddits: List[str]) -> None:
    print(f"\nProcessing {data_type.upper()}...")
    df = read_data(spark, data_type)
    columns = COMMENT_COLUMNS if data_type == "comments" else SUBMISSION_COLUMNS
    df = filter_subreddits(df, subreddits)
    df = select_columns(df, columns)
    show_sample(df, data_type)
    save_data(df, data_type)

def main() -> None:
    print("=" * 80)
    print("REDDIT DATA FILTERING (MASTER NODE)")
    print("=" * 80)
    print(f"NET_ID: {NET_ID}")
    print(f"Spark Master: {MASTER_URL}")
    print(f"Subreddits: {', '.join(EXAMPLE_SUBREDDITS)}")

    spark = create_spark_session()
    start = time.time()

    try:
        process(spark, "comments", EXAMPLE_SUBREDDITS)
        process(spark, "submissions", EXAMPLE_SUBREDDITS)
        print("\n✅ Filtering completed successfully!")
    except Exception as e:
        logger.exception("Processing failed")
        print(f"\n❌ Error: {e}")
    finally:
        spark.stop()
        print(f"\nTotal time: {time.time() - start:.1f} seconds")
        print("=" * 80)

if __name__ == "__main__":
    main()
