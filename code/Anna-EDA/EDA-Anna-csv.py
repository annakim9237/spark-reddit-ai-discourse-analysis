#!/usr/bin/env python3
"""
EDA: Optimal Posting Time Analysis for AI-related Subreddits
When is the optimal time to post content to maximize engagement?
"""
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, dayofweek, hour, count, avg

print("=" * 80)
print("EDA-Anna-POSTING TIME ANALYSIS")
print("=" * 80)

overall_start = time.time()

# Create Spark session
print("\n[1/6] Creating Spark session...")
step_start = time.time()
spark = (
    SparkSession.builder
    .appName("EDA-Anna-PostingTimeAnalysis")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.InstanceProfileCredentialsProvider")
    .getOrCreate()
)
print(f"Spark session created ({time.time() - step_start:.1f}s)")

# Read data
print("\n[2/6] Reading data from S3...")
step_start = time.time()
# s3 bucket updated to hk1505-dsan6000-datasets
comments = spark.read.parquet("s3a://hk1505-dsan6000-datasets/project/reddit/parquet/comments/")
submissions = spark.read.parquet("s3a://hk1505-dsan6000-datasets/project/reddit/parquet/submissions/")
print(f"Data loaded ({time.time() - step_start:.1f}s)")

# Select columns
print("\n[3/6] Selecting columns and combining datasets...")
step_start = time.time()
comments_df = comments.select("id", "subreddit", "created_utc", "score")
submissions_df = submissions.select("id", "subreddit", "created_utc", "score")
combined = comments_df.union(submissions_df)
print(f"Combined datasets ({time.time() - step_start:.1f}s)")

# Add time features
print("\n[4/6] Adding time features (day_of_week, hour)...")
step_start = time.time()
combined = combined.withColumn("day_of_week", dayofweek(from_unixtime("created_utc")))
combined = combined.withColumn("hour", hour(from_unixtime("created_utc")))
print(f"Time features added ({time.time() - step_start:.1f}s)")

# Aggregate
print("\n[5/6] Aggregating data...")
step_start = time.time()
subreddit_agg = combined.groupBy("subreddit", "day_of_week", "hour").agg(
    count("id").alias("num_posts"),
    avg("score").alias("avg_score")
)
overall_agg = combined.groupBy("day_of_week", "hour").agg(
    count("id").alias("num_posts"),
    avg("score").alias("avg_score")
)
print(f"Aggregation complete ({time.time() - step_start:.1f}s)")

# Save to CSV
print("\n[6/6] Saving results to CSV...")
step_start = time.time()
subreddit_agg.toPandas().to_csv("data/csv/EDA_Anna_subreddit_time_analysis.csv", index=False)
overall_agg.toPandas().to_csv("data/csv/EDA_Anna_posting_time_analysis.csv", index=False)
print(f"CSV files saved ({time.time() - step_start:.1f}s)")

print("\n" + "=" * 80)
print(f"ANALYSIS COMPLETE! Total time: {time.time() - overall_start:.1f}s")
print("=" * 80)
print("\nOutput files:")
print("  - data/csv/EDA_Anna_subreddit_time_analysis.csv")
print("  - data/csv/EDA_Anna_posting_time_analysis.csv")

print("Done! Files saved to data/csv/")
