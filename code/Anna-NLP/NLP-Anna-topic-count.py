#!/usr/bin/env python3
"""
NLP-Q1-2: Subreddit x Topic Counts

- Read document-topic distributions parquet
- Compute dominant topic per comment
- Aggregate counts by subreddit and dominant_topic
- Save to CSV for visualization
"""

import time
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F

print("=" * 80)
print("NLP-Q1-Anna: Subreddit x Topic Counts")
print("=" * 80)

overall_start = time.time()

spark = (
    SparkSession.builder
    .appName("NLPQ1-Anna-SubredditTopicCounts")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

print("\n[1/3] Reading doc-topic distributions parquet...")
step_start = time.time()
doc_topic = spark.read.parquet("data/parquet/NLPQ1_Anna_doc_topic_dist_ver2.parquet")
print(f"Loaded doc-topic distributions ({time.time() - step_start:.1f}s)")
print(f"Total rows: {doc_topic.count()}")

print("\n[2/3] Computing dominant topic per document...")

def argmax_topic(dist):
    arr = dist.toArray() if hasattr(dist, "toArray") else list(dist)
    max_idx = max(range(len(arr)), key=lambda i: arr[i])
    return int(max_idx)

argmax_udf = udf(argmax_topic, IntegerType())

doc_topic_dom = doc_topic.withColumn(
    "dominant_topic",
    argmax_udf(col("topicDistribution"))
)

print("\n[3/3] Aggregating counts by subreddit and dominant_topic...")
step_start = time.time()

sub_topic_counts = (
    doc_topic_dom
    .groupBy("subreddit", "dominant_topic")
    .count()
)

sub_topic_counts = sub_topic_counts.orderBy(F.col("count").desc())

os.makedirs("data", exist_ok=True)

sub_topic_pdf = sub_topic_counts.toPandas()
out_path = "data/NLPQ1_Anna_subreddit_topic_counts_ver2.csv"
sub_topic_pdf.to_csv(out_path, index=False)

print(f"Saved subreddit x topic counts to {out_path}")
print(f"Completed in {time.time() - overall_start:.1f}s")
print("=" * 80)
