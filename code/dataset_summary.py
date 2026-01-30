#!/usr/bin/env python3
"""
Milestone 0: Data Acquisition and Initial Filtering
1.Dataset Statistics (save to data/csv/ folder):
1.1. dataset_summary
output: data/csv/dataset_summary.csv - Overall statistics
Columns: data_type (comments/submissions)
         , total_rows
         , size_gb
         , date_range_start
         , date_range_end

"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, min as f_min, max as f_max

spark = SparkSession.builder.appName("1.1.DatasetSummary").getOrCreate()

# 2) Read comments and submissions parquet files
comments = spark.read.parquet("s3://hk1105-dsan6000-datasets/reddit/parquet/comments/")
submissions = spark.read.parquet("s3://hk1105-dsan6000-datasets/reddit/parquet/submissions/")

# 3) Comments summary
comments_minmax = comments.select(
    f_min("created_utc").alias("date_range_start"),
    f_max("created_utc").alias("date_range_end"),
)

comments_total = comments.count()

comments_summary = (
    comments_minmax
    .withColumn("data_type", lit("comments"))
    .withColumn("total_rows", lit(comments_total))
    
    # aws s3 ls s3://hk1105-dsan6000-datasets/project/reddit/parquet/comments/ --recursive --human-readable --summarize
    #Total Objects: 3678
    #Total Size: 7.7 GiB
    .withColumn("size_gb", lit(7.7))
)

# 4) Submissions summary
submissions_minmax = submissions.select(
    f_min("created_utc").alias("date_range_start"),
    f_max("created_utc").alias("date_range_end"),
)
submissions_total = submissions.count()

submissions_summary = (
    submissions_minmax
    .withColumn("data_type", lit("submissions"))
    .withColumn("total_rows", lit(submissions_total))

    # aws s3 ls s3://hk1105-dsan6000-datasets/project/reddit/parquet/submissions/ --recursive --human-readable --summarize
    #Total Objects: 574
    # Total Size: 392.8 MiB → 0.384 (392.8 / 1024)
    .withColumn("size_gb", lit(0.384))
)

# 5) Combine summaries
summary_df = comments_summary.union(submissions_summary)

# 6) Save to CSV
summary_pd = summary_df.toPandas()
summary_pd.to_csv("data/csv/dataset_summary.csv", index=False)