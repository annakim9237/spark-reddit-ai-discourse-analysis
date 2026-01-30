#!/usr/bin/env python3
# ------------------------------------------------------------
# Reddit Subreddit Statistics (Local Spark Test with S3 requester-pays)
# Input  : s3a://ea973-dsan6000-datasets/reddit/parquet/comments/yyyy=YYYY/mm=MM/
#          s3a://ea973-dsan6000-datasets/reddit/parquet/submissions/yyyy=YYYY/mm=MM/
# Output : data/csv/subreddit_statistics_YYYY_MM.csv
#          data/csv/subreddit_statistics_YYYY.csv
# Author : Erika Atoma
# ------------------------------------------------------------

import os, glob, shutil, logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, sum as ssum, round as sround, lit

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("subreddit_local_s3")

# ---------------- Spark Setup ----------------
def build_spark() -> SparkSession:
    """Create a Spark session in local mode with S3 requester-pays access."""
    log.info("🚀 Starting Spark in local[*] mode with S3 requester-pays and hadoop-aws")
    return (
        SparkSession.builder
        .appName("SubredditStatisticsLocalS3")
        .master("local[*]")
        .config("spark.executor.memory", "3g")
        .config("spark.driver.memory", "3g")
        # include Hadoop-AWS + AWS SDK for S3A
        .config("spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.3.6,com.amazonaws:aws-java-sdk-bundle:1.12.603")
        # core S3A settings
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider")
        .config("spark.hadoop.fs.s3a.requester.pays.enabled", "true")
        .config("fs.s3a.requester.pays.enabled", "true")
        # fix timeouts — numbers only
        .config("spark.hadoop.fs.s3a.connection.timeout", "600000")  # 600000 ms = 10 minutes
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
        .config("spark.hadoop.fs.s3a.attempts.maximum", "20")
        # enable adaptive optimization
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


# ---------------- Helper ----------------
def write_single_csv(df, output_path):
    """Writes a Spark DataFrame as a single consolidated CSV file."""
    tmp_dir = f"{output_path}_tmp"
    df.coalesce(1).write.csv(tmp_dir, header=True, mode="overwrite")
    part_files = glob.glob(os.path.join(tmp_dir, "part-*.csv"))
    if part_files:
        shutil.move(part_files[0], output_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log.info(f"✅ Wrote: {output_path}")
    else:
        log.warning(f"⚠️ No CSV part file found in {tmp_dir}")

# ---------------- Main ----------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local Spark test run using S3 requester-pays Parquet files.")
    parser.add_argument("--year", type=int, default=2024, help="Target year, e.g. 2024")
    parser.add_argument("--months", nargs="+", default=["01"], help="List of months, e.g. --months 01 02 03")
    args = parser.parse_args()

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    outdir = "data/csv"
    os.makedirs(outdir, exist_ok=True)
    s3_base = "s3a://ea973-dsan6000-datasets/reddit/parquet"

    year = args.year
    months = args.months
    monthly_dataframes = []

    for month in months:
        comments_path = f"{s3_base}/comments/yyyy={year}/mm={month}/"
        submissions_path = f"{s3_base}/submissions/yyyy={year}/mm={month}/"
        log.info(f"📂 Reading {year}-{month} from {comments_path}")

        try:
            comments_df = spark.read.parquet(comments_path)
        except Exception as e:
            log.warning(f"⚠️ No comments data for {year}-{month}: {e}")
            continue

        try:
            submissions_df = spark.read.parquet(submissions_path)
        except Exception as e:
            log.warning(f"⚠️ No submissions data for {year}-{month}: {e}")
            submissions_df = spark.createDataFrame([], comments_df.schema)

        if comments_df.rdd.isEmpty():
            log.warning(f"⚠️ Empty comments data for {year}-{month}")
            continue

        # --- Aggregations ---
        comments_stats = comments_df.groupBy("subreddit").agg(
            count("id").alias("num_comments"),
            avg("score").alias("avg_score_comments")
        )

        submissions_stats = submissions_df.groupBy("subreddit").agg(
            count("id").alias("num_submissions"),
            avg("score").alias("avg_score_submissions")
        )

        month_stats = (
            comments_stats.join(submissions_stats, "subreddit", "outer")
            .na.fill(0)
            .withColumn("year", lit(year))
            .withColumn("month", lit(month))
            .withColumn("total_rows", col("num_comments") + col("num_submissions"))
            .withColumn("avg_score",
                sround((col("avg_score_comments") + col("avg_score_submissions")) / 2, 2)
            )
            .select("subreddit", "year", "month",
                    "num_comments", "num_submissions", "total_rows", "avg_score")
        )

        monthly_dataframes.append(month_stats)
        monthly_path = os.path.join(outdir, f"subreddit_statistics_{year}_{month}.csv")
        write_single_csv(month_stats, monthly_path)

    # --- Yearly rollup ---
    if monthly_dataframes:
        year_df = monthly_dataframes[0]
        for df in monthly_dataframes[1:]:
            year_df = year_df.unionByName(df, allowMissingColumns=True)

        yearly_stats = year_df.groupBy("subreddit").agg(
            ssum("num_comments").alias("num_comments"),
            ssum("num_submissions").alias("num_submissions"),
            ssum("total_rows").alias("total_rows"),
            avg("avg_score").alias("avg_score")
        ).withColumn("year", lit(year))

        yearly_path = os.path.join(outdir, f"subreddit_statistics_{year}.csv")
        write_single_csv(yearly_stats, yearly_path)

    log.info("🏁 Finished local S3 requester-pays test successfully.")
    spark.stop()


if __name__ == "__main__":
    main()
