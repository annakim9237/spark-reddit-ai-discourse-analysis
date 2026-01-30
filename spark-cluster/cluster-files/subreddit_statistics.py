#!/usr/bin/env python3
# ------------------------------------------------------------
# Reddit Subreddit Statistics (Cluster Version with auto-detect months)
# Input  : s3a://ea973-dsan6000-datasets/reddit/parquet/comments/yyyy=YYYY/mm=MM/
# Output : subreddit_statistics_YYYY_MM.csv (monthly)
#          subreddit_statistics_YYYY.csv (yearly)
#          subreddit_statistics_all_years.csv (combined)
# Author : Erika Atoma
# ------------------------------------------------------------

import os, sys, time, logging, glob, shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, sum as ssum, round as sround, lit

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("subreddit_cluster")

# ---------------- Spark Setup ----------------
def build_spark(master_url: str) -> SparkSession:
    log.info(f"🚀 Starting Spark on {master_url}")
    return (
        SparkSession.builder
        .appName("SubredditStatisticsCluster")
        .master(master_url)
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider")
        .config("spark.hadoop.fs.s3a.requester.pays.enabled", "true")
        .config("fs.s3a.requester.pays.enabled", "true")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

# ---------------- Helper ----------------
def write_single_csv(df, output_path):
    """Flatten Spark CSV output into a single file."""
    tmp_dir = f"{output_path}_tmp"
    df.coalesce(1).write.csv(tmp_dir, header=True, mode="overwrite")
    part_files = glob.glob(os.path.join(tmp_dir, "part-*.csv"))
    if part_files:
        shutil.move(part_files[0], output_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log.info(f"✅ Wrote: {output_path}")
    else:
        log.warning(f"⚠️ No CSV part file found for {output_path}")

# ---------------- Main ----------------
def main():
    if len(sys.argv) > 1:
        master_url = sys.argv[1]
    else:
        ip = os.getenv("MASTER_PRIVATE_IP")
        if not ip:
            print("❌ Provide Spark master URL: spark://<MASTER_PRIVATE_IP>:7077")
            return 1
        master_url = f"spark://{ip}:7077"

    spark = build_spark(master_url)
    spark.sparkContext.setLogLevel("WARN")

    outdir = os.path.expanduser("~/spark-cluster/data/csv")
    os.makedirs(outdir, exist_ok=True)

    s3_base = "s3a://ea973-dsan6000-datasets/reddit/parquet"
    YEARS = [2023, 2024]
    all_years_df = None

    # --- detect existing year/month partitions dynamically ---
    for year in YEARS:
        log.info(f"📅 Checking available months for {year}")
        try:
            # List subfolders (mm=..) under comments/yyyy=YEAR/
            months_df = spark.read.format("parquet").load(f"{s3_base}/comments/yyyy={year}/")
            available_months = sorted({row['month'] for row in months_df.select("month").distinct().collect()})
        except Exception:
            # fallback if metadata is missing: try fixed 06–12 for 2023
            available_months = [f"{m:02d}" for m in range(6, 13)] if year == 2023 else []
        log.info(f"🗓️ Found months for {year}: {available_months}")

        monthly_dataframes = []

        for month in available_months:
            comments_path = f"{s3_base}/comments/yyyy={year}/mm={month}/"
            submissions_path = f"{s3_base}/submissions/yyyy={year}/mm={month}/"
            log.info(f"🔹 Processing {year}-{month}")

            try:
                comments_df = spark.read.parquet(comments_path)
            except Exception as e:
                log.warning(f"⚠️ No comments data for {year}-{month}: {e}")
                continue

            try:
                submissions_df = spark.read.parquet(submissions_path)
            except Exception as e:
                log.warning(f"⚠️ No submissions data for {year}-{month}: {e}")
                # fallback: empty DataFrame with matching schema
                submissions_df = spark.createDataFrame([], comments_df.schema)

            if comments_df.rdd.isEmpty():
                log.warning(f"⚠️ Empty comments for {year}-{month}")
                continue

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
            all_years_df = yearly_stats if all_years_df is None else all_years_df.unionByName(yearly_stats)

    if all_years_df:
        all_years_path = os.path.join(outdir, "subreddit_statistics_all_years.csv")
        write_single_csv(all_years_df, all_years_path)
        log.info("✅ Combined all years written")

    log.info("🏁 Finished successfully.")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
