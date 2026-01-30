"""
DATA QUALITY SCRIPT (Simplified)

Outputs:
    - Missingness summary
    - Deleted/removed content %
    - Duplicate ID count per subreddit
    - Duplicate user % (authors) per subreddit
    - Timestamp range
    - Score distribution summary per subreddit
    - Text quality indicators per subreddit

Author: Mandy Sun
"""


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys


def main():

    if len(sys.argv) < 2:
        raise ValueError("Usage: spark-submit data_quality_checks.py <netid>")

    netid = sys.argv[1]

    spark = (
        SparkSession.builder
        .appName(f"DataQualityChecks_{netid}")
        .getOrCreate()
    )

    # -------------------------------------------------------------
    # Parameterized Project Paths (NO ABSOLUTE PATHS)
    # -------------------------------------------------------------
    base_path = f"s3a://{netid}/project"

    comments_path = f"{base_path}/reddit/comments/"
    submissions_path = f"{base_path}/reddit/submissions/"
    output_dir = f"{base_path}/data_quality"

    print("\nStarting Simplified Data Quality Checks...\n")
    print("COMMENTS PATH:", comments_path)
    print("SUBMISSIONS PATH:", submissions_path)
    print("OUTPUT DIR:", output_dir, "\n")

    # -------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------
    comments = spark.read.parquet(comments_path).withColumn("data_type", F.lit("comment"))
    submissions = spark.read.parquet(submissions_path).withColumn("data_type", F.lit("submission"))

    df = comments.unionByName(submissions, allowMissingColumns=True)

    total_rows = df.count()
    print(f"Loaded {total_rows:,} rows.\n")

    # Helper function: percentage calculation
    def pct(cond):
        return (F.sum(F.when(cond, 1).otherwise(0)) / F.count("*") * 100)

    # -------------------------------------------------------------
    # 1. Missingness per column per subreddit
    # -------------------------------------------------------------
    missingness_by_sub = (
        df.groupBy("subreddit")
        .agg(*[
            F.sum(F.col(c).isNull().cast("int")).alias(f"{c}_nulls")
            for c in df.columns
        ])
    )

    missingness_by_sub.write.csv(
        f"{output_dir}/missingness_by_subreddit",
        header=True, mode="overwrite"
    )

    # -------------------------------------------------------------
    # 2. Deleted / removed content + controversiality per subreddit
    # -------------------------------------------------------------
    deleted_removed = (
        df.groupBy("subreddit").agg(
            pct(F.col("author") == "[deleted]").alias("pct_author_deleted"),
            pct(F.col("body") == "[removed]").alias("pct_body_removed"),
            pct(F.col("selftext") == "[removed]").alias("pct_selftext_removed"),

            F.avg("controversiality").alias("avg_controversiality"),
            pct(F.col("controversiality") > 0).alias("pct_controversial"),

            F.avg("score").alias("avg_score")
        )
    )

    deleted_removed.write.csv(
        f"{output_dir}/deleted_removed_by_subreddit",
        header=True, mode="overwrite"
    )

    # -------------------------------------------------------------
    # 3. Duplicate ID analysis per subreddit
    # -------------------------------------------------------------
    id_counts = df.groupBy("subreddit", "id").count()

    duplicate_ids_by_subreddit = (
        id_counts.filter(F.col("count") > 1)
        .groupBy("subreddit")
        .agg(
            F.count("*").alias("num_duplicate_ids"),
            F.sum("count").alias("total_duplicate_rows")
        )
    )

    duplicate_ids_by_subreddit.write.csv(
        f"{output_dir}/duplicate_ids_by_subreddit",
        header=True, mode="overwrite"
    )

    # -------------------------------------------------------------
    # 4. Duplicate user (authors) analysis per subreddit
    # -------------------------------------------------------------
    author_counts_by_sub = df.groupBy("subreddit", "author").count()

    duplicate_users_by_subreddit = (
        author_counts_by_sub.groupBy("subreddit")
        .agg(
            F.count(F.when(F.col("count") > 1, True)).alias("num_duplicate_authors"),
            F.count("*").alias("total_authors"),
        )
        .withColumn(
            "duplicate_author_pct",
            (F.col("num_duplicate_authors") / F.col("total_authors")) * 100
        )
    )

    duplicate_users_by_subreddit.write.csv(
        f"{output_dir}/duplicate_users_by_subreddit",
        header=True, mode="overwrite"
    )

    # -------------------------------------------------------------
    # 5. Timestamp range
    # -------------------------------------------------------------
    ts_range = df.select(
        F.min("created_utc").alias("min_timestamp"),
        F.max("created_utc").alias("max_timestamp")
    )

    ts_range.write.csv(
        f"{output_dir}/timestamp_range",
        header=True, mode="overwrite"
    )

    # -------------------------------------------------------------
    # 6. Score Distribution Summary per subreddit
    # -------------------------------------------------------------
    score_stats = df.groupBy("subreddit").agg(
        F.min("score").alias("min_score"),
        F.expr("percentile_approx(score, 0.25)").alias("q1"),
        F.expr("percentile_approx(score, 0.50)").alias("median"),
        F.expr("percentile_approx(score, 0.75)").alias("q3"),
        F.max("score").alias("max_score"),
        F.avg("score").alias("avg_score")
    )

    score_stats.write.csv(
        f"{output_dir}/score_stats_by_subreddit",
        header=True, mode="overwrite"
    )

    # -------------------------------------------------------------
    # 7. Text Quality Indicators per subreddit
    # -------------------------------------------------------------
    text_quality_df = df.groupBy("subreddit").agg(
        F.avg(F.length("body")).alias("avg_body_len"),
        pct(F.length("body") < 3).alias("pct_short_body"),
        pct(F.length("body") > 5000).alias("pct_long_body"),

        F.avg(F.length("title")).alias("avg_title_len"),
        pct(F.length("title") < 3).alias("pct_short_title"),
        pct(F.length("title") > 5000).alias("pct_long_title"),

        F.avg(F.length("selftext")).alias("avg_selftext_len"),
        pct(F.length("selftext") < 3).alias("pct_short_selftext"),
        pct(F.length("selftext") > 5000).alias("pct_long_selftext")
    )

    text_quality_df.write.csv(
        f"{output_dir}/text_quality_by_subreddit",
        header=True, mode="overwrite"
    )

    # -------------------------------------------------------------
    print("\n✓ Simplified Data Quality Check Complete!")
    print("Files saved to:", output_dir, "\n")

    spark.stop()


if __name__ == "__main__":
    main()