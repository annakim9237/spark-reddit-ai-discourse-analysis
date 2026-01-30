"""
FINAL AUTHOR EDA — TOP 1% GLOBAL TECH/AI POSTERS (PARAMETERIZED)

Outputs (to S3 under {netid} namespace):
1. top1pct_global_posters/
2. top1pct_example_posts/

Description:
Computes global top 1% posters in TECH/AI subreddits and extracts their top posts.
Fully parameterized using <netid> for portable, per-user S3 routing.

Author: Mandy Sun
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import sys


def main():

    # ---------------------------------------------------------
    # ARGUMENT CHECK
    # ---------------------------------------------------------
    if len(sys.argv) < 2:
        raise ValueError("Usage: spark-submit final_author_eda.py <netid>")

    netid = sys.argv[1]

    # ---------------------------------------------------------
    # SPARK SESSION
    # ---------------------------------------------------------
    spark = (
        SparkSession.builder
        .appName(f"AuthorEDA_Top1Percent_{netid}")
        .getOrCreate()
    )

    print("SPARK MASTER:", spark.sparkContext.master)

    # ---------------------------------------------------------
    # PARAMETERIZED S3 PATHS
    # ---------------------------------------------------------
    bucket_root = f"s3a://{netid}-dsan6000/project"

    comments_path = f"{bucket_root}/reddit/comments/"
    submissions_path = f"{bucket_root}/reddit/submissions/"

    output_root = f"{bucket_root}/author_eda_top1pct"

    print("\nWriting outputs to:", output_root)

    # ---------------------------------------------------------
    # LOAD PARQUET DATA
    # ---------------------------------------------------------
    comments = spark.read.parquet(comments_path).withColumn("data_type", F.lit("comment"))
    submissions = spark.read.parquet(submissions_path).withColumn("data_type", F.lit("submission"))

    df = comments.unionByName(submissions, allowMissingColumns=True).cache()
    print("\nLoaded total rows:", df.count())

    # ---------------------------------------------------------
    # TECH/AI ECOSYSTEM FILTER
    # ---------------------------------------------------------
    TECH_SUBS = [
        "ChatGPT", "ClaudeAI", "PerplexityAI", "OpenAI", "GPT4",
        "LocalLLaMA", "StableDiffusion", "MachineLearning",
        "datascience", "programming", "computerscience",
        "Futurology", "neoliberal", "singularity"
    ]

    df_tech = df.filter(F.col("subreddit").isin(TECH_SUBS)).cache()
    print("Tech-only rows:", df_tech.count())

    # ---------------------------------------------------------
    # GLOBAL AUTHOR STATS (TECH ONLY)
    # ---------------------------------------------------------
    print("\nComputing global author stats in tech/AI ecosystem...")

    author_stats = (
        df_tech.groupBy("author")
               .agg(
                    F.count("*").alias("total_posts"),
                    F.countDistinct("subreddit").alias("num_subreddits"),
                    F.avg("score").alias("avg_score"),
                    F.max("score").alias("max_score"),
                    F.min("created_utc").alias("first_post"),
                    F.max("created_utc").alias("last_post"),
               )
               .withColumn("active_span_seconds", F.col("last_post") - F.col("first_post"))
               .cache()
    )

    total_authors = author_stats.count()
    print("Total tech/AI authors:", total_authors)

    # ---------------------------------------------------------
    # TOP 1% CUTOFF (P99)
    # ---------------------------------------------------------
    p99 = author_stats.agg(
        F.expr("percentile_approx(total_posts, 0.99)").alias("p99_posts")
    ).collect()[0]["p99_posts"]

    print("\nTop 1% threshold (p99 total_posts):", p99)

    # ---------------------------------------------------------
    # SELECT TOP 1% AUTHORS
    # ---------------------------------------------------------
    top1_authors = (
        author_stats.filter(F.col("total_posts") >= p99)
                    .cache()
    )

    print("Top 1% author count:", top1_authors.count())

    # ---------------------------------------------------------
    # SUBREDDIT FOOTPRINT (TECH ONLY)
    # ---------------------------------------------------------
    tech_author_subs = (
        df_tech.groupBy("author")
               .agg(F.collect_set("subreddit").alias("tech_subreddits"))
               .withColumn("num_tech_subreddits", F.size("tech_subreddits"))
               .withColumn("tech_subreddits_str", F.concat_ws(", ", "tech_subreddits"))
               .drop("tech_subreddits")
    )

    # attach footprint
    top1_authors = top1_authors.join(tech_author_subs, "author", "left")

    # ---------------------------------------------------------
    # WRITE TOP 1% GLOBAL AUTHORS
    # ---------------------------------------------------------
    top1_authors.write.mode("overwrite").option("header", True).csv(
        f"{output_root}/top1pct_global_posters"
    )

    print("\nWROTE: top1pct_global_posters")

    # =============================================================
    # EXAMPLE POSTS (TOP 1% AUTHORS ONLY)
    # =============================================================
    print("\nExtracting example posts (top 5 by score per subreddit)...")

    # join avoids driver OOM
    top1_df = df_tech.join(top1_authors.select("author"), "author", "inner").cache()
    print("Rows by top 1% authors:", top1_df.count())

    w = Window.partitionBy("subreddit").orderBy(F.desc("score"))

    example_posts = (
        top1_df.withColumn("rank", F.row_number().over(w))
               .filter(F.col("rank") <= 5)
               .select("subreddit", "author", "title", "body", "score", "created_utc")
    )

    example_posts.write.mode("overwrite").option("header", True).csv(
        f"{output_root}/top1pct_example_posts"
    )

    print("WROTE: top1pct_example_posts")

    # ---------------------------------------------------------
    print("\nTOP 1% AUTHOR EDA COMPLETED.")
    print("Outputs saved to:", output_root)

    spark.stop()


if __name__ == "__main__":
    main()
