"""
SUBREDDIT OVERLAP ANALYSIS — TECH/AI ECOSYSTEM ONLY

Outputs (written to S3):
1. subreddit_overlap_pairs/
      - Pairwise subreddit co-occurrence counts
      - Columns:
            sub_1
            sub_2
            shared_authors

Description:
This script restricts analysis to the TECH/AI subreddit ecosystem
and computes which subreddit PAIRS share the most authors.
A pair (A, B) counts how many distinct authors posted in both A and B.

This forms the foundation for:
- community overlap analysis
- network graph construction
- clustering subreddits by shared users

Author: Mandy Sun
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys


def main():

    # ---------------------------------------------------------
    # ARGUMENT CHECK
    # ---------------------------------------------------------
    if len(sys.argv) < 2:
        raise ValueError("Usage: spark-submit subreddit_overlap_tech_only.py <netid>")

    netid = sys.argv[1]

    # ---------------------------------------------------------
    # SPARK SESSION
    # ---------------------------------------------------------
    spark = (
        SparkSession.builder
        .appName(f"SubredditOverlap_TechOnly_{netid}")
        .getOrCreate()
    )

    print("SPARK MASTER:", spark.sparkContext.master)

    # ---------------------------------------------------------
    # S3 DIRECTORIES (DATASET BUCKET)
    # ---------------------------------------------------------
    comments_path = f"s3a://{netid}-dsan6000-datasets/project/reddit/parquet/comments/"
    submissions_path = f"s3a://{netid}-dsan6000-datasets/project/reddit/parquet/submissions/"


    s3_output_root = f"s3a://{netid}-dsan6000-datasets/project/subreddit_overlap_tech_only"
    print("\nSaving outputs to:", s3_output_root)
    print("COMMENTS PATH:", comments_path)
    print("SUBMISSIONS PATH:", submissions_path)

    # ---------------------------------------------------------
    # LOAD PARQUET DATA
    # ---------------------------------------------------------
    comments = spark.read.parquet(comments_path).withColumn("data_type", F.lit("comment"))
    submissions = spark.read.parquet(submissions_path).withColumn("data_type", F.lit("submission"))

    df = comments.unionByName(submissions, allowMissingColumns=True).cache()
    print("\nLoaded total rows:", df.count())

    # ---------------------------------------------------------
    # TECH/AI SUBREDDITS ONLY
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
    # PAIRWISE SUBREDDIT OVERLAP
    # ---------------------------------------------------------
    print("\nComputing subreddit overlap pairs...")

    # Each row: distinct author + the subs they posted in
    authors_by_sub = df_tech.select("author", "subreddit").distinct()

    # Self-join to find subreddit pairs per author
    pairs = (
        authors_by_sub.alias("a")
        .join(authors_by_sub.alias("b"), "author")
        .filter(F.col("a.subreddit") < F.col("b.subreddit"))   # Prevent (A,B) & (B,A)
        .groupBy("a.subreddit", "b.subreddit")
        .agg(F.countDistinct("author").alias("shared_authors"))
        .withColumnRenamed("a.subreddit", "sub_1")
        .withColumnRenamed("b.subreddit", "sub_2")
        .orderBy(F.desc("shared_authors"))
    )

    print("Total subreddit pairs:", pairs.count())

    # ---------------------------------------------------------
    # WRITE OUTPUT
    # ---------------------------------------------------------
    output_path = f"{s3_output_root}/subreddit_overlap_pairs"
    print("\nWROTE TO:", output_path)

    pairs.write.mode("overwrite").option("header", True).csv(output_path)

    # ---------------------------------------------------------
    print("\nSUBREDDIT OVERLAP ANALYSIS COMPLETED.")
    print("Outputs saved under:", s3_output_root)

    spark.stop()


if __name__ == "__main__":
    main()
