"""
Extract ALL posts from the top 1% authors per subreddit.

Input (parameterized):
    s3a://{netid}/project/reddit/comments/
    s3a://{netid}/project/reddit/submissions/

Output:
    s3a://{netid}/project/top1pct_posts_per_subreddit/
        subreddit=<name>/part-*.csv
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys


def main():

    # ---------------------------------------------------------
    # ARGUMENT CHECK
    # ---------------------------------------------------------
    if len(sys.argv) < 2:
        raise ValueError("Usage: spark-submit top1pct_posts_per_subreddit.py <netid>")

    netid = sys.argv[1]

    spark = (
        SparkSession.builder
        .appName(f"Top1pctPostsPerSubreddit_{netid}")
        .getOrCreate()
    )

    print("SPARK MASTER:", spark.sparkContext.master)

    # ---------------------------------------------------------
    # LOAD PATHS
    # ---------------------------------------------------------
    base_path = f"s3a://{netid}/project"

    comments_path = f"{base_path}/reddit/comments/"
    submissions_path = f"{base_path}/reddit/submissions/"
    output_root = f"{base_path}/top1pct_posts_per_subreddit"

    print("\nComments PATH:", comments_path)
    print("Submissions PATH:", submissions_path)
    print("Output PATH:", output_root, "\n")

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------
    comments = spark.read.parquet(comments_path).withColumn("data_type", F.lit("comment"))
    submissions = spark.read.parquet(submissions_path).withColumn("data_type", F.lit("submission"))

    df = comments.unionByName(submissions, allowMissingColumns=True).cache()

    print("Total rows loaded:", df.count())

    # ---------------------------------------------------------
    # COUNT POSTS PER AUTHOR *IN EACH SUBREDDIT*
    # ---------------------------------------------------------
    author_counts = (
        df.groupBy("subreddit", "author")
          .agg(F.count("*").alias("post_count"))
    )

    # ---------------------------------------------------------
    # COMPUTE 99TH PERCENTILE PER SUBREDDIT
    # ---------------------------------------------------------
    pct = (
        author_counts.groupBy("subreddit")
        .agg(F.expr("percentile_approx(post_count, 0.99)").alias("p99"))
    )

    # ---------------------------------------------------------
    # SELECT TOP 1% AUTHORS PER SUBREDDIT
    # ---------------------------------------------------------
    top1 = (
        author_counts.join(pct, "subreddit")
                     .filter(F.col("post_count") >= F.col("p99"))
                     .select("subreddit", "author")
                     .distinct()
    )

    print("Top 1% author rows:", top1.count())

    # ---------------------------------------------------------
    # EXTRACT ALL POSTS FROM THESE AUTHORS
    # ---------------------------------------------------------
    final_posts = (
        df.join(top1, ["subreddit", "author"], "inner")
          .select("subreddit", "author", "title", "body", "score", "created_utc")
    )

    print("Final extracted posts:", final_posts.count())

    # ---------------------------------------------------------
    # WRITE OUTPUT
    # ---------------------------------------------------------
    final_posts.write \
        .mode("overwrite") \
        .partitionBy("subreddit") \
        .option("header", True) \
        .csv(output_root)

    print("\n✓ WROTE all top 1% posts per subreddit to:")
    print(output_root)

    spark.stop()


if __name__ == "__main__":
    main()
