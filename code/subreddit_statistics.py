"""
Milestone 0: Data Acquisition and Initial Filtering

2. Dataset Statistics (saved to data/csv/ folder)

Output: data/csv/subreddit_statistics.csv
Per-subreddit breakdown:
    `subreddit`
    `num_comments`
    `num_submissions`
    `total_rows`
    `avg_score`
    `date_range`

Author: Mandy Sun 
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys


def main():
    """
    Executes the subreddit statistics analysis for the Reddit dataset.

    Function:
        - Initializes a Spark session.
        - Loads Reddit comment and submission parquet files from S3 or local paths.
        - Converts Unix timestamps to calendar dates.
        - Computes per-subreddit aggregates
        - Writes the results to data/csv/subreddit_statistics.csv.

    Args:
        netid (str):
            Command-line argument. Used to construct S3 bucket paths for loading data.

    Returns:
        None:
            Output is written to disk. No values are returned.

    Errors:
        - Raises ValueError if the <netid> argument is missing.
        - Raises Spark or IO errors if parquet paths are invalid or inaccessible.
    """

    if len(sys.argv) < 2:
        raise ValueError("Usage: python subreddit_statistics.py <netid>")

    netid = sys.argv[1]

    # Starts spark session 
    spark = (
        SparkSession.builder
        .appName("SubredditStatistics")
        .getOrCreate()
    )

    # Input paths on S3 
    comments_path = f"s3a://{netid}-dsan6000-datasets/project/reddit/parquet/comments/" 
    submissions_path = f"s3a://{netid}-dsan6000-datasets/project/reddit/parquet/submissions/"
    
    # Local-mode testing (optional):
    # comments_path = "/home/ubuntu/dsan6000/fall-2025-project-team05/local_data/comments"
    # submissions_path = "/home/ubuntu/dsan6000/fall-2025-project-team05/local_data/submissions"

    # Load datasets
    comments = spark.read.parquet(comments_path)
    submissions = spark.read.parquet(submissions_path)

    # Convert created_utc (epoch seconds) → timestamp → date
    comments = comments.withColumn(
        "date",
        F.to_date(F.from_unixtime(F.col("created_utc")))
    )

    submissions = submissions.withColumn(
        "date",
        F.to_date(F.from_unixtime(F.col("created_utc")))
    )

    # Per-subreddit: comment stats
    comment_stats = (
        comments
        .groupBy("subreddit")
        .agg(
            F.count("*").alias("num_comments"),
            F.avg("score").alias("avg_comment_score"),
            F.min("date").alias("first_comment_date"),
            F.max("date").alias("last_comment_date")
        )
    )

    # Per-subreddit: submission stats
    submission_stats = (
        submissions
        .groupBy("subreddit")
        .agg(
            F.count("*").alias("num_submissions"),
            F.avg("score").alias("avg_submission_score"),
            F.min("date").alias("first_submission_date"),
            F.max("date").alias("last_submission_date")
        )
    )

    # Merge comment + submission stats per subreddit
    stats = (
        comment_stats
        .join(submission_stats, on="subreddit", how="outer")
        .withColumn(
            "total_rows",
            F.coalesce(F.col("num_comments"), F.lit(0)) +
            F.coalesce(F.col("num_submissions"), F.lit(0))
        )
        .withColumn(
            "avg_score",
            (
                F.coalesce(F.col("avg_comment_score"), F.lit(0)) +
                F.coalesce(F.col("avg_submission_score"), F.lit(0))
            ) / 2
        )
        .withColumn(
            "date_range",
            F.concat_ws(
                " to ",
                F.least(
                    F.col("first_comment_date"),
                    F.col("first_submission_date")
                ),
                F.greatest(
                    F.col("last_comment_date"),
                    F.col("last_submission_date")
                )
            )
        )
        .select(
            "subreddit",
            "num_comments",
            "num_submissions",
            "total_rows",
            "avg_score",
            "date_range"
        )
        .orderBy(F.desc("total_rows"))
    )

    # Save output
    stats.write.csv(
        "data/csv/subreddit_statistics.csv",
        header=True,
        mode="overwrite"
    )

    # Stops spark session 
    spark.stop()


if __name__ == "__main__":
    main()
