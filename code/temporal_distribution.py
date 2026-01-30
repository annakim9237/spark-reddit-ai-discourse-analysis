"""
Milestone 0: Temporal Distribution Only

3. Temporal Distribution (saved to data/csv/ folder)

Output:
- data/csv/temporal_distribution.csv
  Columns:
      year_month
      num_comments
      num_submissions
      total_rows

Author: Mandy Sun
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys


def main():
    """
    Executes the temporal distribution analysis for Reddit comments and submissions.

    Function:
        - Initializes a Spark session.
        - Loads Reddit data from either S3 or local parquet files.
        - Converts Unix timestamps into calendar dates and formatted year-month strings.
        - Aggregates the number of comments and submissions per year-month.
        - Merges both aggregates and computes total activity per month.
        - Writes the result to data/csv/temporal_distribution.csv as Spark output files.

    Args:
        netid (str):
            Passed from the command line as sys.argv[1].
            Used to construct S3 input paths for full dataset runs.

    Returns:
        None:
            The function writes output to disk but does no in-memory object.

    Errors:
        - Raises ValueError if the <netid> argument is missing.
        - May raise FileNotFoundError / AnalysisException if input paths are incorrect.
    """

    if len(sys.argv) < 2:
        raise ValueError("Usage: spark-submit temporal_distribution.py <netid>")

    netid = sys.argv[1]

    spark = (
        SparkSession.builder
        .appName("TemporalDistribution")
        .getOrCreate()
    )

    # Input data paths 
    comments_path = f"s3a://{netid}-dsan6000-datasets/project/reddit/parquet/comments/"
    submissions_path = f"s3a://{netid}-dsan6000-datasets/project/reddit/parquet/submissions/"

    # Local-mode testing (optional):
    # comments_path = "/home/ubuntu/dsan6000/fall-2025-project-team05/local_data/comments"
    # submissions_path = "/home/ubuntu/dsan6000/fall-2025-project-team05/local_data/submissions"

    # Load 
    comments = spark.read.parquet(comments_path)
    submissions = spark.read.parquet(submissions_path)

    # Convert timestamp to date + year_month
    comments = (
        comments
        .withColumn("date", F.to_date(F.from_unixtime(F.col("created_utc"))))
        .withColumn("year_month", F.date_format("date", "yyyy-MM"))
    )

    submissions = (
        submissions
        .withColumn("date", F.to_date(F.from_unixtime(F.col("created_utc"))))
        .withColumn("year_month", F.date_format("date", "yyyy-MM"))
    )

    # Aggregate
    temporal_comments = (
        comments
        .groupBy("year_month")
        .agg(F.count("*").alias("num_comments"))
    )

    temporal_submissions = (
        submissions
        .groupBy("year_month")
        .agg(F.count("*").alias("num_submissions"))
    )

    # Merge + compute total_rows
    temporal = (
        temporal_comments
        .join(temporal_submissions, on="year_month", how="outer")
        .withColumn("num_comments", F.coalesce(F.col("num_comments"), F.lit(0)))
        .withColumn("num_submissions", F.coalesce(F.col("num_submissions"), F.lit(0)))
        .withColumn("total_rows", F.col("num_comments") + F.col("num_submissions"))
        .orderBy("year_month")
    )

    # Save output
    temporal.write.csv(
        "data/csv/temporal_distribution.csv",
        header=True,
        mode="overwrite"
    )

    # Stop spark session 
    spark.stop()


if __name__ == "__main__":
    main()


