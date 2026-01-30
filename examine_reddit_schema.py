#!/usr/bin/env python3
"""
Examine Reddit parquet file schema and sample data.

This script reads Reddit comments and submissions parquet files and displays:
- Complete schema with data types
- Sample rows
- Interesting columns for analysis

PREREQUISITES:
Before running this script, download sample files to /tmp/:
See SCHEMA_EXAMINATION.md for complete instructions.

Quick download commands:
  aws s3 cp s3://dsan6000-datasets/reddit/parquet/comments/yyyy=2024/mm=01/comments_RC_2024-01.zst_97.parquet \
    /tmp/reddit_comments_sample.parquet --request-payer requester

  aws s3 cp s3://dsan6000-datasets/reddit/parquet/submissions/yyyy=2024/mm=01/submissions_RS_2024-01.zst_1.parquet \
    /tmp/reddit_submissions_sample.parquet --request-payer requester
"""

import logging
from typing import (
    Dict,
    List,
)

import polars as pl


# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)


def _examine_parquet_schema(
    file_path: str,
    data_type: str,
) -> None:
    """
    Examine and display parquet file schema and sample data.

    Args:
        file_path: Path to the parquet file
        data_type: Type of data (comments or submissions)
    """
    logging.info(f"\n{'=' * 80}")
    logging.info(f"Examining {data_type.upper()} data")
    logging.info(f"File: {file_path}")
    logging.info(f"{'=' * 80}\n")

    try:
        df = pl.read_parquet(file_path)

        logging.info(f"Total rows: {len(df):,}")
        logging.info(f"Total columns: {len(df.columns)}\n")

        logging.info("SCHEMA:")
        logging.info("-" * 80)
        for col_name, col_type in zip(df.columns, df.dtypes):
            logging.info(f"  {col_name:30s} {str(col_type)}")

        logging.info("\n" + "=" * 80)
        logging.info("SAMPLE DATA (First 3 rows):")
        logging.info("=" * 80)
        print(df.head(3))

        logging.info("\n" + "=" * 80)
        logging.info("INTERESTING COLUMNS FOR ANALYSIS:")
        logging.info("=" * 80)

        interesting_cols = _identify_interesting_columns(df, data_type)
        for col in interesting_cols:
            if col in df.columns:
                logging.info(f"  - {col}")

        if data_type == "comments":
            _show_comment_samples(df, interesting_cols)
        else:
            _show_submission_samples(df, interesting_cols)

    except Exception as e:
        logging.error(f"Error examining {data_type}: {e}")
        raise


def _identify_interesting_columns(
    df: pl.DataFrame,
    data_type: str,
) -> List[str]:
    """
    Identify columns typically of interest for analysis.

    Args:
        df: DataFrame to examine
        data_type: Type of data (comments or submissions)

    Returns:
        List of interesting column names
    """
    common_cols = [
        "subreddit",
        "author",
        "score",
        "created_utc",
        "id",
    ]

    if data_type == "comments":
        specific_cols = [
            "body",
            "parent_id",
            "link_id",
            "controversiality",
            "gilded",
        ]
    else:
        specific_cols = [
            "title",
            "selftext",
            "url",
            "num_comments",
            "upvote_ratio",
            "over_18",
            "link_flair_text",
        ]

    all_cols = common_cols + specific_cols
    return [col for col in all_cols if col in df.columns]


def _show_comment_samples(
    df: pl.DataFrame,
    interesting_cols: List[str],
) -> None:
    """
    Show sample comment data with interesting columns.

    Args:
        df: DataFrame with comment data
        interesting_cols: List of columns to display
    """
    logging.info("\n" + "=" * 80)
    logging.info("SAMPLE COMMENT DATA:")
    logging.info("=" * 80)

    available_cols = [col for col in interesting_cols if col in df.columns]
    sample = df.select(available_cols).head(2)

    for i in range(min(2, len(sample))):
        logging.info(f"\nComment {i + 1}:")
        logging.info("-" * 40)
        row = sample.row(i, named=True)
        for key, value in row.items():
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            logging.info(f"  {key}: {value}")


def _show_submission_samples(
    df: pl.DataFrame,
    interesting_cols: List[str],
) -> None:
    """
    Show sample submission data with interesting columns.

    Args:
        df: DataFrame with submission data
        interesting_cols: List of columns to display
    """
    logging.info("\n" + "=" * 80)
    logging.info("SAMPLE SUBMISSION DATA:")
    logging.info("=" * 80)

    available_cols = [col for col in interesting_cols if col in df.columns]
    sample = df.select(available_cols).head(2)

    for i in range(min(2, len(sample))):
        logging.info(f"\nSubmission {i + 1}:")
        logging.info("-" * 40)
        row = sample.row(i, named=True)
        for key, value in row.items():
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            logging.info(f"  {key}: {value}")


def main() -> None:
    """Main function to examine Reddit parquet files."""
    comments_file = "/tmp/reddit_comments_sample.parquet"
    submissions_file = "/tmp/reddit_submissions_sample.parquet"

    _examine_parquet_schema(comments_file, "comments")
    _examine_parquet_schema(submissions_file, "submissions")

    logging.info("\n" + "=" * 80)
    logging.info("SCHEMA EXAMINATION COMPLETE")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
