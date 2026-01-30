#!/usr/bin/env python3
"""
file: reddit_nlp_cleaning.py

Reddit Data Preprocessing Pipeline with Spark NLP - CLUSTER VERSION
Processes filtered Reddit comments and submissions from S3 on Spark cluster.

S3 Location: s3a://{NETID}-reddit-datasets/project/reddit/parquet/
  - comments/
  - submissions/

Usage (cluster):

  uv run python cluster-files/ml/reddit_nlp_cleaning.py \
      spark://<PRIVATE_IP>:7077 \
      --s3-input s3a://ea973-dsan6000-datasets-final/project/reddit/parquet/ \
      --data-type comments \
      --sample 0.03 \
      --output-parquet data/parquet/reddit_cleaned.parquet

Date: December 2025
"""

import argparse
import os
import logging
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, FloatType
from pyspark.ml import Pipeline

import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, LemmatizerModel

# Visualization imports (not used in segmentation pipeline, but kept for reuse)
import matplotlib
matplotlib.use('Agg')  # For headless environments (EC2, cluster)
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ================================================================
# LOGGING CONFIG
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Cache env variable once
MASTER_PRIVATE_IP = os.getenv("MASTER_PRIVATE_IP")


# ================================================================
# SPARK SESSION BUILDER (CLUSTER)
# ================================================================
def build_spark(master_url: str, app_name: str = "Reddit_Preprocessing_Cluster") -> SparkSession:
    """
    Initialize Spark session for CLUSTER execution.

    Uses Spark NLP 5.1.3:
    com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.3
    """
    logger.info(f"Initializing Spark session on cluster: {master_url}")
    logger.info(f"MASTER_PRIVATE_IP env: {MASTER_PRIVATE_IP}")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master_url)
        # Jars
        .config(
            "spark.jars.packages",
            ",".join([
                "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.3",
                "org.apache.hadoop:hadoop-aws:3.3.2",
                "com.amazonaws:aws-java-sdk-bundle:1.12.262",
            ])
        )
        # Memory / cores
        .config("spark.executor.instances", "3")
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")
        .config("spark.executor.memory", "5g")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        # S3
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.InstanceProfileCredentialsProvider"
        )
        # Timeouts
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "60s")
        # Performance
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        # UI (optional: pin to 4040)
        .config("spark.ui.port", "4040")
        # Helpful for jar download behind proxies
        .config("spark.driver.extraJavaOptions", "-Djava.net.useSystemProxies=true")
        .config("spark.executor.extraJavaOptions", "-Djava.net.useSystemProxies=true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    logger.info("✓ Spark cluster session created successfully")
    logger.info(f"  Spark version: {spark.version}")
    logger.info(f"  Spark NLP version: {sparknlp.version()}")

    return spark


# ================================================================
# LOAD REDDIT DATA
# ================================================================
def load_reddit_data(
    spark: SparkSession,
    s3_path: str,
    data_type: str,
    sample_fraction: float | None = None,
):
    """
    Load Reddit data from S3.

    Args:
        spark: SparkSession
        s3_path: Base S3 path
        data_type: 'comments' or 'submissions'
        sample_fraction: Optional sampling (e.g., 0.01 for 1%)

    Returns:
        Spark DataFrame
    """
    full_path = f"{s3_path.rstrip('/')}/{data_type}/"
    logger.info(f"Loading {data_type} from: {full_path}")

    try:
        df = spark.read.parquet(full_path)

        if sample_fraction and 0 < sample_fraction < 1:
            logger.info(f"Sampling {sample_fraction * 100:.2f}% of data...")
            df = df.sample(fraction=sample_fraction, seed=42)

        count = df.count()
        logger.info(f"Loaded {count:,} {data_type}")
        logger.info(f"Columns: {df.columns}")

        return df

    except Exception as e:
        logger.error(f"Error loading {data_type} from {full_path}: {e}", exc_info=True)
        raise


# ================================================================
# TEXT CLEANING
# ================================================================
def clean_reddit_text(df, text_col: str):
    """
    Clean Reddit-specific patterns using PySpark regex.

    Args:
        df: Input DataFrame
        text_col: Name of text column ('body' or 'selftext' or 'title')

    Returns:
        DataFrame with cleaned text column <text_col>_cleaned
    """
    logger.info(f"Cleaning text in column: {text_col}")

    cleaned_col = f"{text_col}_cleaned"

    # Remove URLs
    df = df.withColumn(
        cleaned_col,
        F.regexp_replace(
            F.col(text_col),
            r'https?://\S+|www\.\S+|bit\.ly/\S+|redd\.it/\S+',
            '',
        ),
    )

    # Remove [deleted], [removed]
    df = df.withColumn(
        cleaned_col,
        F.regexp_replace(F.col(cleaned_col), r'\[deleted\]|\[removed\]', ''),
    )

    # Markdown links [text](url) -> text
    df = df.withColumn(
        cleaned_col,
        F.regexp_replace(F.col(cleaned_col), r'\[([^\]]+)\]\([^\)]+\)', r'$1'),
    )

    # Username mentions u/username
    df = df.withColumn(
        cleaned_col,
        F.regexp_replace(F.col(cleaned_col), r'u/\w+', ''),
    )

    # Subreddit mentions r/subreddit
    df = df.withColumn(
        cleaned_col,
        F.regexp_replace(F.col(cleaned_col), r'r/\w+', ''),
    )

    # Collapse whitespace + trim
    df = df.withColumn(
        cleaned_col,
        F.trim(F.regexp_replace(F.col(cleaned_col), r'\s+', ' ')),
    )

    # Filter out very short texts
    initial_count = df.count()
    df = df.filter(F.length(F.col(cleaned_col)) > 10)
    final_count = df.count()

    logger.info(f"Filtered out {initial_count - final_count:,} short texts")
    logger.info(f"Remaining records: {final_count:,}")

    return df


# ================================================================
# SPARK NLP PREPROCESSING PIPELINE
# ================================================================
def build_preprocessing_pipeline(
    input_col: str = "body_cleaned",
    remove_stopwords: bool = True,
    apply_lemmatization: bool = True,
) -> Pipeline:
    """
    Build Spark NLP preprocessing pipeline.

    Returns:
        Spark ML Pipeline that outputs "tokens" (string array column).
    """
    logger.info("Building Spark NLP preprocessing pipeline...")

    stages = []

    # 1. Document Assembler
    document = (
        DocumentAssembler()
        .setInputCol(input_col)
        .setOutputCol("document")
    )
    stages.append(document)

    # 2. Tokenizer
    tokenizer = (
        Tokenizer()
        .setInputCols(["document"])
        .setOutputCol("token")
    )
    stages.append(tokenizer)

    # 3. Normalizer
    normalizer = (
        Normalizer()
        .setInputCols(["token"])
        .setOutputCol("normalized")
        .setLowercase(True)
        .setCleanupPatterns(["[^\\w\\s]"])  # strip punctuation
    )
    stages.append(normalizer)

    last_col = "normalized"

    # 4. StopWordsCleaner (optional)
    if remove_stopwords:
        stopwords_cleaner = (
            StopWordsCleaner()
            .setInputCols(["normalized"])
            .setOutputCol("cleanTokens")
            .setCaseSensitive(False)
        )
        stages.append(stopwords_cleaner)
        last_col = "cleanTokens"
        logger.info("  - Stop words removal: ENABLED")
    else:
        logger.info("  - Stop words removal: DISABLED")

    # 5. Lemmatizer (pretrained, optional)
    if apply_lemmatization:
        lemmatizer = (
            LemmatizerModel.pretrained("lemma_antbnc", "en")
            .setInputCols([last_col])
            .setOutputCol("lemmatized")
        )
        stages.append(lemmatizer)
        last_col = "lemmatized"
        logger.info("  - Lemmatization: ENABLED (lemma_antbnc)")
    else:
        logger.info("  - Lemmatization: DISABLED")

    # 6. Finisher -> "tokens" (array<string>)
    finisher = (
        Finisher()
        .setInputCols([last_col])
        .setOutputCols(["tokens"])
        .setCleanAnnotations(True)
    )
    stages.append(finisher)

    logger.info(f"Pipeline has {len(stages)} stages")
    return Pipeline(stages=stages)


# ================================================================
# CLI ARGUMENTS
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean Reddit comments/submissions with Spark NLP and write a cleaned parquet."
    )

    parser.add_argument(
        "master_url",
        nargs="?",
        help="Spark master URL (e.g. spark://<PRIVATE_IP>:7077). "
             "If omitted, use MASTER_PRIVATE_IP or local[*].",
    )

    parser.add_argument(
        "--s3-input",
        required=True,
        help="Base S3 path with parquet data (e.g. s3a://.../project/reddit/parquet/)",
    )

    parser.add_argument(
        "--data-type",
        choices=["comments", "submissions"],
        default="comments",
        help="Which dataset to load (default: comments).",
    )

    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Optional fraction of rows to sample (e.g. 0.03 for 3%%).",
    )

    parser.add_argument(
        "--output-parquet",
        required=True,
        help="Output parquet path (file or directory), e.g. data/parquet/reddit_cleaned.parquet",
    )

    return parser.parse_args()


# ================================================================
# MAIN
# ================================================================
def main():
    args = parse_args()

    # Resolve master URL
    master_url = args.master_url
    if not master_url:
        if MASTER_PRIVATE_IP:
            master_url = f"spark://{MASTER_PRIVATE_IP}:7077"
            logger.info(f"No master_url arg; using MASTER_PRIVATE_IP → {master_url}")
        else:
            master_url = "local[*]"
            logger.info("No master_url arg and no MASTER_PRIVATE_IP; using local[*].")

    spark = None
    try:
        # 1) Build Spark session
        spark = build_spark(master_url, app_name="Reddit_NLP_Cleaning_Export")

        # 2) Load raw Reddit data
        df = load_reddit_data(
            spark,
            s3_path=args.s3_input,
            data_type=args.data_type,
            sample_fraction=args.sample,
        )

        # Choose text column
        if args.data_type == "comments":
            text_col = "body"
        else:
            # Try sensible defaults for submissions
            if "selftext" in df.columns:
                text_col = "selftext"
            elif "body" in df.columns:
                text_col = "body"
            elif "title" in df.columns:
                text_col = "title"
            else:
                raise ValueError("Could not find a suitable text column in submissions dataset.")

        # 3) Clean raw text
        df = clean_reddit_text(df, text_col=text_col)
        cleaned_col = f"{text_col}_cleaned"

        # 4) Spark NLP pipeline → tokens
        pipeline = build_preprocessing_pipeline(
            input_col=cleaned_col,
            remove_stopwords=True,
            apply_lemmatization=True,
        )
        model = pipeline.fit(df)
        df = model.transform(df)

        # 5) Add token_count + processed_text
        df = df.withColumn("token_count", F.size(F.col("tokens")))
        df = df.withColumn("processed_text", F.array_join(F.col("tokens"), " "))

        # 6) Write cleaned parquet
        out_path = args.output_parquet
        parent = Path(out_path).parent
        if parent:
            parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing cleaned parquet to: {out_path}")
        df.write.mode("overwrite").parquet(out_path)
        logger.info("✅ Finished writing cleaned parquet.")

        return 0

    except Exception as e:
        logger.exception(f"Error in reddit_nlp_cleaning main: {e}")
        return 1

    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
