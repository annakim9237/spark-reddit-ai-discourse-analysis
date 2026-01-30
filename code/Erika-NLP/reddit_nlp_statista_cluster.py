"""
file: reddit_nlp_statista_cluster.py

Reddit AI/ML Community Analysis vs AI Coding Benefits (CLUSTER VERSION)

This script (cluster mode):

- Connects to a Spark cluster (Spark master URL passed as arg or via MASTER_PRIVATE_IP).
- Reads a Statista Excel file with AI coding benefits survey results.
- Reads RAW Reddit comments OR submissions parquet from S3.
- Performs Spark NLP preprocessing (cleaning + tokenization + lemmatization) ON CLUSTER.
- Uses tokens to detect which comments/submissions mention each benefit category.
- Computes % of Reddit rows mentioning each benefit category.
- Visualizes survey % vs Reddit % per benefit category.
- Copies visualization(s) to ~/reddit-nlp-cluster for easy scp.

S3 Location (raw Reddit):
  s3a://{NETID}-dsan6000-datasets-final/project/reddit/parquet/
    - comments/
    - submissions/

Date: December 2025
"""

import argparse
import os
import sys
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import TimestampType

import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, LemmatizerModel
from pyspark.ml import Pipeline

# Visualization libraries
import matplotlib
matplotlib.use("Agg")  # For headless environments (EC2, cluster)
import matplotlib.pyplot as plt


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
# BENEFIT CATEGORIES & KEYWORDS
# ================================================================
BENEFIT_CATEGORIES = [
    "Less time searching",
    "Faster coding",
    "Repetitive tasks",
    "Increased productivity",
    "Faster learning",
    "Less mental effort",
    "Better experience",
    "Better code quality",
]

STATISTA_TO_CANONICAL = {
    "Less time spent searching for information": "Less time searching",
    "Faster coding and development": "Faster coding",
    "Faster completion of repetitive tasks": "Repetitive tasks",
    "Increased productivity": "Increased productivity",
    "Faster learning of new technologies, frameworks, languages, etc.": "Faster learning",
    "Less mental effort required for coding and development": "Less mental effort",
    "Better coding and development experience": "Better experience",
    "Better quality of code and development solutions": "Better code quality",
    # "Other" is intentionally ignored
}


# Simple keyword heuristics for each category (using tokens)
BENEFIT_KEYWORDS = {
    "Less time searching": [
        "search", "searching", "googling", "google", "stackoverflow",
        "docs", "documentation",
    ],
    "Faster coding": [
        "faster", "fast", "speed", "speedup", "quickly",
        "autocomplete", "boilerplate",
    ],
    "Repetitive tasks": [
        "repetitive", "boilerplate", "tedious", "repeating",
        "repeat", "copy", "paste",
    ],
    "Increased productivity": [
        "productivity", "productive", "more", "efficient", "efficiency",
        "workflow", "get", "done",
    ],
    "Faster learning": [
        "learn", "learning", "learned", "tutorial",
        "explain", "explanation", "walkthrough",
    ],
    "Less mental effort": [
        "mental", "effort", "brain", "tiring",
        "thinking", "cognitive", "overwhelming",
    ],
    "Better experience": [
        "experience", "developer", "dev", "dx", "user", "ux",
    ],
    "Better code quality": [
        "code", "quality", "cleaner", "clean", "refactor",
        "refactoring", "bugs", "bugfree",
    ],
}


def category_to_col(cat: str) -> str:
    """
    Convert a human-readable category name into a safe column name.
    e.g. "Less time searching" -> "cat_less_time_searching"
    """
    slug = re.sub(r"[^a-z0-9]+", "_", cat.lower()).strip("_")
    return f"cat_{slug}"


# ================================================================
# SPARK SESSION BUILDER (CLUSTER)
# ================================================================
def build_spark(master_url: str, app_name: str = "Reddit_Statista_Cluster") -> SparkSession:
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
            "com.amazonaws.auth.InstanceProfileCredentialsProvider",
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
# STATISTA BENEFITS LOADER
# ================================================================
def load_statista_benefits_data(file_path: str) -> pd.DataFrame:
    """
    Load AI coding benefits survey data from a Statista-style Excel file.

    Expected Excel structure (Statista export):

    - Sheet name: "Data"
    - Data starts at row index ~5 (0-based) -> we slice from row 5 onwards
    - We keep:
        col 1 -> benefit description
        col 2 -> percentage value
    """
    logger.info(f"Loading benefits survey from: {file_path}")

    try:
        df = pd.read_excel(file_path, sheet_name="Data", header=None)
    except ImportError as e:
        logger.error(
            "Pandas Excel engine missing. Install 'openpyxl' in your environment.\n"
            "Example: uv add openpyxl  (or) pip install openpyxl"
        )
        raise
    except Exception as e:
        logger.error(f"Failed to read Excel file {file_path}: {e}", exc_info=True)
        raise

    # Adjust slice if your file structure differs
    data = df.iloc[5:, [1, 2]].copy()
    data.columns = ["benefit", "percentage"]
    data["percentage"] = pd.to_numeric(data["percentage"], errors="coerce")

    data = data.dropna().reset_index(drop=True)
    logger.info(f"✓ Loaded {len(data)} benefit categories from Excel.")
    logger.info("Benefit rows loaded:")
    logger.info("\n" + data.to_string(index=False))

    return data


# ================================================================
# LOAD RAW REDDIT PARQUET
# ================================================================
def load_reddit_data(
    spark: SparkSession,
    s3_path: str,
    data_type: str,
    sample_fraction: float | None = None,
):
    """
    Load RAW Reddit data from S3.

    Args:
        spark: SparkSession
        s3_path: Base S3 path to parquet/ (with comments/ and submissions/)
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
        text_col: Name of text column ('body' or 'text')

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
            r"https?://\S+|www\.\S+|bit\.ly/\S+|redd\.it/\S+",
            "",
        ),
    )

    # Remove [deleted], [removed]
    df = df.withColumn(
        cleaned_col,
        F.regexp_replace(F.col(cleaned_col), r"\[deleted\]|\[removed\]", ""),
    )

    # Markdown links [text](url) -> text
    df = df.withColumn(
        cleaned_col,
        F.regexp_replace(F.col(cleaned_col), r"\[([^\]]+)\]\([^\)]+\)", r"$1"),
    )

    # Username mentions u/username
    df = df.withColumn(
        cleaned_col,
        F.regexp_replace(F.col(cleaned_col), r"u/\w+", ""),
    )

    # Subreddit mentions r/subreddit
    df = df.withColumn(
        cleaned_col,
        F.regexp_replace(F.col(cleaned_col), r"r/\w+", ""),
    )

    # Collapse whitespace + trim
    df = df.withColumn(
        cleaned_col,
        F.trim(F.regexp_replace(F.col(cleaned_col), r"\s+", " ")),
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
# MAIN PREPROCESSING FUNCTION (INLINE, NO WRITE)
# ================================================================
def preprocess_reddit(
    spark: SparkSession,
    s3_input_path: str,
    data_type: str,
    sample_fraction: float | None = None,
    remove_stopwords: bool = True,
    apply_lemmatization: bool = True,
):
    """
    Preprocess RAW Reddit comments or submissions, returning a DF with tokens.
    Does NOT write to S3; used inline for Statista comparison.
    """
    logger.info("=" * 80)
    logger.info(f"PREPROCESSING {data_type.upper()} (CLUSTER INLINE)")
    logger.info("=" * 80)

    df = load_reddit_data(spark, s3_input_path, data_type, sample_fraction)

    if data_type == "comments":
        text_col = "body"
        if text_col not in df.columns:
            raise ValueError(f"Missing column: {text_col}")

        logger.info("\nSample comments BEFORE preprocessing:")
        df.select("subreddit", text_col).show(3, truncate=80)

        df_cleaned = clean_reddit_text(df, text_col)

        logger.info("\nSample comments AFTER cleaning:")
        df_cleaned.select("subreddit", text_col, f"{text_col}_cleaned").show(3, truncate=80)

        text_col_to_process = f"{text_col}_cleaned"

    else:  # submissions
        if "title" not in df.columns:
            raise ValueError("Missing column: title")

        logger.info("\nCombining 'title' and 'selftext' columns...")
        if "selftext" in df.columns:
            df = df.withColumn(
                "text",
                F.concat_ws(" ", F.col("title"), F.coalesce(F.col("selftext"), F.lit(""))),
            )
        else:
            df = df.withColumn("text", F.col("title"))

        logger.info("\nSample submissions BEFORE preprocessing:")
        df.select("subreddit", "title", "selftext").show(3, truncate=80)

        text_col = "text"
        df_cleaned = clean_reddit_text(df, text_col)

        logger.info("\nSample submissions AFTER cleaning:")
        df_cleaned.select("subreddit", "title", f"{text_col}_cleaned").show(3, truncate=80)

        text_col_to_process = f"{text_col}_cleaned"

    pipeline = build_preprocessing_pipeline(
        input_col=text_col_to_process,
        remove_stopwords=remove_stopwords,
        apply_lemmatization=apply_lemmatization,
    )

    logger.info("\nFitting pipeline...")
    pipeline_model = pipeline.fit(df_cleaned)

    logger.info("Transforming data...")
    df_processed = pipeline_model.transform(df_cleaned)

    # Add token_count and processed_text
    df_processed = df_processed.withColumn("token_count", F.size(F.col("tokens")))
    df_processed = df_processed.withColumn("processed_text", F.array_join(F.col("tokens"), " "))

    # Select useful columns
    if data_type == "comments":
        columns_to_keep = [
            "id",
            "subreddit",
            "author",
            "body",
            "body_cleaned",
            "tokens",
            "processed_text",
            "token_count",
            "score",
            "created_utc",
            "parent_id",
            "link_id",
            "controversiality",
            "gilded",
        ]
    else:  # submissions
        columns_to_keep = [
            "id",
            "subreddit",
            "author",
            "title",
            "selftext",
            "text_cleaned",
            "tokens",
            "processed_text",
            "token_count",
            "score",
            "created_utc",
            "num_comments",
            "url",
            "over_18",
        ]

    columns_to_keep = [c for c in columns_to_keep if c in df_processed.columns]
    df_final = df_processed.select(*columns_to_keep)

    logger.info("\nFinal preprocessed data sample:")
    df_final.show(3, truncate=80, vertical=True)

    logger.info("\nPreprocessing Statistics:")
    stats = df_final.select(
        F.count("subreddit").alias("total_records"),
        F.avg("token_count").alias("avg_tokens"),
        F.min("token_count").alias("min_tokens"),
        F.max("token_count").alias("max_tokens"),
    ).collect()[0]

    logger.info(f"  Total records: {stats['total_records']:,}")
    logger.info(f"  Average tokens: {stats['avg_tokens']:.2f}")
    logger.info(f"  Min tokens: {stats['min_tokens']}")
    logger.info(f"  Max tokens: {stats['max_tokens']}")

    return df_final


# ================================================================
# MONTHLY AGGREGATION (OPTIONAL DIAGNOSTIC)
# ================================================================
def aggregate_reddit_activity_monthly(df):
    """
    Aggregate Reddit activity to a monthly grain.

    Requires:
      - created_utc (UNIX timestamp)
      - author
      - score
      - token_count (from preprocessing)
    """
    logger.info("Aggregating monthly Reddit activity...")

    required = {"created_utc", "author", "score", "token_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for monthly aggregation: {missing}")

    df_ = df.withColumn(
        "timestamp", F.from_unixtime(F.col("created_utc")).cast(TimestampType())
    ).withColumn(
        "year_month", F.date_format("timestamp", "yyyy-MM")
    )

    monthly = df_.groupBy("year_month").agg(
        F.count("*").alias("comment_count"),
        F.countDistinct("author").alias("unique_authors"),
        F.avg("token_count").alias("avg_tokens_per_comment"),
        F.avg("score").alias("avg_score"),
    )

    pdf = monthly.toPandas()
    pdf["date"] = pd.to_datetime(pdf["year_month"] + "-01")
    pdf = pdf.sort_values("date")

    logger.info(f"✓ Aggregated into {len(pdf)} months.")
    return pdf


# ================================================================
# NLP: ADD BENEFIT FLAGS TO EACH ROW
# ================================================================
def add_benefit_flags(df):
    """
    For each benefit category, add a boolean column:
        cat_less_time_searching, cat_faster_coding, ...

    A row is True for a category if any of that category's keywords
    appear as exact tokens.
    """
    if "tokens" not in df.columns:
        raise ValueError("Expected 'tokens' column from preprocessing, but it was not found.")

    for cat, keywords in BENEFIT_KEYWORDS.items():
        col_name = category_to_col(cat)

        cond = None
        for kw in keywords:
            this = F.array_contains(F.col("tokens"), kw)
            cond = this if cond is None else (cond | this)

        if cond is None:
            cond = F.lit(False)

        df = df.withColumn(col_name, cond)

    return df


# ================================================================
# COMPUTE REDDIT PERCENTAGES PER BENEFIT
# ================================================================
def compute_reddit_benefit_percentages(df):
    """
    Compute % of Reddit rows that mention each benefit category.
    """
    total = df.count()
    if total == 0:
        logger.warning("No Reddit rows available; returning zeros for all categories.")
        return {cat: 0.0 for cat in BENEFIT_CATEGORIES}

    agg_exprs = []
    for cat in BENEFIT_CATEGORIES:
        col_name = category_to_col(cat)
        agg_exprs.append(F.sum(F.col(col_name).cast("int")).alias(col_name))

    row = df.agg(*agg_exprs).collect()[0]

    reddit_pct = {}
    for cat in BENEFIT_CATEGORIES:
        col_name = category_to_col(cat)
        count_cat = row[col_name]
        reddit_pct[cat] = (count_cat / total) * 100.0 if total else 0.0

    logger.info("Reddit benefit category percentages (by row):")
    for cat in BENEFIT_CATEGORIES:
        logger.info(f"  {cat}: {reddit_pct[cat]:.2f}% of rows")

    return reddit_pct


# ================================================================
# VISUALIZATION: SURVEY vs REDDIT (%)
# ================================================================
def visualize_benefits_two_row(benefits_df, reddit_pct_map, out_path, title_suffix=""):
    """
    Create a two-row horizontal bar chart:

    - Top row: Statista survey percentages per AI coding benefit.
    - Bottom row: % of Reddit rows that mention each benefit (via tokens).
    """
    logger.info("Creating Visualization (Survey vs Reddit Mentions)...")

    # Map Statista's long benefit labels -> our canonical category names
    df = benefits_df.copy()
    # Make sure benefit is plain strings (not categoricals)
    df["benefit"] = df["benefit"].astype(str)
    df["canonical"] = df["benefit"].map(STATISTA_TO_CANONICAL)

    # Keep only rows that map to one of our canonical categories
    df = df[df["canonical"].isin(BENEFIT_CATEGORIES)].copy()

    if df.empty:
        logger.warning("No matching benefit rows after mapping; check STATISTA_TO_CANONICAL.")
        # Build an empty frame with zeros so the plot still runs
        df = pd.DataFrame({
            "benefit": BENEFIT_CATEGORIES,
            "percentage": [0.0] * len(BENEFIT_CATEGORIES),
            "reddit_pct": [reddit_pct_map.get(cat, 0.0) for cat in BENEFIT_CATEGORIES],
        })
    else:
        # Aggregate in case there are multiple rows per canonical category
        df = (
            df.groupby("canonical", as_index=False)["percentage"]
              .mean()  # or sum(), depending on Statista semantics
        )

        # Reindex to ensure we always have rows in the BENEFIT_CATEGORIES order
        df = (
            df.set_index("canonical")
              .reindex(BENEFIT_CATEGORIES)
              .reset_index()
              .rename(columns={"canonical": "benefit"})
        )

        # Clean up dtypes
        logger.info("Cleaning percentage data types...")
        df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce").fillna(0.0)

        # IMPORTANT: break categorical → go to float BEFORE fillna
        reddit_pct = df["benefit"].astype(str).map(reddit_pct_map)  # map using strings
        reddit_pct = reddit_pct.astype("float64")                   # force float
        reddit_pct = reddit_pct.fillna(0.0)                         # now fill NaNs
        df["reddit_pct"] = reddit_pct


    # ---- Plotting (no categoricals, just strings + floats) ----
    y = np.arange(len(df))
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(15, 10))

    # Top row = Survey (Statista)
    ax.barh(
        y - bar_h / 2,
        df["percentage"],
        height=bar_h,
        color="#3A86FF",
        edgecolor="black",
        label="Survey Benefits (2024)",
    )

    # Bottom row = Reddit (actual percentages)
    ax.barh(
        y + bar_h / 2,
        df["reddit_pct"],
        height=bar_h,
        color="#FF006E",
        edgecolor="black",
        label="Reddit Rows Mentioning Benefit",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["benefit"], fontsize=12)
    ax.set_xlabel("Percentage (%)", fontsize=12)
    title = "AI Coding Benefits: Survey vs Reddit Discussion"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title, fontsize=16)

    # Labels on bars
    for i, (s, r) in enumerate(zip(df["percentage"], df["reddit_pct"])):
        ax.text(float(s) + 0.5, i - bar_h / 2, f"{s:.0f}%", va="center")
        ax.text(float(r) + 0.5, i + bar_h / 2, f"{r:.1f}%", va="center")

    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Saved visualization to: {out_path}")




# ================================================================
# DROP FOLDER COPY (FOR SCP)
# ================================================================
def copy_to_drop_folder(files: list[str]):
    """Copy visualization files to ~/reddit-nlp-cluster for easy scp."""
    drop = Path.home() / "reddit-nlp-cluster"
    drop.mkdir(exist_ok=True)
    for f in files:
        if os.path.exists(f):
            shutil.copy2(f, drop / Path(f).name)
            logger.info(f"  Copied {f} to {drop}/")


# ================================================================
# MAIN
# ================================================================
NETID = os.getenv("NETID", "ea973")
def main():
    parser = argparse.ArgumentParser(
        description="Reddit AI/ML Community vs AI Coding Benefits (CLUSTER, INLINE NLP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Comments only, full dataset, using MASTER_PRIVATE_IP env
  export MASTER_PRIVATE_IP=10.0.1.5
  python reddit_nlp_statista_cluster.py \\
      --data-type comments \\
      --benefits ai-tools-ways-of-coding-and-development-enhancements-globally-2024.xlsx

  # Submissions only, 5% sample, explicit master URL, no lemmatization
  python reddit_nlp_statista_cluster.py spark://10.0.1.5:7077 \\
      --data-type submissions \\
      --benefits ai-tools-ways-of-coding-and-development-enhancements-globally-2024.xlsx \\
      --sample 0.05 \\
      --no-lemmatization
        """,
    )

    parser.add_argument(
        "master_url",
        nargs="?",
        default=None,
        help="Spark master URL (e.g., spark://10.0.1.5:7077)",
    )

    parser.add_argument(
        "--benefits",
        required=True,
        help="Path to Statista benefits Excel (AI coding benefits survey)",
    )

    parser.add_argument(
        "--s3-input",
        default=f"s3a://{NETID}-dsan6000-datasets-final/project/reddit/parquet/",
        help="Base S3A path containing raw parquet/ (comments/ and submissions/)",
    )

    parser.add_argument(
        "--data-type",
        choices=["comments", "submissions"],
        default="comments",
        help="Which Reddit data to use (default: comments)",
    )

    parser.add_argument(
        "--sample",
        type=float,
        help="Optional sample fraction in (0, 1] for Reddit rows",
    )

    parser.add_argument(
        "--no-stopwords",
        action="store_true",
        help="Disable stop words removal in preprocessing",
    )

    parser.add_argument(
        "--no-lemmatization",
        action="store_true",
        help="Disable lemmatization in preprocessing",
    )

    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Directory for outputs (PNG visualization)",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    master_url = args.master_url
    if not master_url:
        master_private_ip = os.getenv("MASTER_PRIVATE_IP")
        if master_private_ip:
            master_url = f"spark://{master_private_ip}:7077"
        else:
            print("=" * 70)
            print("❌ Error: Master URL not provided")
            print("Usage: python reddit_nlp_statista_cluster.py spark://MASTER_IP:7077 [options]")
            print("   or: export MASTER_PRIVATE_IP=xxx.xxx.xxx.xxx")
            print("=" * 70)
            return 1

    if args.sample and not (0 < args.sample <= 1):
        parser.error("Sample fraction must be between 0 and 1")

    logger.info("=" * 70)
    logger.info("REDDIT AI/ML COMMUNITY vs AI CODING BENEFITS (CLUSTER MODE)")
    logger.info("=" * 70)
    logger.info(f"Spark Master    : {master_url}")
    logger.info(f"Benefits Excel  : {args.benefits}")
    logger.info(f"S3 Input Base   : {args.s3_input}")
    logger.info(f"Data Type       : {args.data_type}")
    logger.info(f"Sample Fraction : {args.sample if args.sample else 'FULL DATASET'}")
    logger.info(f"Output Dir      : {args.output_dir}")
    logger.info(f"Remove Stopwords   : {not args.no_stopwords}")
    logger.info(f"Apply Lemmatization: {not args.no_lemmatization}")
    logger.info("=" * 70)

    spark = None
    start_time = datetime.now()

    try:
        spark = build_spark(master_url)

        # 1. Load Statista benefits file (local to driver)
        benefits_df = load_statista_benefits_data(args.benefits)

        # 2. Preprocess Reddit (comments OR submissions) on cluster
        reddit_df = preprocess_reddit(
            spark,
            args.s3_input,
            args.data_type,
            sample_fraction=args.sample,
            remove_stopwords=not args.no_stopwords,
            apply_lemmatization=not args.no_lemmatization,
        )

        # 3. Add benefit flags and compute Reddit percentages
        reddit_df = add_benefit_flags(reddit_df)
        reddit_pct_map = compute_reddit_benefit_percentages(reddit_df)

        # Monthly aggregation just for diagnostics/logging
        try:
            monthly_df = aggregate_reddit_activity_monthly(reddit_df)
            logger.info("Head of monthly Reddit activity (diagnostic):")
            logger.info("\n" + monthly_df.head().to_string(index=False))
        except Exception as e:
            logger.warning(f"Monthly aggregation skipped (missing cols?): {e}")

        # 4. Visualization: Survey vs Reddit mentions
        fname = f"reddit_{args.data_type}_vs_survey_benefits.png"
        output_path = os.path.join(args.output_dir, fname)
        visualize_benefits_two_row(benefits_df, reddit_pct_map, output_path, title_suffix=f"({args.data_type.capitalize()})")

        # Copy to drop folder for easy scp
        copy_to_drop_folder([output_path])

        elapsed = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 70)
        print("✅ CLUSTER ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"Total execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"Visualization saved to: {output_path}")
        print("Visualization copied to: ~/reddit-nlp-cluster/")
        print("=" * 70)

        logger.info("\n" + "=" * 80)
        logger.info("✅ CLUSTER ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info(f"Total execution time: {elapsed:.1f} seconds")
        logger.info(f"Visualization saved to: {output_path}")
        logger.info("Visualization copied to: ~/reddit-nlp-cluster/")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\n❌ Error during cluster analysis: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1

    finally:
        if spark:
            spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    sys.exit(main())
