"""
file: reddit_nlp_bad_words_cluster.py

Reddit Data Preprocessing Pipeline with Spark NLP - CLUSTER VERSION
Processes filtered Reddit comments and submissions from S3 on Spark cluster.

S3 Location: s3a://{NETID}-reddit-datasets/project/reddit/parquet/
  - comments/
  - submissions/

Date: November 2025
"""

import argparse
import os
import sys
import logging
import shutil
from datetime import datetime
from pathlib import Path
from collections import Counter

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, FloatType
from pyspark.sql.functions import udf

import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, LemmatizerModel
from pyspark.ml import Pipeline

# Visualization imports
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
# BAD WORDS LOADER
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BAD_WORDS_PATH = os.path.join(SCRIPT_DIR, "bad_words_cleaned.txt")

def load_bad_words(file_path: str = BAD_WORDS_PATH) -> set:
    """
    Load bad words from text file.

    Returns:
        Set of bad words (lowercase).
    """
    logger.info(f"Loading bad words from: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            bad_words = {word.strip().lower() for word in f if word.strip()}

        logger.info(f"Loaded {len(bad_words)} bad words")
        return bad_words

    except Exception as e:
        logger.error(f"Error loading bad words file: {e}", exc_info=True)
        raise


# ================================================================
# BAD WORD ANALYSIS BY SUBREDDIT
# ================================================================

def analyze_bad_words_by_subreddit(df, bad_words: set, spark: SparkSession):
    """
    Fully Spark-native bad word analysis.
    No Python UDFs, no driver-side looping, safe for massive datasets.
    """

    logger.info("Analyzing bad words by subreddit (Spark-native)...")

    # Broadcast bad words
    bad_words_b = spark.sparkContext.broadcast(bad_words)

    # Explode token array (this is distributed)
    exploded = df.select(
        "subreddit",
        F.explode_outer("tokens").alias("token")
    )

    # Filter only bad words
    filtered = exploded.filter(
        F.lower(F.col("token")).isin([bw for bw in bad_words_b.value])
    )

    # Count total posts and tokens per subreddit
    posts_and_tokens = (
        df
        .groupBy("subreddit")
        .agg(
            F.count("*").alias("total_posts"),
            F.sum(F.size("tokens")).alias("total_tokens")
        )
    )

    # Count bad tokens per subreddit
    bad_counts = (
        filtered
        .groupBy("subreddit")
        .agg(F.count("*").alias("total_bad_words"))
    )

    # Join everything
    stats = (
        posts_and_tokens
        .join(bad_counts, on="subreddit", how="left")
        .fillna({"total_bad_words": 0})
        .withColumn(
            "overall_bad_word_pct",
            (F.col("total_bad_words") / F.col("total_tokens")) * 100.0
        )
        .orderBy(F.desc("overall_bad_word_pct"))
    )

    logger.info("Bad word analysis complete:")
    stats.show(25, truncate=False)

    return stats


from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType
from pyspark.sql.window import Window

def compute_bad_word_frequencies(df, bad_words: set, top_n: int = 100):
    """
    Compute top bad word frequencies per subreddit using Spark-only operations.
    """

    spark = df.sparkSession

    # Broadcast small set
    bad_words_b = spark.sparkContext.broadcast(bad_words)

    # UDF returning BOOLEAN
    def is_bad(word):
        return bool(word and word.lower() in bad_words_b.value)

    is_bad_udf = F.udf(is_bad, BooleanType())

    # 1. Explode (distributed)
    exploded = df.select(
        "subreddit",
        F.explode_outer("tokens").alias("token")
    )

    # 2. Filter only bad words (UDF returns boolean)
    filtered = exploded.filter(is_bad_udf(F.col("token")))

    # 3. Count = # of bad word occurrences per subreddit & token
    freqs = (
        filtered.groupBy("subreddit", "token")
        .agg(F.count("*").alias("count"))
    )

    # 4. Window to get top N
    w = Window.partitionBy("subreddit").orderBy(F.desc("count"))

    ranked = (
        freqs
        .withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= top_n)
        .orderBy("subreddit", F.desc("count"))
    )

    # 5. Collect small aggregated result
    result = {}
    for row in ranked.collect():
        sub = row["subreddit"]
        tok = row["token"]
        cnt = row["count"]

        if sub not in result:
            result[sub] = []
        result[sub].append((tok, cnt))

    return result




# ================================================================
# VISUALIZATIONS
# ================================================================
def visualize_bad_words(subreddit_stats, output_path: str = "bad_words_by_subreddit.png"):
    """
    Create horizontal bar chart of bad word percentages by subreddit.

    Uses pure Python lists (no toPandas) to avoid PySpark ↔ Spark version issues.
    """
    logger.info("Creating bad word visualization...")

    # Pull aggregated stats back to driver as a list of Row objects
    rows = (
        subreddit_stats
        .select("subreddit", "overall_bad_word_pct", "total_posts")
        .orderBy(F.col("overall_bad_word_pct").asc())
        .collect()
    )

    if not rows:
        logger.warning("No subreddit stats available for visualization; skipping plot.")
        return

    # Build Python lists
    subreddits = [r["subreddit"] for r in rows]
    pcts = [float(r["overall_bad_word_pct"] or 0.0) for r in rows]

    # Matplotlib plotting
    fig, ax = plt.subplots(figsize=(10, 12))

    bars = ax.barh(
        subreddits,
        pcts,
        color="#FF4444",
        alpha=0.7,
        edgecolor="darkred",
        linewidth=1.5,
    )

    ax.set_xlabel("Profanity Word Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Subreddit", fontsize=12, fontweight="bold")
    ax.set_title(
        "Profanity Word Usage by Subreddit\n(Reddit AI/ML Communities)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    for bar, pct in zip(bars, pcts):
        width = bar.get_width()
        ax.text(
            width + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.2f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Bar chart saved to: {output_path}")
    plt.close()


def visualize_bad_word_cloud_by_subreddit(df, bad_words, output_path="bad_clouds.png"):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    
    logger.info("Computing Spark-native bad word frequencies...")

    freq_map = compute_bad_word_frequencies(df, bad_words, top_n=100)

    subreddits = sorted(freq_map.keys())
    n = len(subreddits)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()

    for idx, subreddit in enumerate(subreddits):
        ax = axes[idx]
        freq_dict = dict(freq_map[subreddit])

        if freq_dict:
            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="Reds",
                max_words=100
            ).generate_from_frequencies(freq_dict)

            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(subreddit)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, "No bad words", ha="center", va="center")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved word clouds to: {output_path}")
    plt.close()



# ================================================================
# MAIN PREPROCESSING FUNCTION
# ================================================================
def preprocess_data(
    spark: SparkSession,
    s3_input_path: str,
    s3_output_path: str,
    data_type: str,
    sample_fraction: float | None = None,
    remove_stopwords: bool = True,
    apply_lemmatization: bool = True,
):
    """
    Main preprocessing function for comments or submissions.
    """
    logger.info("=" * 80)
    logger.info(f"PREPROCESSING {data_type.upper()}")
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

    df_processed = df_processed.withColumn("token_count", F.size(F.col("tokens")))
    df_processed = df_processed.withColumn("processed_text", F.array_join(F.col("tokens"), " "))

    if data_type == "comments":
        columns_to_save = [
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
    else:
        columns_to_save = [
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

    columns_to_save = [col for col in columns_to_save if col in df_processed.columns]
    df_final = df_processed.select(*columns_to_save)

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

    # === PROFANITY ANALYSIS ===
    logger.info("\n" + "=" * 80)
    logger.info("PROFANITY WORDS ANALYSIS")
    logger.info("=" * 80)

    try:
        bad_words = load_bad_words()
        subreddit_stats = analyze_bad_words_by_subreddit(df_final, bad_words, spark)
        visualize_bad_words(subreddit_stats, f"profanity_words_{data_type}_by_subreddit.png")
        visualize_bad_word_cloud_by_subreddit(
            df_final,
            bad_words,
            f"profanity_word_clouds_{data_type}.png",
        )
        logger.info("\n✓ Profanity words analysis completed successfully!")
    except Exception as e:
        logger.warning(f"Bad word analysis failed: {e}", exc_info=True)

    output_path = f"{s3_output_path.rstrip('/')}/{data_type}_preprocessed/"
    logger.info(f"\nSaving preprocessed data to: {output_path}")

    try:
        df_final.write.mode("overwrite").parquet(output_path)
        logger.info(f"✓ Successfully saved {data_type} to: {output_path}")
    except Exception as err:
        logger.error(f"❌ S3 WRITE FAILED: {err}", exc_info=True)
        return df_final

    return df_final


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
        description="Reddit Data Preprocessing Pipeline with Spark NLP - CLUSTER VERSION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 1%% sample of comments
  python reddit_nlp_bad_words_cluster.py spark://10.0.1.5:7077 --data-type comments --sample 0.01

  # Process all submissions
  python reddit_nlp_bad_words_cluster.py spark://10.0.1.5:7077 --data-type submissions

  # Use environment variable for master
  export MASTER_PRIVATE_IP=10.0.1.5
  python reddit_nlp_bad_words_cluster.py --data-type both
        """,
    )

    parser.add_argument(
        "master_url",
        nargs="?",
        default=None,
        help="Spark master URL (e.g., spark://10.0.1.5:7077)",
    )
    parser.add_argument(
        "--s3-input",
        default=f"s3a://{NETID}-reddit-datasets/project/reddit/parquet/",
        help="Input S3 path",
    )
    parser.add_argument(
        "--s3-output",
        default=f"s3a://{NETID}-reddit-datasets/project/reddit/preprocessed/",
        help="Output S3 path",
    )
    parser.add_argument(
        "--data-type",
        choices=["comments", "submissions", "both"],
        default="comments",
        help="Type of data to process",
    )
    parser.add_argument(
        "--sample",
        type=float,
        help="Sample fraction for testing (e.g., 0.01 for 1%%)",
    )
    parser.add_argument(
        "--no-stopwords",
        action="store_true",
        help="Disable stop words removal",
    )
    parser.add_argument(
        "--no-lemmatization",
        action="store_true",
        help="Disable lemmatization",
    )

    args = parser.parse_args()

    master_url = args.master_url
    if not master_url:
        master_private_ip = os.getenv("MASTER_PRIVATE_IP")
        if master_private_ip:
            master_url = f"spark://{master_private_ip}:7077"
        else:
            print("=" * 70)
            print("❌ Error: Master URL not provided")
            print("Usage: python reddit_nlp_bad_words_cluster.py spark://MASTER_IP:7077 [options]")
            print("   or: export MASTER_PRIVATE_IP=xxx.xxx.xxx.xxx")
            print("=" * 70)
            return 1

    if args.sample and not (0 < args.sample <= 1):
        parser.error("Sample fraction must be between 0 and 1")

    print("=" * 70)
    print("REDDIT NLP PREPROCESSING PIPELINE (CLUSTER MODE)")
    print("AI/ML Subreddit Analysis with Profanity Detection")
    print("=" * 70)
    print(f"Connecting to Spark Master at: {master_url}")
    print(f"Data type: {args.data_type}")
    print(f"Sample: {args.sample * 100:.2f}%" if args.sample else "Sample: FULL DATASET")
    print("=" * 70)

    logger.info(f"Using Spark master URL: {master_url}")
    logger.info(f"Data type: {args.data_type}")
    logger.info(f"Sample fraction: {args.sample if args.sample else 'None (full data)'}")

    spark = None
    start_time = datetime.now()

    try:
        spark = build_spark(master_url)

        if args.data_type in ["comments", "both"]:
            logger.info("\n" + "=" * 80)
            logger.info("PROCESSING COMMENTS")
            logger.info("=" * 80)
            preprocess_data(
                spark=spark,
                s3_input_path=args.s3_input,
                s3_output_path=args.s3_output,
                data_type="comments",
                sample_fraction=args.sample,
                remove_stopwords=not args.no_stopwords,
                apply_lemmatization=not args.no_lemmatization,
            )

        if args.data_type in ["submissions", "both"]:
            logger.info("\n" + "=" * 80)
            logger.info("PROCESSING SUBMISSIONS")
            logger.info("=" * 80)
            preprocess_data(
                spark=spark,
                s3_input_path=args.s3_input,
                s3_output_path=args.s3_output,
                data_type="submissions",
                sample_fraction=args.sample,
                remove_stopwords=not args.no_stopwords,
                apply_lemmatization=not args.no_lemmatization,
            )

        if args.data_type == "both":
            viz_files = [
                "profanity_words_comments_by_subreddit.png",
                "profanity_word_clouds_comments.png",
                "profanity_words_submissions_by_subreddit.png",
                "profanity_word_clouds_submissions.png",
            ]
        else:
            viz_files = [
                f"profanity_words_{args.data_type}_by_subreddit.png",
                f"profanity_word_clouds_{args.data_type}.png",
            ]
        copy_to_drop_folder(viz_files)

        elapsed = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 70)
        print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print(f"Total execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"Preprocessed data saved to: {args.s3_output}")
        print("Visualizations copied to: ~/reddit-nlp-cluster/")
        print("=" * 70)

        logger.info("\n" + "=" * 80)
        logger.info("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        logger.info(f"Total execution time: {elapsed:.1f} seconds")
        logger.info(f"Preprocessed data saved to: {args.s3_output}")
        logger.info("Visualizations copied to: ~/reddit-nlp-cluster/")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\n❌ Error during preprocessing: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1

    finally:
        if spark:
            spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    sys.exit(main())
