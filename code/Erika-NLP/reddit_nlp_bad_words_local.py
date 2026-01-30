"""
Reddit Data Preprocessing Pipeline with Spark NLP
Processes filtered Reddit comments and submissions from S3

S3 Location: s3://ea973-reddit-datasets/project/reddit/parquet/
  - comments/
  - submissions/

Author: Erika
Date: November 2025
"""

import argparse
import os
import logging
from datetime import datetime
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, FloatType

import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, LemmatizerModel
# from sparknlp.annotator import Stemmer (--- IGNORE ---)
from pyspark.ml import Pipeline

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import sys


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_spark(app_name="Reddit_Preprocessing"):
    from pyspark.sql import SparkSession
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Initializing Spark session with fixed S3A configs...")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config(
            "spark.jars.packages",
            "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.4,"
            "org.apache.hadoop:hadoop-aws:3.3.2,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262"
        )
        # Core S3A setup
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")

        # 🔥 FIX ALL INVALID TIME STRINGS 🔥
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60000")
        .config("spark.hadoop.fs.s3a.connection.ttl", "300000")
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400000")
        .config("spark.hadoop.fs.s3a.retry.interval", "500")
        .config("spark.hadoop.fs.s3a.retry.throttle.interval", "100")

        # Optional stability configs
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
        .config("spark.hadoop.fs.s3a.paging.maximum", "5000")

        .getOrCreate()
    )

    logger.info("Spark session created successfully.")
    return spark


def load_reddit_data(spark: SparkSession, 
                     s3_path: str,
                     data_type: str,
                     sample_fraction: float = None) -> "DataFrame":
    """
    Load Reddit data from S3
    
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
            logger.info(f"Sampling {sample_fraction*100}% of data...")
            df = df.sample(fraction=sample_fraction, seed=42)
        
        count = df.count()
        logger.info(f"Loaded {count:,} {data_type}")
        logger.info(f"Columns: {df.columns}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def clean_reddit_text(df, text_col: str) -> "DataFrame":
    """
    Clean Reddit-specific patterns using PySpark regex
    
    Args:
        df: Input DataFrame
        text_col: Name of text column ('body' or 'title')
    
    Returns:
        DataFrame with cleaned text column
    """
    logger.info(f"Cleaning text in column: {text_col}")
    
    # Remove URLs
    df = df.withColumn(
        f"{text_col}_cleaned",
        F.regexp_replace(
            F.col(text_col), 
            r'https?://\S+|www\.\S+|bit\.ly/\S+|redd\.it/\S+', 
            ''
        )
    )
    
    # Remove Reddit formatting
    df = df.withColumn(
        f"{text_col}_cleaned",
        F.regexp_replace(
            F.col(f"{text_col}_cleaned"), 
            r'\[deleted\]|\[removed\]', 
            ''
        )
    )
    
    # Remove markdown links [text](url) -> text
    df = df.withColumn(
        f"{text_col}_cleaned",
        F.regexp_replace(
            F.col(f"{text_col}_cleaned"), 
            r'\[([^\]]+)\]\([^\)]+\)', 
            r'$1'
        )
    )
    
    # Remove username mentions (u/username)
    df = df.withColumn(
        f"{text_col}_cleaned",
        F.regexp_replace(F.col(f"{text_col}_cleaned"), r'u/\w+', '')
    )
    
    # Remove subreddit mentions (r/subreddit)
    df = df.withColumn(
        f"{text_col}_cleaned",
        F.regexp_replace(F.col(f"{text_col}_cleaned"), r'r/\w+', '')
    )
    
    # Remove extra whitespace and trim
    df = df.withColumn(
        f"{text_col}_cleaned",
        F.trim(F.regexp_replace(F.col(f"{text_col}_cleaned"), r'\s+', ' '))
    )
    
    # Filter out very short texts
    initial_count = df.count()
    df = df.filter(F.length(F.col(f"{text_col}_cleaned")) > 10)
    final_count = df.count()
    
    logger.info(f"Filtered out {initial_count - final_count:,} short texts")
    logger.info(f"Remaining records: {final_count:,}")
    
    return df


def build_preprocessing_pipeline(input_col: str = "body_cleaned",
                                 remove_stopwords: bool = True,
                                 apply_lemmatization: bool = True) -> Pipeline:
    """
    Build Spark NLP preprocessing pipeline
    
    Pipeline stages:
    1. DocumentAssembler - Convert text to document format
    2. Tokenizer - Split into tokens
    3. Normalizer - Lowercase, remove punctuation
    4. StopWordsCleaner - Remove stop words (optional)
    # REMOVE 5. Stemmer - Reduce to root form (optional)
    6. Lemmatizer - Convert to base form (optional)
    7. Finisher - Convert back to regular columns
    
    Args:
        input_col: Input text column name
        remove_stopwords: Whether to remove stop words
        # apply_stemming: Whether to apply stemming
        apply_lemmatization: Whether to apply lemmatization
    
    Returns:
        Spark ML Pipeline
    """
    logger.info("Building Spark NLP preprocessing pipeline...")
    
    stages = []
    
    # Stage 1: Document Assembler
    document = DocumentAssembler() \
        .setInputCol(input_col) \
        .setOutputCol("document")
    stages.append(document)
    
    # Stage 2: Tokenizer
    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")
    stages.append(tokenizer)
    
    # Stage 3: Normalizer
    normalizer = Normalizer() \
        .setInputCols(["token"]) \
        .setOutputCol("normalized") \
        .setLowercase(True) \
        .setCleanupPatterns([
            "[^\\w\\s]",  # Remove punctuation
        ])
    stages.append(normalizer)
    
    last_col = "normalized"
    
    # Stage 4: Stop Words Remover (optional)
    if remove_stopwords:
        stopwords_cleaner = StopWordsCleaner() \
            .setInputCols(["normalized"]) \
            .setOutputCol("cleanTokens") \
            .setCaseSensitive(False)
        stages.append(stopwords_cleaner)
        last_col = "cleanTokens"
        logger.info("  - Stop words removal: ENABLED")
    

    # Stage 5: Lemmatizer (optional)
    # (Commented out for now; can be added similarly to Stemmer)    
    # Stage 5: Lemmatizer (pretrained)
    if apply_lemmatization:
        lemmatizer = LemmatizerModel.pretrained("lemma_antbnc", "en") \
            .setInputCols([last_col]) \
            .setOutputCol("lemmatized")
        stages.append(lemmatizer)
        last_col = "lemmatized"
        logger.info("  - Lemmatization: ENABLED (pretrained)")
    else:
        logger.info("  - Lemmatization: DISABLED")

    
    # Stage 6: Finisher
    finisher = Finisher() \
        .setInputCols([last_col]) \
        .setOutputCols(["tokens"]) \
        .setCleanAnnotations(True)
    stages.append(finisher)
    
    logger.info(f"Pipeline has {len(stages)} stages")
    
    return Pipeline(stages=stages)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BAD_WORDS_PATH = os.path.join(SCRIPT_DIR, "bad_words_cleaned.txt")

def load_bad_words(file_path: str = BAD_WORDS_PATH) -> set:
    """
    Load bad words from text file
    
    Args:
        file_path: Path to bad words file
    
    Returns:
        Set of bad words (lowercase)
    """
    logger.info(f"Loading bad words from: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            bad_words = set(word.strip().lower() for word in f if word.strip())
        
        logger.info(f"Loaded {len(bad_words)} bad words")
        return bad_words
    
    except Exception as e:
        logger.error(f"Error loading bad words file: {e}")
        raise

def analyze_bad_words_by_subreddit(df, bad_words: set, spark: SparkSession) -> "DataFrame":
    """
    Calculate percentage of bad words per subreddit
    
    Args:
        df: DataFrame with 'tokens' and 'subreddit' columns
        bad_words: Set of bad words
        spark: SparkSession for broadcasting
    
    Returns:
        DataFrame with bad word statistics per subreddit
    """
    from pyspark.sql.functions import udf
    
    logger.info("Analyzing bad words by subreddit...")
    
    # Broadcast bad words for efficiency
    bad_words_broadcast = spark.sparkContext.broadcast(bad_words)
    
    # UDF to count bad words in token list
    def count_bad_words(tokens):
        if not tokens:
            return 0.0
        bad_set = bad_words_broadcast.value
        bad_count = sum(1 for token in tokens if token.lower() in bad_set)
        return float(bad_count)
    
    # UDF to calculate percentage
    def calc_bad_word_percentage(tokens):
        if not tokens or len(tokens) == 0:
            return 0.0
        bad_set = bad_words_broadcast.value
        bad_count = sum(1 for token in tokens if token.lower() in bad_set)
        return (bad_count / len(tokens)) * 100
    
    count_bad_udf = udf(count_bad_words, FloatType())
    pct_bad_udf = udf(calc_bad_word_percentage, FloatType())
    
    # Add bad word counts to dataframe
    df_with_bad_words = df.withColumn("bad_word_count", count_bad_udf(F.col("tokens")))
    df_with_bad_words = df_with_bad_words.withColumn("bad_word_pct", pct_bad_udf(F.col("tokens")))
    
    # Aggregate by subreddit
    subreddit_stats = df_with_bad_words.groupBy("subreddit").agg(
        F.count("*").alias("total_posts"),
        F.sum("bad_word_count").alias("total_bad_words"),
        F.sum("token_count").alias("total_tokens"),
        F.avg("bad_word_pct").alias("avg_bad_word_pct")
    )
    
    # Calculate overall percentage
    subreddit_stats = subreddit_stats.withColumn(
        "overall_bad_word_pct",
        (F.col("total_bad_words") / F.col("total_tokens")) * 100
    )
    
    # Sort by bad word percentage (descending)
    subreddit_stats = subreddit_stats.orderBy(F.desc("overall_bad_word_pct"))
    
    logger.info("\nBad word analysis by subreddit:")
    subreddit_stats.show(25, truncate=False)
    
    return subreddit_stats

def visualize_bad_words(subreddit_stats, output_path: str = "bad_words_by_subreddit.png"):
    """
    Create vertical bar chart of bad word percentages by subreddit
    
    Args:
        subreddit_stats: DataFrame with subreddit statistics
        output_path: Path to save the chart
    """
    import pandas as pd
    
    logger.info("Creating bad word visualization...")
    
    # Convert to Pandas for plotting
    stats_pd = subreddit_stats.select(
        "subreddit", 
        "overall_bad_word_pct",
        "total_posts"
    ).toPandas()
    
    # Sort by percentage (ascending for horizontal bars)
    stats_pd = stats_pd.sort_values("overall_bad_word_pct", ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Create horizontal bar chart
    bars = ax.barh(
        stats_pd['subreddit'], 
        stats_pd['overall_bad_word_pct'],
        color='#FF4444',
        alpha=0.7,
        edgecolor='darkred',
        linewidth=1.5
    )
    
    # Customize
    ax.set_xlabel('Profanity Word Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Subreddit', fontsize=12, fontweight='bold')
    ax.set_title('Profanity Word Usage by Subreddit\n(Reddit AI/ML Communities)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, stats_pd['overall_bad_word_pct'])):
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{pct:.2f}%',
                va='center', fontsize=9, fontweight='bold')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Bar chart saved to: {output_path}")
    
    plt.close()


def visualize_bad_word_cloud_by_subreddit(df, bad_words: set, output_path: str = "bad_word_clouds.png"):
    """
    Create word clouds showing most common bad words per subreddit
    
    Args:
        df: DataFrame with 'tokens' and 'subreddit' columns
        bad_words: Set of bad words
        output_path: Path to save the visualization
    """
    logger.info("Creating bad word clouds by subreddit...")
    
    # Get list of subreddits
    subreddits = [row['subreddit'] for row in df.select("subreddit").distinct().collect()]
    subreddits.sort()
    
    # Determine grid layout
    n_subreddits = len(subreddits)
    n_cols = 3
    n_rows = (n_subreddits + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_subreddits > 1 else [axes]
    
    for idx, subreddit in enumerate(subreddits):
        logger.info(f"  Processing word cloud for: {subreddit}")
        
        # Filter data for this subreddit
        subreddit_df = df.filter(F.col("subreddit") == subreddit)
        
        # Collect all tokens and filter for bad words
        all_tokens = subreddit_df.select("tokens").rdd.flatMap(lambda x: x[0] if x[0] else []).collect()
        
        # Filter to only bad words and count frequencies
        bad_word_counts = Counter()
        for token in all_tokens:
            if token.lower() in bad_words:
                bad_word_counts[token.lower()] += 1
        
        # Create word cloud
        ax = axes[idx]
        
        if bad_word_counts:
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                colormap='Reds',
                relative_scaling=0.5,
                min_font_size=10,
                max_words=50,
                prefer_horizontal=0.7
            ).generate_from_frequencies(bad_word_counts)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            total_bad = sum(bad_word_counts.values())
            unique_bad = len(bad_word_counts)
            ax.set_title(f'{subreddit}\n({total_bad:,} profanity words, {unique_bad} unique)', 
                        fontsize=12, fontweight='bold', pad=10)
        else:
            ax.text(0.5, 0.5, 'No profanity words found', 
                   ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(f'{subreddit}', fontsize=12, fontweight='bold')
        
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(n_subreddits, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Most Common Profanity Words by AI Subreddit', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Word clouds saved to: {output_path}")
    
    plt.close()


def preprocess_data(spark: SparkSession,
                   s3_input_path: str,
                   s3_output_path: str,
                   data_type: str,
                   sample_fraction: float = None,
                   remove_stopwords: bool = True,
                   apply_lemmatization: bool = True) -> "DataFrame":
    """
    Main preprocessing function
    
    Args:
        spark: SparkSession
        s3_input_path: Input S3 path
        s3_output_path: Output S3 path
        data_type: 'comments' or 'submissions'
        sample_fraction: Optional sampling
        remove_stopwords: Remove stop words
        # apply_stemming: Apply stemming
        apply_lemmatization: Apply lemmatization
    """
    logger.info("=" * 80)
    logger.info(f"PREPROCESSING {data_type.upper()}")
    logger.info("=" * 80)


    # Load data
    df = load_reddit_data(spark, s3_input_path, data_type, sample_fraction)

    
    
    # Handle text columns based on data type
    if data_type == "comments":
        # Comments use 'body' column
        text_col = "body"
        if text_col not in df.columns:
            logger.error(f"Column '{text_col}' not found in comments data")
            logger.info(f"Available columns: {df.columns}")
            raise ValueError(f"Missing column: {text_col}")
        
        # Show sample before preprocessing
        logger.info("\nSample comments BEFORE preprocessing:")
        df.select("subreddit", text_col).show(3, truncate=80)
        
        # Clean text
        df_cleaned = clean_reddit_text(df, text_col)
        
        # Show sample after cleaning
        logger.info("\nSample comments AFTER cleaning:")
        df_cleaned.select("subreddit", text_col, f"{text_col}_cleaned").show(3, truncate=80)
        
        text_col_to_process = f"{text_col}_cleaned"
        
    else:  # submissions
        # Submissions use 'title' and 'selftext' - combine them
        if "title" not in df.columns:
            logger.error("Column 'title' not found in submissions data")
            logger.info(f"Available columns: {df.columns}")
            raise ValueError("Missing column: title")
        
        # Combine title and selftext (selftext can be empty)
        logger.info("\nCombining 'title' and 'selftext' columns...")
        if "selftext" in df.columns:
            df = df.withColumn(
                "text",
                F.concat_ws(" ", F.col("title"), F.coalesce(F.col("selftext"), F.lit("")))
            )
        else:
            df = df.withColumn("text", F.col("title"))
        
        # Show sample before preprocessing
        logger.info("\nSample submissions BEFORE preprocessing:")
        df.select("subreddit", "title", "selftext").show(3, truncate=80)
        
        # Clean combined text
        text_col = "text"
        df_cleaned = clean_reddit_text(df, text_col)
        
        # Show sample after cleaning
        logger.info("\nSample submissions AFTER cleaning:")
        df_cleaned.select("subreddit", "title", f"{text_col}_cleaned").show(3, truncate=80)
        
        text_col_to_process = f"{text_col}_cleaned"
    
    # Build and fit pipeline
    pipeline = build_preprocessing_pipeline(
        input_col=text_col_to_process,
        remove_stopwords=remove_stopwords,
        apply_lemmatization=apply_lemmatization
    )
    
    logger.info("\nFitting pipeline...")
    pipeline_model = pipeline.fit(df_cleaned)
    
    logger.info("Transforming data...")
    df_processed = pipeline_model.transform(df_cleaned)
    
    # Add useful columns
    df_processed = df_processed.withColumn(
        "token_count",
        F.size(F.col("tokens"))
    )
    
    df_processed = df_processed.withColumn(
        "processed_text",
        F.array_join(F.col("tokens"), " ")
    )
    
    # Select final columns to save based on data type
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
            "gilded"
        ]
    else:  # submissions
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
            "over_18"
        ]
    
    # Only keep columns that exist in the dataframe
    columns_to_save = [col for col in columns_to_save if col in df_processed.columns]
    
    df_final = df_processed.select(*columns_to_save)
    
    # Show final results
    logger.info("\nFinal preprocessed data sample:")
    if data_type == "comments":
        logger.info("\nAll columns in final dataset:")
        df_final.select(*df_final.columns).show(3, truncate=80, vertical=True)
    else:
        logger.info("\nAll columns in final dataset:")
        df_final.select(*df_final.columns).show(3, truncate=80, vertical=True)
    
    # Statistics
    logger.info("\nPreprocessing Statistics:")
    stats = df_final.select(
        F.count("subreddit").alias("total_records"),
        F.avg("token_count").alias("avg_tokens"),
        F.min("token_count").alias("min_tokens"),
        F.max("token_count").alias("max_tokens")
    ).collect()[0]
    
    logger.info(f"  Total records: {stats['total_records']:,}")
    logger.info(f"  Average tokens: {stats['avg_tokens']:.2f}")
    logger.info(f"  Min tokens: {stats['min_tokens']}")
    logger.info(f"  Max tokens: {stats['max_tokens']}")

    # === Profanity WORDS ANALYSIS ===
    logger.info("\n" + "=" * 80)
    logger.info("PROFANITY WORDS ANALYSIS")
    logger.info("=" * 80)
    
    try:
        bad_words = load_bad_words()
        
        # Analyze bad words by subreddit
        subreddit_stats = analyze_bad_words_by_subreddit(df_final, bad_words, spark)
        
        # Create vertical bar chart
        visualize_bad_words(subreddit_stats, f"profanity_words_{data_type}_by_subreddit.png")
        
        # Create word clouds
        visualize_bad_word_cloud_by_subreddit(df_final, bad_words, f"profanity_word_clouds_{data_type}.png")
        
        logger.info("\n✓ Profanity words analysis completed successfully!")
        
    except Exception as e:
        logger.warning(f"Bad word analysis failed: {e}")
    
    # Save to S3
    output_path = f"{s3_output_path.rstrip('/')}/{data_type}_preprocessed/"
    logger.info(f"\nSaving preprocessed data to: {output_path}")
    logger.info(f"DEBUG — About to write to S3 path: {output_path}")
    logger.info(f"DEBUG — Using AWS credentials from EC2 instance profile")

    try:
        df_final.write.mode("overwrite").parquet(output_path)
        logger.info(f"✓ Successfully saved {data_type} to: {output_path}")
    except Exception as err:
        logger.error(f"❌ S3 WRITE FAILED: {err}")
        return

    
    df_final.write \
        .mode("overwrite") \
        .parquet(output_path)
    
    logger.info(f"✓ Successfully saved preprocessed {data_type} to S3!")
    
    return df_final


def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(
        description="Reddit Data Preprocessing Pipeline with Spark NLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 1% sample of comments
  python reddit_preprocessing.py --data-type comments --sample 0.01
  
  # Process all submissions
  python reddit_preprocessing.py --data-type submissions
  
  # Process both with custom paths
  python reddit_preprocessing.py --data-type both --output s3://bucket/output/
  
  # Without stop words removal
  python reddit_preprocessing.py --data-type comments --no-stopwords
        """
    )
    
    parser.add_argument(
        "--s3-input",
        default="s3://ea973-reddit-datasets/project/reddit/parquet/",
        help="Input S3 path (default: s3://ea973-reddit-datasets/project/reddit/parquet/)"
    )
    
    parser.add_argument(
        "--s3-output",
        default="s3://ea973-reddit-datasets/project/reddit/preprocessed/",
        help="Output S3 path (default: s3://ea973-reddit-datasets/project/reddit/preprocessed/)"
    )
    
    parser.add_argument(
        "--data-type",
        choices=["comments", "submissions", "both"],
        default="comments",
        help="Type of data to process (default: comments)"
    )
    
    parser.add_argument(
        "--sample",
        type=float,
        help="Sample fraction for testing (e.g., 0.01 for 1%%)"
    )
    
    parser.add_argument(
        "--no-stopwords",
        action="store_true",
        help="Disable stop words removal"
    )

    parser.add_argument(
        "--no-lemmatization",
        action="store_true",
        help="Disable lemmatization"
    )
    
    args = parser.parse_args()
    
    # Validate sample fraction
    if args.sample and not (0 < args.sample <= 1):
        parser.error("Sample fraction must be between 0 and 1")
    
    # Build Spark session
    spark = build_spark()
    
    start_time = datetime.now()
    
    try:
        # Process based on data type
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
                # apply_stemming=not args.no_stemming,
                apply_lemmatization=not args.no_lemmatization
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
                # apply_stemming=not args.no_stemming,
                apply_lemmatization=not args.no_lemmatization
            )
        
        # Success summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("\n" + "=" * 80)
        logger.info("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        logger.info(f"Total execution time: {elapsed:.1f} seconds")
        logger.info(f"Preprocessed data saved to: {args.s3_output}")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"\n❌ Error during preprocessing: {e}", exc_info=True)
        raise
    
    finally:
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    main()