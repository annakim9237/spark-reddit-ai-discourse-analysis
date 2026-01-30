#!/usr/bin/env python3
"""
Question 5: How has sentiment toward different generative AI tools shifted over time across subreddits?

Spark Cluster Version - This script performs sentiment analysis on Reddit comments and submissions
using PySpark ML Pipeline (optimized based on task1_sentiment_classification.py) to track how 
sentiment toward different AI tools has evolved over time.

Key Optimizations (from task1_sentiment_classification.py):
- Uses Spark ML Pipeline for text preprocessing (Tokenizer -> StopWordsRemover -> HashingTF -> IDF)
- Efficient feature extraction with configurable numFeatures (default: 5000)
- Optimized UDF usage for sentiment calculation

Analysis Type: NLP

Compatible with:
- Spark 3.4.4
- PySpark 3.4.x (>=3.4.0, <3.5.0)
"""

import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, when, length, regexp_replace, lower, hour, dayofweek, month,
    from_unixtime, to_date, udf, lit, concat_ws, split, size,
    coalesce, date_format, year, quarter, avg, stddev, count, sum as spark_sum
)
from pyspark.sql.types import DoubleType, StringType, BooleanType
# For Spark 3.4.4, pandas_udf is in pyspark.sql.functions
# Note: In Spark 3.4.4, pandas_udf is still available in pyspark.sql.functions
try:
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    HAS_PANDAS_UDF = True
except ImportError:
    # Fallback for older versions or if not available
    try:
        from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType
        HAS_PANDAS_UDF = True
    except ImportError:
        # Fallback: use regular UDF if pandas_udf not available
        pandas_udf = None
        PandasUDFType = None
        HAS_PANDAS_UDF = False
import pandas as pd
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sentiment_lexicons(project_root: str) -> Tuple[set, set]:
    """
    Load positive and negative word lists from files.
    
    Args:
        project_root: Path to the project root directory
        
    Returns:
        tuple: (positive_words_set, negative_words_set)
    """
    positive_file = os.path.join(project_root, "positive_words.txt")
    negative_file = os.path.join(project_root, "bad_words_cleaned.txt")
    
    positive_words = set()
    negative_words = set()
    
    # Load positive words
    if os.path.exists(positive_file):
        logger.info(f"Loading positive words from {positive_file}...")
        with open(positive_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip().lower()
                if line and not line.startswith(';'):
                    positive_words.add(line)
        logger.info(f"Loaded {len(positive_words)} positive words")
    else:
        logger.warning(f"Positive words file not found: {positive_file}")
    
    # Load negative words
    if os.path.exists(negative_file):
        logger.info(f"Loading negative words from {negative_file}...")
        with open(negative_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip().lower()
                if line and not line.startswith(';'):
                    negative_words.add(line)
        logger.info(f"Loaded {len(negative_words)} negative words")
    else:
        logger.warning(f"Negative words file not found: {negative_file}")
    
    return positive_words, negative_words


def create_spark_session(master_url: str = "local[*]", app_name: str = "NLP_Sentiment_Analysis",
                        local_mode: bool = True) -> SparkSession:
    """
    Create a Spark session optimized for NLP processing.
    Compatible with Spark 3.4.4 / PySpark 3.4.x
    
    Args:
        master_url: URL of the Spark master node
        app_name: Application name
        local_mode: Whether running in local mode
        
    Returns:
        Configured SparkSession
    """
    logger.info(f"Creating Spark session for Spark 3.4.4 (master: {master_url}, local_mode: {local_mode})...")
    
    # Stop any existing SparkSession to avoid conflicts
    try:
        existing_session = SparkSession.getActiveSession()
        if existing_session:
            logger.info("Stopping existing SparkSession...")
            existing_session.stop()
            time.sleep(1)
    except Exception as e:
        logger.debug(f"Error during cleanup: {e}")
    
    # Build SparkSession compatible with Spark 3.4.4
    builder = (
        SparkSession.builder
        .appName(app_name)
    )
    
    # Set master URL if provided (for cluster mode)
    if master_url and master_url != "local[*]":
        logger.info(f"Using cluster mode with master: {master_url}")
        builder = builder.master(master_url)
    else:
        logger.info("Using local mode (default)")
    
    if local_mode:
        builder = (builder
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "4g")
            .config("spark.driver.maxResultSize", "2g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.default.parallelism", "200"))
    else:
        # Cluster mode configuration matching setup-spark-cluster_nlp.sh
        # Cluster setup: 1 Master + 3 Workers, all t3.large instances
        # t3.large specs: 2 vCPU, 8GB RAM per node
        # See: spark-cluster/setup-spark-cluster_nlp.sh for cluster setup details
        builder = (builder
            # Memory configuration (leave ~1GB for system on 8GB nodes)
            .config("spark.executor.memory", "6g")
            .config("spark.driver.memory", "2g")
            .config("spark.driver.maxResultSize", "1g")
            # Core configuration (t3.large has 2 vCPU)
            .config("spark.executor.cores", "2")
            # Adaptive execution
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            # Performance optimizations
            .config("spark.sql.shuffle.partitions", "200")  # Increase parallelism
            .config("spark.default.parallelism", "200")
            # S3 configuration (matching setup-spark-cluster_nlp.sh for Spark 3.4.4)
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                    "com.amazonaws.auth.InstanceProfileCredentialsProvider")
            # S3 connection timeouts
            .config("spark.hadoop.fs.s3a.connection.timeout", "600000")  # 10 minutes
            .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
            .config("spark.hadoop.fs.s3a.attempts.maximum", "20")
            # Serialization
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            # Arrow for pandas integration
            .config("spark.sql.execution.arrow.pyspark.enabled", "true"))
    
    spark = builder.getOrCreate()
    
    # Verify Spark version compatibility (must be 3.4.4 as per setup-spark-cluster_nlp.sh)
    spark_version = spark.version
    logger.info(f"Spark session created successfully")
    logger.info(f"Spark version: {spark_version}")
    if spark_version.startswith("3.4"):
        logger.info(f"✅ Spark version {spark_version} is compatible (expected 3.4.4)")
    else:
        logger.warning(f"⚠️  Expected Spark 3.4.4 (as per setup-spark-cluster_nlp.sh), but got {spark_version}. Compatibility may vary.")
    
    return spark


def read_data_from_local_or_s3(spark: SparkSession, net_id: str = None, 
                                data_type: str = "submissions",
                                local_mode: bool = True,
                                local_data_dir: str = None) -> DataFrame:
    """
    Read Reddit data (comments or submissions) from local files or S3.
    
    Args:
        spark: Active SparkSession
        net_id: Student's net ID for S3 bucket access (if not local_mode)
        data_type: Either "comments" or "submissions"
        local_mode: If True, read from local files
        local_data_dir: Local directory containing parquet files
        
    Returns:
        DataFrame with Reddit data
    """
    if local_mode and local_data_dir:
        local_path = os.path.join(local_data_dir, f"{data_type}.parquet")
        if os.path.exists(local_path):
            logger.info(f"Reading from local file: {local_path}")
            df = spark.read.parquet(local_path)
            logger.info(f"Loaded data from local file (counting rows skipped for performance)")
            return df
        else:
            raise FileNotFoundError(f"Local data file not found: {local_path}")
    
    if not net_id:
        raise ValueError("net_id is required when not in local_mode")
    
    # Try project path first (as specified by user), then fallback to other paths
    project_path = f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/{data_type}/"
    filtered_path = f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/{data_type}/"
    raw_path = f"s3a://{net_id}-dsan6000-datasets/reddit/parquet/{data_type}/"
    
    # Try project path first (user specified location)
    try:
        logger.info(f"Attempting to read from project data: {project_path}")
        df = spark.read.parquet(project_path)
        logger.info(f"Loaded data from project path (counting rows skipped for performance)")
        return df
    except Exception as e:
        logger.warning(f"Could not read from project path: {e}")
        # Try filtered data
        try:
            logger.info(f"Attempting to read from filtered data: {filtered_path}")
            df = spark.read.parquet(filtered_path)
            logger.info(f"Loaded data from filtered path (counting rows skipped for performance)")
            return df
        except Exception as e2:
            logger.warning(f"Could not read from filtered path: {e2}")
            logger.info(f"Reading from raw data: {raw_path}")
            df = spark.read.parquet(raw_path)
            logger.info(f"Loaded data from raw path (counting rows skipped for performance)")
            return df


def preprocess_text_spark(df: DataFrame, text_column: str, num_features: int = 5000) -> Tuple[DataFrame, Any]:
    """
    Preprocess text using Spark ML Pipeline (similar to task1_sentiment_classification.py).
    Uses: Tokenizer -> StopWordsRemover -> HashingTF -> IDF
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column to preprocess
        num_features: Number of features for HashingTF (default: 5000, matching reference)
        
    Returns:
        tuple: (DataFrame with processed_text and features, fitted Pipeline model)
    """
    logger.info(f"Preprocessing text column: {text_column} using Spark ML Pipeline")
    
    # Step 1: Clean text using Spark SQL (remove URLs, Reddit formatting, markdown)
    df_cleaned = df.withColumn(
        "cleaned_text",
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    coalesce(col(text_column), lit("")),
                    r"http\S+|www\S+", ""
                ),
                r"\[deleted\]|\[removed\]", ""
            ),
            r"\[([^\]]+)\]\([^\)]+\)", "$1"
        )
    )
    
    # Remove extra whitespace
    df_cleaned = df_cleaned.withColumn(
        "cleaned_text",
        regexp_replace(col("cleaned_text"), r"\s+", " ")
    )
    
    # Filter out empty texts
    df_cleaned = df_cleaned.filter(length(col("cleaned_text")) > 0)
    
    # Step 2: Create Spark ML Pipeline for text processing
    # Following the pattern from task1_sentiment_classification.py
    tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(
        inputCol="filtered_words", 
        outputCol="raw_features", 
        numFeatures=num_features
    )
    idf = IDF(inputCol="raw_features", outputCol="text_features")
    
    # Create and fit pipeline
    feature_pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
    feature_model = feature_pipeline.fit(df_cleaned)
    
    # Transform data
    df_transformed = feature_model.transform(df_cleaned)
    
    # Keep processed_text for sentiment analysis (use cleaned_text)
    df_transformed = df_transformed.withColumnRenamed("cleaned_text", "processed_text")
    
    logger.info(f"Text preprocessing complete (using {num_features} TF-IDF features)")
    
    return df_transformed, feature_model


def create_sentiment_udf(positive_words: set, negative_words: set):
    """
    Create vectorized pandas UDF for sentiment analysis using lexicon-based approach.
    Falls back to regular UDF if pandas_udf is not available.
    This is much faster than regular Python UDF for large datasets.
    
    Args:
        positive_words: Set of positive sentiment words
        negative_words: Set of negative sentiment words
        
    Returns:
        UDF function for sentiment calculation
    """
    def calculate_single(text: str) -> float:
        """Calculate sentiment score for a single text."""
        if not text or len(str(text).strip()) == 0:
            return 0.0
        
        text_lower = str(text).lower()
        words = text_lower.split()
        words = [w for w in words if w.isalnum()]
        
        if not words:
            return 0.0
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0
        
        # Normalize to -1 to 1 range
        score = (positive_count - negative_count) / max(total_sentiment_words, 1)
        
        # Apply intensity adjustments
        if positive_count > negative_count:
            score = min(1.0, score * 1.2)
        elif negative_count > positive_count:
            score = max(-1.0, score * 1.2)
        
        return float(score)
    
    # Try to use pandas_udf for better performance (Spark 3.4.4 supports this)
    if HAS_PANDAS_UDF and pandas_udf is not None:
        try:
            @pandas_udf(returnType=DoubleType())
            def calculate_sentiment_vectorized(texts: pd.Series) -> pd.Series:
                """Calculate sentiment score using lexicon (vectorized for Spark 3.4.4)."""
                return texts.apply(calculate_single)
            logger.info("Using pandas_udf for vectorized sentiment calculation (Spark 3.4.4)")
            return calculate_sentiment_vectorized
        except Exception as e:
            logger.warning(f"Failed to create pandas_udf: {e}. Falling back to regular UDF.")
            return udf(calculate_single, DoubleType())
    else:
        # Fallback to regular UDF (compatible with all Spark versions)
        logger.info("Using regular UDF for sentiment calculation")
        return udf(calculate_single, DoubleType())


def add_sentiment_scores_spark(df: DataFrame, text_column: str, sentiment_udf, 
                               num_features: int = 5000) -> Tuple[DataFrame, Any]:
    """
    Add sentiment scores to DataFrame using UDF.
    Uses Spark ML Pipeline for text preprocessing (optimized approach).
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        sentiment_udf: UDF function for sentiment calculation
        num_features: Number of features for HashingTF (default: 5000)
        
    Returns:
        tuple: (DataFrame with sentiment column, fitted Pipeline model)
    """
    logger.info("Analyzing sentiment using optimized Spark ML Pipeline...")
    
    # Preprocess text using Spark ML Pipeline
    df_processed, feature_model = preprocess_text_spark(df, text_column, num_features)
    
    # Cache preprocessed data to avoid recomputation
    df_processed = df_processed.cache()
    
    # Calculate sentiment using lexicon-based UDF
    df_with_sentiment = df_processed.withColumn(
        "sentiment",
        sentiment_udf(col("processed_text"))
    )
    
    # Fill null sentiments with 0
    df_with_sentiment = df_with_sentiment.fillna({"sentiment": 0.0})
    
    logger.info("Sentiment analysis complete")
    
    return df_with_sentiment, feature_model


def create_mentions_ai_tool_condition():
    """
    Create Spark SQL condition to check if text mentions any AI tools.
    Uses native Spark SQL functions instead of UDF for better performance.
    
    Returns:
        Spark SQL Column expression for AI tool mention detection
    """
    ai_tools_patterns = [
        # Text AI tools
        r'(?i)\b(chatgpt|gpt-3|gpt-4|gpt3|gpt4|openai|claude|perplexity|bard|gemini|copilot|github copilot)\b',
        # Creative AI tools
        r'(?i)\b(midjourney|stable diffusion|stablediffusion|dall-e|dalle|sora|ai art|aiart|runway|pika|leonardo)\b',
        # General AI terms
        r'(?i)\b(generative ai|generativeai|artificial intelligence|machine learning|deep learning|neural network|llm|large language model|diffusion model)\b'
    ]
    
    # Combine all patterns with OR
    combined_pattern = '|'.join(ai_tools_patterns)
    
    def mentions_ai_tool(text_col):
        """Check if text mentions any AI tools using regex."""
        # Use col().rlike() method instead of rlike() function
        # This is compatible with PySpark 3.4+ and 4.0+
        return (
            (length(text_col) > 0) &
            lower(text_col).rlike(combined_pattern)
        )
    
    return mentions_ai_tool


def categorize_subreddits_spark(df: DataFrame) -> DataFrame:
    """
    Categorize subreddits into AI tool groups using Spark SQL.
    Includes AskReddit as baseline category.
    
    Args:
        df: Input DataFrame with subreddit column
        
    Returns:
        DataFrame with ai_category column
    """
    logger.info("Categorizing subreddits...")
    
    def categorize_subreddit_udf(subreddit: str) -> str:
        """Categorize subreddit into AI tool groups."""
        if not subreddit:
            return "Other"
        
        subreddit_lower = str(subreddit).lower()
        
        # Baseline: AskReddit
        if 'askreddit' in subreddit_lower:
            return 'Baseline (AskReddit)'
        
        # Text AI tools
        if any(tool in subreddit_lower for tool in ['chatgpt', 'openai', 'gpt4', 'claude', 'perplexity']):
            return 'Text AI'
        elif any(tool in subreddit_lower for tool in ['midjourney', 'stablediffusion', 'sora', 'aiart']):
            return 'Creative AI'
        elif any(tool in subreddit_lower for tool in ['generativeai', 'artificialintelligence', 'machinelearning']):
            return 'Research/Tech'
        else:
            return 'Other'
    
    categorize_udf = udf(categorize_subreddit_udf, StringType())
    
    df_categorized = df.withColumn("ai_category", categorize_udf(col("subreddit")))
    
    return df_categorized


def aggregate_sentiment_by_time_spark(
    df: DataFrame, 
    time_period: str = 'monthly',
    text_column: str = 'processed_text'
) -> DataFrame:
    """
    Aggregate sentiment scores by subreddit and time period using Spark SQL.
    Includes baseline detection for AskReddit posts mentioning AI tools.
    
    Args:
        df: Input DataFrame with sentiment, subreddit, created_utc columns
        time_period: Time period for aggregation ('monthly', 'quarterly', or 'daily')
        text_column: Column name containing the text (for AI tool mention detection)
        
    Returns:
        Aggregated DataFrame
    """
    logger.info(f"Aggregating sentiment by {time_period}...")
    
    # Convert timestamp to date
    df_with_date = df.withColumn(
        "date",
        to_date(from_unixtime(col("created_utc")))
    )
    
    # Create time period column
    if time_period == 'monthly':
        df_with_date = df_with_date.withColumn(
            "time_period",
            date_format(col("date"), "yyyy-MM")
        )
    elif time_period == 'quarterly':
        df_with_date = df_with_date.withColumn(
            "year",
            year(col("date"))
        )
        df_with_date = df_with_date.withColumn(
            "quarter",
            quarter(col("date"))
        )
        df_with_date = df_with_date.withColumn(
            "time_period",
            concat_ws("-Q", col("year"), col("quarter"))
        )
    else:  # daily
        df_with_date = df_with_date.withColumn(
            "time_period",
            date_format(col("date"), "yyyy-MM-dd")
        )
    
    # Filter out null dates
    df_with_date = df_with_date.filter(col("date").isNotNull())
    
    # For AskReddit (baseline), check if text mentions AI tools
    # Create a special baseline category for AskReddit posts mentioning AI tools
    if text_column in df_with_date.columns:
        mentions_ai_condition = create_mentions_ai_tool_condition()
        
        # Update ai_category for AskReddit posts mentioning AI tools (using native Spark SQL)
        df_with_date = df_with_date.withColumn(
            "ai_category",
            when(
                (col("ai_category") == "Baseline (AskReddit)") & 
                mentions_ai_condition(col(text_column)),
                lit("Baseline (AskReddit - AI mentioned)")
            ).otherwise(col("ai_category"))
        )
        
        logger.info("AI tool mention detection complete (counting skipped for performance)")
    
    # Aggregate by subreddit, ai_category, and time_period
    sentiment_agg = df_with_date.groupBy("subreddit", "ai_category", "time_period").agg(
        avg("sentiment").alias("avg_sentiment"),
        stddev("sentiment").alias("sentiment_std"),
        count("sentiment").alias("sentiment_count"),
        avg("score").alias("avg_score"),
        count("id").alias("post_count")
    )
    
    # Sort by time period
    sentiment_agg = sentiment_agg.orderBy("time_period", "subreddit")
    
    return sentiment_agg


def save_results_to_s3_or_local(
    spark: SparkSession,
    sentiment_agg: DataFrame,
    net_id: str = None,
    output_dir: str = None
) -> None:
    """
    Save sentiment analysis results to S3 or local files.
    
    Args:
        spark: Active SparkSession
        sentiment_agg: Aggregated sentiment DataFrame
        net_id: Student's net ID (required if output_dir is S3 path)
        output_dir: Output directory (local path or s3a:// path)
    """
    # Determine output path
    if output_dir:
        if output_dir.startswith("s3a://") or output_dir.startswith("s3://"):
            base_path = output_dir.rstrip('/') + "/"
        else:
            base_path = os.path.join(output_dir, "")
            os.makedirs(base_path, exist_ok=True)
    else:
        # Default: use S3 with net_id
        if not net_id:
            raise ValueError("net_id is required when output_dir is not specified")
        base_path = f"s3a://{net_id}-dsan6000-datasets/project/nlp_results/"
    
    logger.info(f"Saving results to: {base_path}")
    
    # Check if output path is S3
    is_s3 = base_path.startswith("s3a://") or base_path.startswith("s3://")
    
    if is_s3:
        # Save overall results by subreddit and time
        output_path = f"{base_path}nlp_sentiment_by_subreddit_time/"
        sentiment_agg.coalesce(1).write.mode("overwrite").parquet(output_path)
        logger.info(f"✅ Saved to S3: {output_path}")
        
        # Save summary by AI category
        category_summary = sentiment_agg.groupBy("ai_category", "time_period").agg(
            avg("avg_sentiment").alias("avg_sentiment"),
            spark_sum("post_count").alias("post_count"),
            avg("avg_score").alias("avg_score")
        ).orderBy("time_period", "ai_category")
        
        category_path = f"{base_path}nlp_sentiment_by_category_time/"
        category_summary.coalesce(1).write.mode("overwrite").parquet(category_path)
        logger.info(f"✅ Saved to S3: {category_path}")
    else:
        # Save to local as CSV
        import pandas as pd
        
        # Convert to Pandas for CSV export
        sentiment_pd = sentiment_agg.toPandas()
        output_file = os.path.join(base_path, "nlp_sentiment_by_subreddit_time.csv")
        sentiment_pd.to_csv(output_file, index=False)
        logger.info(f"✅ Saved: {output_file}")
        
        # Save summary by AI category
        category_summary = sentiment_agg.groupBy("ai_category", "time_period").agg(
            avg("avg_sentiment").alias("avg_sentiment"),
            spark_sum("post_count").alias("post_count"),
            avg("avg_score").alias("avg_score")
        ).orderBy("time_period", "ai_category")
        
        category_pd = category_summary.toPandas()
        category_file = os.path.join(base_path, "nlp_sentiment_by_category_time.csv")
        category_pd.to_csv(category_file, index=False)
        logger.info(f"✅ Saved: {category_file}")


def print_summary_statistics(sentiment_agg: DataFrame):
    """Print summary statistics from Spark DataFrame."""
    logger.info("\n" + "=" * 80)
    logger.info("SENTIMENT ANALYSIS SUMMARY STATISTICS")
    logger.info("=" * 80)
    
    # Collect statistics (for small aggregated data, this is safe)
    stats = sentiment_agg.agg(
        count("subreddit").alias("total_records"),
        avg("avg_sentiment").alias("overall_avg_sentiment"),
        stddev("avg_sentiment").alias("overall_sentiment_std")
    ).collect()[0]
    
    logger.info("\nOverall Statistics:")
    logger.info(f"  Total records: {stats['total_records']:,}")
    logger.info(f"  Overall average sentiment: {stats['overall_avg_sentiment']:.3f}")
    logger.info(f"  Overall sentiment std dev: {stats['overall_sentiment_std']:.3f}")
    
    # Unique counts (only for aggregated data which is small)
    # Note: This is safe because sentiment_agg is already aggregated (small dataset)
    try:
        unique_subreddits = sentiment_agg.select("subreddit").distinct().count()
        unique_periods = sentiment_agg.select("time_period").distinct().count()
        logger.info(f"  Unique subreddits: {unique_subreddits}")
        logger.info(f"  Unique time periods: {unique_periods}")
    except Exception as e:
        logger.warning(f"Could not count unique values: {e}")
    
    # By AI category
    logger.info("\nStatistics by AI Category:")
    category_stats = sentiment_agg.groupBy("ai_category").agg(
        avg("avg_sentiment").alias("mean_sentiment"),
        stddev("avg_sentiment").alias("std_sentiment"),
        count("subreddit").alias("subreddit_count"),
        spark_sum("post_count").alias("total_posts")
    ).orderBy("ai_category")
    
    for row in category_stats.collect():
        # Include all categories except 'Other' (but show baseline categories)
        if row['ai_category'] != 'Other':
            logger.info(f"  {row['ai_category']}:")
            logger.info(f"    Average sentiment: {row['mean_sentiment']:.3f} (±{row['std_sentiment']:.3f})")
            logger.info(f"    Subreddits: {row['subreddit_count']}, Total posts: {row['total_posts']:,}")
    
    logger.info("\n" + "=" * 80)


def main() -> int:
    """Main function for Spark sentiment analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NLP: Sentiment Analysis Over Time")
    parser.add_argument("--master", type=str, default="local[*]",
                       help="Spark master URL (e.g., 'spark://MASTER_PRIVATE_IP:7077' for cluster mode)")
    parser.add_argument("--net-id", type=str, default=None,
                       help="Net ID for S3 access (required if not using local mode)")
    parser.add_argument("--local-mode", action="store_true", default=False,
                       help="Use local files instead of S3 (default: False, use S3)")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Local data directory (if local-mode)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (local path or s3a:// path). Default: s3a://{net_id}-dsan6000-datasets/project/nlp_results/")
    parser.add_argument("--time-period", type=str, default="monthly",
                       choices=["daily", "monthly", "quarterly"],
                       help="Time period for aggregation (default: monthly)")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Question 5: Sentiment Analysis Over Time (Spark Version)")
    logger.info("=" * 80)
    
    # Determine if we're in local mode based on output paths
    actual_local_mode = args.local_mode and not (
        args.output_dir and (args.output_dir.startswith("s3a://") or args.output_dir.startswith("s3://"))
    )
    
    # Validate net_id if using S3
    if not actual_local_mode and not args.net_id:
        logger.error("--net-id is required when using S3 (not in local mode)")
        return 1
    
    # Set default output directory if not provided
    if not args.output_dir:
        if actual_local_mode:
            args.output_dir = "./csv"
        else:
            args.output_dir = f"s3a://{args.net_id}-dsan6000-datasets/project/output"
    
    logger.info(f"Using NET_ID: {args.net_id}")
    logger.info(f"Using Spark Master: {args.master}")
    logger.info(f"Local mode: {actual_local_mode}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Determine project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    try:
        # Load sentiment lexicons
        logger.info("\n" + "=" * 80)
        logger.info("Loading Sentiment Lexicons")
        logger.info("=" * 80)
        positive_words, negative_words = load_sentiment_lexicons(project_root)
        
        if not positive_words or not negative_words:
            logger.warning("Sentiment lexicons are empty. Using fallback.")
            positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful'}
            negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst'}
        
        # Create Spark session
        logger.info("\n" + "=" * 80)
        logger.info("Creating Spark Session")
        logger.info("=" * 80)
        spark = create_spark_session(args.master, local_mode=actual_local_mode)
        
        # Read comments data
        logger.info("\n" + "=" * 80)
        logger.info("Loading Comments Data")
        logger.info("=" * 80)
        comments_df = read_data_from_local_or_s3(
            spark, 
            net_id=args.net_id,
            data_type="comments",
            local_mode=actual_local_mode,
            local_data_dir=args.data_dir
        )
        
        # Read submissions data
        logger.info("\n" + "=" * 80)
        logger.info("Loading Submissions Data")
        logger.info("=" * 80)
        submissions_df = read_data_from_local_or_s3(
            spark,
            net_id=args.net_id,
            data_type="submissions",
            local_mode=actual_local_mode,
            local_data_dir=args.data_dir
        )
        
        # Create sentiment UDF
        sentiment_udf = create_sentiment_udf(positive_words, negative_words)
        
        # Analyze sentiment for comments
        logger.info("\n" + "=" * 80)
        logger.info("Analyzing Sentiment for COMMENTS...")
        logger.info("=" * 80)
        NUM_FEATURES = 5000  # Matching task1_sentiment_classification.py
        
        if 'body' in comments_df.columns:
            comments_with_sentiment, comments_feature_model = add_sentiment_scores_spark(
                comments_df, 'body', sentiment_udf, num_features=NUM_FEATURES
            )
            # Include processed_text for AI tool mention detection in AskReddit baseline
            comments_agg = comments_with_sentiment.select(
                col("subreddit"),
                col("created_utc"),
                col("sentiment"),
                col("score"),
                col("id"),
                col("processed_text")
            )
        else:
            logger.warning("'body' column not found in comments, skipping sentiment analysis")
            comments_agg = comments_df.select(
                col("subreddit"),
                col("created_utc"),
                lit(0.0).alias("sentiment"),
                col("score"),
                col("id"),
                lit("").alias("processed_text")
            )
            comments_feature_model = None
        
        # Analyze sentiment for submissions
        logger.info("\n" + "=" * 80)
        logger.info("Analyzing Sentiment for SUBMISSIONS...")
        logger.info("=" * 80)
        if 'title' in submissions_df.columns and 'selftext' in submissions_df.columns:
            # Combine title and selftext
            submissions_df = submissions_df.withColumn(
                "combined_text",
                concat_ws(" ",
                         coalesce(col("title"), lit("")),
                         coalesce(col("selftext"), lit("")))
            )
            submissions_with_sentiment, submissions_feature_model = add_sentiment_scores_spark(
                submissions_df, 'combined_text', sentiment_udf, num_features=NUM_FEATURES
            )
            submissions_agg = submissions_with_sentiment.select(
                col("subreddit"),
                col("created_utc"),
                col("sentiment"),
                col("score"),
                col("id"),
                col("processed_text")
            )
        elif 'title' in submissions_df.columns:
            logger.info("Using only 'title' column for submissions")
            submissions_with_sentiment, submissions_feature_model = add_sentiment_scores_spark(
                submissions_df, 'title', sentiment_udf, num_features=NUM_FEATURES
            )
            submissions_agg = submissions_with_sentiment.select(
                col("subreddit"),
                col("created_utc"),
                col("sentiment"),
                col("score"),
                col("id"),
                col("processed_text")
            )
        else:
            logger.warning("No text columns found in submissions, skipping sentiment analysis")
            submissions_agg = submissions_df.select(
                col("subreddit"),
                col("created_utc"),
                lit(0.0).alias("sentiment"),
                col("score"),
                col("id"),
                lit("").alias("processed_text")
            )
            submissions_feature_model = None
        
        # Combine comments and submissions
        logger.info("\nCombining comments and submissions...")
        combined_df = comments_agg.union(submissions_agg)
        # Cache combined data as it will be used multiple times
        combined_df = combined_df.cache()
        logger.info("Combined dataset ready (counting skipped for performance)")
        
        # Categorize subreddits
        logger.info("\nCategorizing subreddits...")
        combined_df = categorize_subreddits_spark(combined_df)
        
        # Aggregate sentiment by time
        logger.info("\n" + "=" * 80)
        logger.info("Aggregating Sentiment by Time Period...")
        logger.info("=" * 80)
        sentiment_agg = aggregate_sentiment_by_time_spark(
            combined_df, 
            time_period=args.time_period,
            text_column='processed_text'
        )
        
        # Show sample results
        logger.info("\nSample aggregated results:")
        sentiment_agg.show(10, truncate=False)
        
        # Print summary statistics
        print_summary_statistics(sentiment_agg)
        
        # Save results
        logger.info("\n" + "=" * 80)
        logger.info("Saving Results")
        logger.info("=" * 80)
        save_results_to_s3_or_local(
            spark, 
            sentiment_agg, 
            net_id=args.net_id,
            output_dir=args.output_dir
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Sentiment Analysis completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {args.output_dir}/")
        
        spark.stop()
        return 0
        
    except Exception as e:
        logger.exception(f"Error during sentiment analysis: {str(e)}")
        if 'spark' in locals():
            spark.stop()
        return 1


if __name__ == "__main__":
    sys.exit(main())



