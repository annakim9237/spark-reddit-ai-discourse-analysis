#!/usr/bin/env python3
"""
reddit_nlp_ai_tools_cluster.py

Reddit AI Tools Emotion + Sentiment Radar (CLUSTER VERSION)

This script (cluster mode):

- Connects to a Spark cluster (Spark master URL passed as arg or via MASTER_PRIVATE_IP).
- Reads RAW Reddit comments OR submissions parquet from S3.
- Performs Spark NLP preprocessing (cleaning + tokenization + lemmatization) ON CLUSTER.
- Assumes AI tools are encoded in the `subreddit` column:
    - "ChatGPT"
    - "OpenAI"
    - "GPT4"
    - "ClaudeAI"
    - "PerplexityAI"
- Filters to those subreddits and creates a `Tool` column.
- Adds:
    - sentiment (positive / negative / neutral) via ViveknSentimentModel
    - emotion_class via a lightweight keyword-based heuristic over tokens
- Aggregates Tool × emotion_class:
    - Activity (count)
    - activity_percent (within each Tool)
- Produces:
    - ai_tools_emotion_summary.csv  (includes emotion_class + sentiment)
    - ai_tools_emotion_radar.png   (radar chart of emotions for each AI tool)
- Copies outputs to ~/reddit-nlp-cluster/ for easy scp.

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
from pyspark.sql.types import TimestampType, StringType
from pyspark.ml import Pipeline

import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import (
    Tokenizer,
    Normalizer,
    StopWordsCleaner,
    LemmatizerModel,
    ViveknSentimentModel,
)

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

# AI tools of interest (for filtering + radar legend)
AI_TOOLS = ["ChatGPT", "OpenAI", "GPT4", "ClaudeAI", "PerplexityAI"]

# ================================================================
# LIGHTWEIGHT EMOTION LEXICON (TOKEN-BASED HEURISTIC)
# ================================================================
# This is intentionally small + interpretable. You can expand as needed.
EMOTION_KEYWORDS = {
    "joy": {
        "happy", "glad", "excited", "awesome", "amazing", "great", "fun",
        "enjoy", "enjoying", "enjoyed", "cool", "yay", "love", "lmao",
    },
    "love": {
        "love", "adore", "admire", "cherish", "favorite", "favourite",
        "loving", "loved", "wholesome",
    },
    "anger": {
        "angry", "mad", "furious", "rage", "raging", "annoyed", "annoying",
        "pissed", "hate", "hating", "hated", "frustrated", "frustrating",
        "stupid", "dumb", "idiot", "idiotic",
    },
    "fear": {
        "afraid", "scared", "terrified", "worried", "anxious", "anxiety",
        "fear", "fearful", "panic", "panicking", "danger",
    },
    "sadness": {
        "sad", "depressed", "depressing", "unhappy", "crying", "cried",
        "tears", "miserable", "lonely", "heartbroken",
    },
    "surprise": {
        "surprised", "shocked", "shocking", "wow", "omg", "unexpected",
        "suddenly", "didnt", "didn't", "never thought",
    },
}

EMOTION_LABELS = ["joy", "love", "anger", "fear", "sadness", "surprise", "neutral"]

# ================================================================
# SPARK SESSION BUILDER (CLUSTER)
# ================================================================
def build_spark(master_url: str, app_name: str = "Reddit_AITools_Cluster") -> SparkSession:
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
    Does NOT write to S3; used inline for AI tools analysis.
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
# AI TOOLS FROM SUBREDDIT
# ================================================================
def select_ai_tools_from_subreddit(df):
    """
    Filter rows where the `subreddit` corresponds to one of the AI tools
    and create a canonical `Tool` column.

    Assumes `subreddit` contains tool names like:
        ChatGPT, OpenAI, GPT4, ClaudeAI, PerplexityAI
    (case-insensitive)
    """
    if "subreddit" not in df.columns:
        raise ValueError("Expected 'subreddit' column but it was not found.")

    logger.info("Selecting AI tool rows based on subreddit...")

    df_tools = df.withColumn("subreddit_upper", F.upper(F.col("subreddit")))

    df_tools = df_tools.withColumn(
        "Tool",
        F.when(F.col("subreddit_upper") == "CHATGPT", F.lit("ChatGPT"))
         .when(F.col("subreddit_upper") == "OPENAI", F.lit("OpenAI"))
         .when(F.col("subreddit_upper") == "GPT4", F.lit("GPT4"))
         .when(F.col("subreddit_upper") == "CLAUDEAI", F.lit("ClaudeAI"))
         .when(F.col("subreddit_upper") == "PERPLEXITYAI", F.lit("PerplexityAI"))
         .otherwise(F.lit(None))
    ).drop("subreddit_upper")

    df_tools = df_tools.filter(F.col("Tool").isin(AI_TOOLS))

    count = df_tools.count()
    logger.info(f"✓ Found {count:,} rows where subreddit represents an AI tool.")
    return df_tools

# ================================================================
# SENTIMENT PIPELINE (VIVEKN)
# ================================================================
def build_sentiment_pipeline(input_col: str = "processed_text") -> Pipeline:
    """
    Build a Spark NLP sentiment pipeline using ViveknSentimentModel.

    input_col should be a STRING column (e.g., 'processed_text').
    """
    logger.info(f"Building sentiment pipeline with Vivekn on '{input_col}'")

    document_assembler = (
        DocumentAssembler()
        .setInputCol(input_col)
        .setOutputCol("sent_document")
    )

    token = (
        Tokenizer()
        .setInputCols(["sent_document"])
        .setOutputCol("sent_token")
    )

    vivekn = (
        ViveknSentimentModel.pretrained("sentiment_vivekn", "en")
        .setInputCols(["sent_document", "sent_token"])
        .setOutputCol("sentiment_result")
    )

    pipeline = Pipeline(stages=[document_assembler, token, vivekn])
    logger.info("✓ Sentiment pipeline created.")
    return pipeline


def add_sentiment(df, input_col: str = "processed_text"):
    """
    Run Vivekn sentiment pipeline and add 'sentiment' column.

    Expects:
        - df[input_col]: STRING column

    Produces:
        - df with 'sentiment' column: 'positive' / 'negative' / 'neutral'
    """
    if input_col not in df.columns:
        raise ValueError(
            f"Input column '{input_col}' not found for sentiment pipeline. "
            f"Available columns: {df.columns}"
        )

    logger.info(f"Running Vivekn sentiment pipeline on '{input_col}'...")
    pipeline = build_sentiment_pipeline(input_col=input_col)
    model = pipeline.fit(df)
    result = model.transform(df)

    # sentiment_result.result is array<string> with first element as label
    result = result.withColumn(
        "raw_sentiment",
        F.element_at(F.col("sentiment_result.result"), 1)
    )

    # Normalize to lower-case + handle nulls
    result = result.withColumn(
        "sentiment",
        F.when(F.col("raw_sentiment").isNull(), F.lit("neutral"))
         .otherwise(F.lower(F.col("raw_sentiment")))
    )

    # Cleanup annotation columns
    result = result.drop("sentiment_result", "raw_sentiment", "sent_document", "sent_token")

    logger.info("✓ Added 'sentiment' column using Vivekn.")
    return result

# ================================================================
# EMOTION HEURISTIC (TOKEN-BASED)
# ================================================================
def create_emotion_udf():
    """
    Create a PySpark UDF that assigns an emotion_class label
    based on token overlaps with EMOTION_KEYWORDS.
    """
    lexicon = {k: set(v) for k, v in EMOTION_KEYWORDS.items()}
    emotions = list(lexicon.keys())

    def detect_emotion(tokens):
        if not tokens:
            return "neutral"

        counts = {emo: 0 for emo in emotions}
        for t in tokens:
            if not t:
                continue
            w = t.lower()
            for emo, words in lexicon.items():
                if w in words:
                    counts[emo] += 1

        # pick emotion with max count; if all zero -> neutral
        best_emo = None
        best_count = 0
        for emo, c in counts.items():
            if c > best_count:
                best_count = c
                best_emo = emo

        if best_count == 0 or best_emo is None:
            return "neutral"
        return best_emo

    return F.udf(detect_emotion, StringType())


EMOTION_UDF = create_emotion_udf()


def add_emotion_class(df, tokens_col: str = "tokens"):
    """
    Add an 'emotion_class' column using a lightweight keyword-based heuristic.

    Expects:
        - df[tokens_col]: array<string>

    Produces:
        - df with 'emotion_class' in EMOTION_LABELS
    """
    if tokens_col not in df.columns:
        raise ValueError(
            f"Tokens column '{tokens_col}' not found for emotion heuristic. "
            f"Available columns: {df.columns}"
        )

    logger.info(f"Assigning emotion_class using keyword heuristic on '{tokens_col}'...")
    df = df.withColumn("emotion_class", EMOTION_UDF(F.col(tokens_col)))
    logger.info("✓ Added 'emotion_class' via heuristic.")
    return df

# ================================================================
# AGGREGATION: Tool × emotion_class
# ================================================================
def compute_ai_tools_emotion_summary(df):
    """
    Compute per-tool per-emotion_class aggregates,
    grouping by 'Tool' and 'emotion_class'.

    Expected columns in `df`:
        - Tool (string)
        - emotion_class (string)
    Optional columns:
        - score (for avg_score diagnostic)
        - token_count (for avg_tokens diagnostic)
        - sentiment (stored in CSV, but not used here)

    Returns:
        Spark DataFrame with columns:
            Tool, emotion_class, Activity, [avg_score], [avg_tokens]
    """
    required = {"Tool", "emotion_class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in DF: {missing}")

    agg_exprs = [F.count("*").alias("Activity")]

    if "score" in df.columns:
        agg_exprs.append(F.avg("score").alias("avg_score"))
    if "token_count" in df.columns:
        agg_exprs.append(F.avg("token_count").alias("avg_tokens"))

    summary = df.groupBy("Tool", "emotion_class").agg(*agg_exprs)
    logger.info("✓ Computed AI tools emotion summary (Tool × emotion_class).")
    return summary


def to_emotion_percentage_frame(summary_df) -> pd.DataFrame:
    """
    Convert Spark summary DF (Tool × emotion_class) to pandas and compute
    activity_percent per Tool.

    Input Spark DF columns:
        - Tool
        - emotion_class
        - Activity

    Returns:
        pandas DataFrame with columns:
            Tool, emotion_class, Activity, Activity_total, activity_percent, ...
    """
    pdf = summary_df.toPandas()

    if pdf.empty:
        logger.warning("AI tools emotion summary DataFrame is empty; nothing to plot.")
        return pdf

    # Drop rows with missing tool labels
    pdf = pdf[pdf["Tool"].notna() & (pdf["Tool"] != "")]
    if pdf.empty:
        logger.warning("No valid Tool labels after filtering; nothing to summarize.")
        return pdf

    # Total activity per tool
    totals = (
        pdf.groupby("Tool")["Activity"]
        .sum()
        .reset_index()
        .rename(columns={"Activity": "Activity_total"})
    )

    pdf = pdf.merge(totals, on="Tool", how="left")
    pdf["activity_percent"] = pdf["Activity"] / pdf["Activity_total"]

    logger.info("✓ Computed activity_percent per (Tool, emotion_class).")
    return pdf

# ================================================================
# RADAR CHART: emotions per AI tool
# ================================================================
def plot_ai_tools_emotion_radar(
    ai_summary_emotion_df: pd.DataFrame,
    output_path: str,
    tools_order=None,
    emotion_order=None,
    title: str = "AI Tools Emotion Radar"
) -> str:
    """
    Create a radar chart comparing emotion distributions across AI tools.

    Expects ai_summary_emotion_df with columns:
        - Tool
        - emotion_class
        - activity_percent  (0–1)

    Args:
        ai_summary_emotion_df: pandas DataFrame as returned by to_emotion_percentage_frame().
        output_path: where to save the PNG (e.g. "./outputs/ai_tools_emotion_radar.png").
        tools_order: optional list to control order of tools on the legend.
        emotion_order: optional list to control order of emotions around the circle.
        title: figure title.

    Returns:
        The output_path (or "" if nothing was plotted).
    """
    df = ai_summary_emotion_df.copy()

    if df.empty:
        logger.warning("Empty DataFrame; skipping radar chart.")
        return ""

    # Normalize dtypes
    df["Tool"] = df["Tool"].astype(str)
    df["emotion_class"] = df["emotion_class"].astype(str)
    df["activity_percent"] = pd.to_numeric(df["activity_percent"], errors="coerce").fillna(0.0)

    # Filter to AI tools of interest
    if tools_order is None:
        tools_order = [t for t in AI_TOOLS if t in df["Tool"].unique()]
    else:
        tools_order = [t for t in tools_order if t in df["Tool"].unique()]

    if not tools_order:
        logger.warning("No matching tools found in DataFrame; skipping radar chart.")
        return ""

    # Emotion order
    if emotion_order is None:
        # Use EMOTION_LABELS order but only those that appear
        available = df["emotion_class"].unique().tolist()
        emotion_order = [e for e in EMOTION_LABELS if e in available]
    else:
        emotion_order = [e for e in emotion_order if e in df["emotion_class"].unique()]

    if not emotion_order:
        logger.warning("No valid emotion_class values found; skipping radar chart.")
        return ""

    # Close the polygon by repeating the first emotion at the end
    categories = [*emotion_order, emotion_order[0]]
    num_points = len(categories)

    # Helper to get series for a tool
    def series_for(tool_name: str) -> list[float]:
        sub = (
            df[df["Tool"] == tool_name]
            .set_index("emotion_class")
            .reindex(emotion_order)
        )
        if sub.empty:
            return []

        vals = sub["activity_percent"].fillna(0.0).tolist()
        vals = [*vals, vals[0]]  # close polygon
        return vals

    tool_series = {}
    for t in tools_order:
        vals = series_for(t)
        if vals:
            tool_series[t] = vals
        else:
            logger.warning(f"No emotion rows found for tool: {t}; it will be skipped in the radar chart.")

    if not tool_series:
        logger.warning("No data to plot after building tool series; skipping radar chart.")
        return ""

    # Angles for each category
    angles = np.linspace(start=0, stop=2 * np.pi, num=num_points)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(polar=True)

    # Determine max radius
    max_val = max(max(v) for v in tool_series.values())
    if max_val <= 0:
        max_val = 1.0

    for tool_name, vals in tool_series.items():
        ax.plot(angles, vals, label=tool_name)
        ax.fill(angles, vals, alpha=0.1)

    ax.set_title(title, size=18, y=1.05)

    # Use only first N-1 labels (last angle is the repeated one)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotion_order)

    ax.set_ylim(0, max_val * 1.1)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Saved AI tools emotion radar chart to: {output_path}")
    return output_path

# ================================================================
# DROP FOLDER COPY (FOR SCP)
# ================================================================
def copy_to_drop_folder(files: list[str]):
    """Copy files to ~/reddit-nlp-cluster for easy scp."""
    drop = Path.home() / "reddit-nlp-cluster"
    drop.mkdir(exist_ok=True)
    for f in files:
        if f and os.path.exists(f):
            dest = drop / Path(f).name
            try:
                shutil.copy2(f, dest)
                logger.info(f"  Copied {f} to {dest}")
            except Exception as e:
                logger.warning(f"  Failed to copy {f} to {dest}: {e}")

# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reddit AI Tools Emotion + Sentiment Radar (CLUSTER, INLINE NLP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Comments only, full dataset, using MASTER_PRIVATE_IP env
  export MASTER_PRIVATE_IP=10.0.1.5
  uv run python reddit_nlp_ai_tools_cluster.py \\
      spark://$MASTER_PRIVATE_IP:7077 \\
      --data-type comments \\
      --s3-input s3a://ea973-dsan6000-datasets-final/project/reddit/parquet/ \\
      --output-dir ./outputs

  # Submissions only, 5% sample, explicit master URL, no lemmatization
  uv run python reddit_nlp_ai_tools_cluster.py \\
      spark://10.0.1.5:7077 \\
      --data-type submissions \\
      --s3-input s3a://ea973-dsan6000-datasets-final/project/reddit/parquet/ \\
      --sample 0.05 \\
      --no-lemmatization \\
      --output-dir ./outputs
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
        default="s3a://ea973-dsan6000-datasets-final/project/reddit/parquet/",
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
        help="Directory for outputs (CSV + radar PNG)",
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
            print("Usage: uv run python reddit_nlp_ai_tools_cluster.py spark://MASTER_IP:7077 [options]")
            print("   or: export MASTER_PRIVATE_IP=xxx.xxx.xxx.xxx")
            print("=" * 70)
            return 1

    if args.sample and not (0 < args.sample <= 1):
        parser.error("Sample fraction must be between 0 and 1")

    logger.info("=" * 70)
    logger.info("REDDIT AI TOOLS EMOTION + SENTIMENT RADAR (CLUSTER MODE)")
    logger.info("=" * 70)
    logger.info(f"Spark Master      : {master_url}")
    logger.info(f"S3 Input Base     : {args.s3_input}")
    logger.info(f"Data Type         : {args.data_type}")
    logger.info(f"Sample Fraction   : {args.sample if args.sample else 'FULL DATASET'}")
    logger.info(f"Output Dir        : {args.output_dir}")
    logger.info(f"Remove Stopwords  : {not args.no_stopwords}")
    logger.info(f"Apply Lemmatize   : {not args.no_lemmatization}")
    logger.info("=" * 70)

    spark = None
    start_time = datetime.now()

    try:
        # 1. Spark session
        spark = build_spark(master_url)

        # 2. Preprocess Reddit (comments OR submissions) on cluster
        reddit_df = preprocess_reddit(
            spark,
            args.s3_input,
            args.data_type,
            sample_fraction=args.sample,
            remove_stopwords=not args.no_stopwords,
            apply_lemmatization=not args.no_lemmatization,
        )

        # # 3. Monthly aggregation just for diagnostics/logging
        # try:
        #     monthly_df = aggregate_reddit_activity_monthly(reddit_df)
        #     logger.info("Head of monthly Reddit activity (diagnostic):")
        #     logger.info("\n" + monthly_df.head().to_string(index=False))
        # except Exception as e:
        #     logger.warning(f"Monthly aggregation skipped (missing cols?): {e}")

        # 4. Select AI tools based on subreddit
        reddit_ai = select_ai_tools_from_subreddit(reddit_df)
        if reddit_ai.count() == 0:
            logger.warning("No AI tool subreddits found; nothing to summarize.")
            return 0

        # 5. Add sentiment via Vivekn
        reddit_ai = add_sentiment(reddit_ai, input_col="processed_text")

        # 6. Add emotion_class via keyword-based heuristic
        reddit_ai = add_emotion_class(reddit_ai, tokens_col="tokens")

        # 7. Compute summary (Tool × emotion_class)
        ai_emotion_summary_spark = compute_ai_tools_emotion_summary(reddit_ai)

        # 8. Convert to pandas + compute activity_percent
        ai_summary_emotion_df = to_emotion_percentage_frame(ai_emotion_summary_spark)

        if ai_summary_emotion_df.empty:
            logger.warning("No AI tools data after emotion aggregation; nothing to save or plot.")
            return 0

        # 9. Save CSV (include sentiment by joining back if needed)
        # For now, we save just the aggregated summary + you still have sentiment on reddit_ai
        csv_path = os.path.join(args.output_dir, "ai_tools_emotion_summary.csv")
        ai_summary_emotion_df.to_csv(csv_path, index=False)
        logger.info(f"Saved AI tools emotion summary CSV to: {csv_path}")

        # 10. Plot emotions radar chart
        radar_path = os.path.join(args.output_dir, "ai_tools_emotion_radar.png")
        plot_ai_tools_emotion_radar(
            ai_summary_emotion_df,
            radar_path,
            tools_order=AI_TOOLS,
        )

        # 11. Copy outputs to drop folder for easy scp
        copy_to_drop_folder([csv_path, radar_path])

        elapsed = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 70)
        print("✅ CLUSTER AI TOOLS EMOTION + SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"Total execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"Outputs saved to: {args.output_dir}")
        print("Files copied to: ~/reddit-nlp-cluster/")
        print("=" * 70)

        logger.info("\n" + "=" * 80)
        logger.info("✅ CLUSTER AI TOOLS EMOTION + SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info(f"Total execution time: {elapsed:.1f} seconds")
        logger.info(f"Outputs saved to: {args.output_dir}")
        logger.info("Files copied to: ~/reddit-nlp-cluster/")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\n❌ Error during AI tools cluster analysis: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1

    finally:
        if spark:
            spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    sys.exit(main())
