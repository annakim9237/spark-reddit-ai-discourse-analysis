#!/usr/bin/env python3
"""
Question 5: How has sentiment toward different generative AI tools shifted over time across subreddits?

Spark Cluster Version - Optimized for Speed
Key Optimizations:
1. Removed expensive TF-IDF/HashingTF computation (not needed for lexicon analysis).
2. Replaced ML Pipeline with efficient Regex-based text cleaning.
3. Implemented Vectorized Pandas UDF for fast dictionary lookup.
4. Added data filtering to remove empty/short comments early.
5. Tuned partitions to 800 to handle data skew and memory pressure.

Compatible with: Spark 3.4.4 / PySpark 3.4.x
"""


import logging
import os
import sys
from typing import Tuple, List

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, lower, regexp_replace, from_unixtime, to_date, 
    date_format, year, quarter, avg, stddev, count, 
    concat_ws, coalesce, lit, udf, when, length  # <--- FIXED: Added length here
)
from pyspark.sql.types import DoubleType, StringType, ArrayType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA

# Pandas UDF for Spark 3.4+
try:
    from pyspark.sql.functions import pandas_udf
    HAS_PANDAS_UDF = True
except ImportError:
    HAS_PANDAS_UDF = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_spark_session(master_url: str = "local[*]", app_name: str = "NLP_Final") -> SparkSession:
    logger.info(f"Creating Spark session ({master_url})...")
    builder = SparkSession.builder.appName(app_name).master(master_url)
    
    # Config optimized for NLP tasks on 8GB nodes
    builder = (builder
        .config("spark.executor.memory", "6g")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "400")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")
    )
    return builder.getOrCreate()


def load_lexicons(script_dir: str) -> Tuple[set, set]:
    """
    Load sentiment words.
    FIXED: Checks both script directory and parent directory.
    """
    # Try multiple possible locations
    possible_roots = [script_dir, os.path.dirname(script_dir), os.getcwd()]
    
    pos_words = set()
    neg_words = set()
    
    pos_loaded = False
    neg_loaded = False

    for root in possible_roots:
        pos_path = os.path.join(root, "positive_words.txt")
        neg_path = os.path.join(root, "bad_words_cleaned.txt")
        
        if not pos_loaded and os.path.exists(pos_path):
            logger.info(f"Loading positive words from: {pos_path}")
            with open(pos_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    w = line.strip().lower()
                    if w and not w.startswith(';'): pos_words.add(w)
            pos_loaded = True
            
        if not neg_loaded and os.path.exists(neg_path):
            logger.info(f"Loading negative words from: {neg_path}")
            with open(neg_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    w = line.strip().lower()
                    if w and not w.startswith(';'): neg_words.add(w)
            neg_loaded = True
            
        if pos_loaded and neg_loaded:
            break
    
    if not pos_loaded: logger.warning("⚠️ Positive words file NOT found in any common paths.")
    if not neg_loaded: logger.warning("⚠️ Negative words file NOT found in any common paths.")
            
    return pos_words, neg_words


def read_data(spark: SparkSession, net_id: str, data_type: str, local_mode: bool, data_dir: str) -> DataFrame:
    """Robust data reader."""
    if local_mode:
        path = os.path.join(data_dir, f"{data_type}.parquet")
        return spark.read.parquet(path)
    
    paths = [
        f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/{data_type}/",
        f"s3a://{net_id}-dsan6000-datasets/reddit/parquet/{data_type}/"
    ]
    for p in paths:
        try:
            logger.info(f"Reading: {p}")
            df = spark.read.parquet(p)
            if df.limit(1).count() > 0: return df
        except: continue
    raise RuntimeError(f"Could not read {data_type} from S3.")


def preprocess_text_basic(df: DataFrame, text_col: str) -> DataFrame:
    """
    Fast Regex cleaning for Sentiment Analysis.
    """
    logger.info("Preprocessing text (Regex)...")
    # Keep only letters and spaces
    clean_pattern = r"http\S+|www\S+|\[.*?\]|[^a-zA-Z\s]"
    df = df.withColumn("processed_text", lower(regexp_replace(col(text_col), clean_pattern, "")))
    df = df.withColumn("processed_text", regexp_replace(col("processed_text"), r"\s+", " "))
    
    # Filter short texts (Requires 'length' function)
    return df.filter(length(col("processed_text")) > 3)


def perform_sentiment_analysis(df: DataFrame, pos_words: set, neg_words: set) -> DataFrame:
    """
    Apply Vectorized UDF for Sentiment Analysis.
    """
    logger.info("Performing Sentiment Analysis...")
    # Convert sets to lists for broadcasting if needed, but sets work in closure
    pos_set = pos_words
    neg_set = neg_words

    if HAS_PANDAS_UDF:
        @pandas_udf(DoubleType())
        def analyze_sentiment(texts: pd.Series) -> pd.Series:
            def get_score(text):
                if not text: return 0.0
                try:
                    words = set(text.split())
                    if not words: return 0.0
                    p = len(words.intersection(pos_set))
                    n = len(words.intersection(neg_set))
                    if p + n == 0: return 0.0
                    return float((p - n) / (p + n))
                except: return 0.0
            return texts.apply(get_score)
        
        return df.withColumn("sentiment", analyze_sentiment(col("processed_text")))
    else:
        # Fallback
        def simple_sentiment(text):
            if not text: return 0.0
            words = text.split()
            p = sum(1 for w in words if w in pos_set)
            n = sum(1 for w in words if w in neg_set)
            return float((p - n) / (p + n)) if (p+n) > 0 else 0.0
            
        return df.withColumn("sentiment", udf(simple_sentiment, DoubleType())(col("processed_text")))


def perform_topic_modeling(df: DataFrame, num_topics: int = 5, max_iter: int = 20):
    """
    Perform LDA Topic Modeling (Improved with Custom Stopwords).
    """
    logger.info("Performing Topic Modeling (LDA) with Custom Filtering...")
    
    # 1. Sampling
    sample_fraction = 0.1
    df_sample = df.sample(withReplacement=False, fraction=sample_fraction, seed=42).limit(20000)
    df_sample.cache()
    
    logger.info(f"LDA Training on sample size: {df_sample.count()}")
    
    # 2. Pipeline for Vectorization
    tokenizer = Tokenizer(inputCol="processed_text", outputCol="tokens")
    
    # --- UPDATE: Define extensive custom stopwords ---
    # 1. Standard English stopwords
    base_stopwords = StopWordsRemover.loadDefaultStopWords("english")
    
    # 2. Noise words caused by regex cleaning (no apostrophes)
    cleaning_noise = ["im", "dont", "cant", "wont", "didnt", "isnt", "thats", "youre", "ive"]
    
    # 3. Reddit specific noise & bots
    reddit_noise = ["http", "https", "com", "www", "reddit", "subreddit", "post", "comment", 
                    "bot", "moderator", "mod", "deleted", "removed", "please", "contact", "message", "action"]
    
    # 4. High frequency filler words (Conversational)
    common_fillers = ["like", "just", "get", "got", "one", "time", "people", "think", "know", 
                      "make", "really", "good", "bad", "much", "many", "well", "see", "way", 
                      "use", "also", "even", "now", "go", "going", "want", "would", "could"]
    
    all_stopwords = base_stopwords + cleaning_noise + reddit_noise + common_fillers
    
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    remover.setStopWords(all_stopwords)  # Set the custom list
    
    # --- UPDATE: Tweak CountVectorizer ---
    # maxDF=0.4: Ignore words appearing in more than 40% of docs (removes background noise)
    # minDF=5.0: Ignore words appearing in fewer than 5 docs (removes typos)
    cv = CountVectorizer(inputCol="filtered_tokens", outputCol="features", 
                         vocabSize=2000, minDF=5.0, maxDF=0.4)
    
    pipeline = Pipeline(stages=[tokenizer, remover, cv])
    model = pipeline.fit(df_sample)
    vectorized_df = model.transform(df_sample)
    
    # 3. Train LDA Model (Increased iterations slightly for better quality)
    lda = LDA(k=num_topics, maxIter=max_iter, featuresCol="features")
    lda_model = lda.fit(vectorized_df)
    
    # 4. Extract Topics
    logger.info("Extracting Topics...")
    vocab = model.stages[-1].vocabulary
    topics = lda_model.describeTopics(maxTermsPerTopic=10) # Get top 10 words to see clearly
    
    topics_list = topics.collect()
    result_topics = []
    
    for row in topics_list:
        topic_idx = row['topic']
        term_indices = row['termIndices']
        
        terms = [vocab[idx] for idx in term_indices]
        topic_desc = ", ".join([f"{term}" for term in terms])
        result_topics.append({"topic_id": topic_idx, "terms": topic_desc})
        
        logger.info(f"Topic {topic_idx}: {topic_desc}")
        
    df_sample.unpersist()
    return pd.DataFrame(result_topics)


def categorize_subreddits(df: DataFrame) -> DataFrame:
    """Categorize subreddits into groups."""
    def categorize(sub):
        if not sub: return "Other"
        s = sub.lower()
        if 'askreddit' in s: return 'Baseline'
        if any(x in s for x in ['chatgpt', 'openai', 'gpt']): return 'Text AI'
        if any(x in s for x in ['midjourney', 'stable']): return 'Creative AI'
        if any(x in s for x in ['machine', 'data', 'tech']): return 'Research/Tech'
        return 'Other'
    return df.withColumn("ai_category", udf(categorize, StringType())(col("subreddit")))


def aggregate_results(df: DataFrame, time_period: str) -> DataFrame:
    """Aggregate sentiment by time and category."""
    df = df.withColumn("date", to_date(from_unixtime(col("created_utc"))))
    
    if time_period == 'monthly':
        df = df.withColumn("period", date_format(col("date"), "yyyy-MM"))
    else:
        df = df.withColumn("period", date_format(col("date"), "yyyy-MM-dd"))
        
    return df.groupBy("ai_category", "period").agg(
        avg("sentiment").alias("avg_sentiment"),
        count("id").alias("post_count")
    ).orderBy("period")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--net-id", type=str, required=True)
    parser.add_argument("--master", type=str, default="local[*]")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--local-mode", action="store_true")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--time-period", type=str, default="monthly")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Start Spark
    spark = create_spark_session(args.master)
    
    try:
        # 2. Load Lexicons (using improved path detection)
        pos_words, neg_words = load_lexicons(script_dir)
        if not pos_words: 
            logger.warning("Using fallback dummy lexicons. RESULTS WILL BE POOR.")
            pos_words, neg_words = {'good'}, {'bad'}

        # 3. Read Data
        df_comm = read_data(spark, args.net_id, "comments", args.local_mode, args.data_dir)
        df_sub = read_data(spark, args.net_id, "submissions", args.local_mode, args.data_dir)
        
        # 4. Standardize & Union
        sel_cols = ["subreddit", "created_utc", "id"]
        
        if 'body' in df_comm.columns:
            df_comm = df_comm.select(*sel_cols, col("body").alias("text"))
        else:
            df_comm = None
            
        if 'title' in df_sub.columns:
            if 'selftext' in df_sub.columns:
                df_sub = df_sub.select(*sel_cols, concat_ws(" ", coalesce("title", lit("")), coalesce("selftext", lit(""))).alias("text"))
            else:
                df_sub = df_sub.select(*sel_cols, col("title").alias("text"))
        else:
            df_sub = None
            
        if df_comm and df_sub:
            df = df_comm.union(df_sub)
        elif df_comm:
            df = df_comm
        else:
            df = df_sub
            
        if not df: raise RuntimeError("No data loaded.")
        
        # 5. Preprocess
        df = preprocess_text_basic(df, "text")
        
        # 6. Sentiment Analysis
        df = perform_sentiment_analysis(df, pos_words, neg_words)
        
        # 7. Categorization
        df = categorize_subreddits(df)
        
        # 8. Topic Modeling
        topic_df = perform_topic_modeling(df, num_topics=5)
        
        # 9. Aggregation
        agg_df = aggregate_results(df, args.time_period)
        
        # 10. Save Results
        out_path = args.output_dir or f"s3a://{args.net_id}-dsan6000-datasets/project/nlp_results/"
        logger.info(f"Saving results to {out_path}")
        
        spark.createDataFrame(agg_df.toPandas()).write.mode("overwrite").csv(os.path.join(out_path, "sentiment_trends"), header=True)
        spark.createDataFrame(topic_df).write.mode("overwrite").csv(os.path.join(out_path, "discovered_topics"), header=True)
        
        logger.info("NLP Analysis Completed Successfully.")
        
    except Exception as e:
        logger.error(f"Job Failed: {e}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()