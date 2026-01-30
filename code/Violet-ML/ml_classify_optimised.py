#!/usr/bin/env python3
"""
ML Question: What are the comparative strengths and weaknesses of Logistic Regression, 
Random Forests, SVMs, and Neural Networks when classifying Reddit posts into topical 
categories (Text AI, Creative AI, Research/Tech, Other)?

Spark Cluster Version - Optimized for Imbalanced Data
Updates:
1. Exact Subreddit Mapping based on user database.
2. Smart Sampling: Downsample 'Other', Upsample 'Creative AI'.
3. Increased TF-IDF features (100 -> 1000).
4. Deep Neural Network architecture.
5. Fixed LinearSVC multi-class issue using OneVsRest.
"""

import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, when, length, regexp_replace, lower, hour, 
    from_unixtime, udf, lit, concat_ws, split, size, coalesce, count
)
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, Tokenizer, 
    HashingTF, IDF, StringIndexer
)
from pyspark.ml.classification import (
    RandomForestClassifier, LogisticRegression, 
    LinearSVC, MultilayerPerceptronClassifier, OneVsRest
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_spark_session(master_url: str = "local[*]", app_name: str = "ML_Topic_Classification",
                         local_mode: bool = True) -> SparkSession:
    """
    Create a Spark session optimized for ML processing.
    """
    logger.info(f"Creating Spark session (master: {master_url}, local_mode: {local_mode})...")
    
    # Stop any existing session
    try:
        existing = SparkSession.getActiveSession()
        if existing:
            existing.stop()
    except:
        pass

    builder = SparkSession.builder.appName(app_name)
    
    if master_url and master_url != "local[*]":
        builder = builder.master(master_url)
    
    if local_mode:
        builder = (builder
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true"))
    else:
        # Optimized for cluster (8GB env)
        builder = (builder
            .config("spark.executor.memory", "6g")
            .config("spark.driver.memory", "2g")
            .config("spark.driver.maxResultSize", "2g")
            .config("spark.executor.cores", "2")
            .config("spark.sql.adaptive.enabled", "true")
            # S3 Configuration
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                    "com.amazonaws.auth.InstanceProfileCredentialsProvider")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer"))
    
    return builder.getOrCreate()


def read_data_from_local_or_s3(spark: SparkSession, net_id: str = None,
                               data_type: str = "submissions",
                               local_mode: bool = True,
                               local_data_dir: str = None) -> DataFrame:
    """
    Read Reddit data from local files or S3.
    """
    if local_mode and local_data_dir:
        local_path = os.path.join(local_data_dir, f"{data_type}.parquet")
        if os.path.exists(local_path):
            logger.info(f"Reading from local file: {local_path}")
            return spark.read.parquet(local_path)
        raise FileNotFoundError(f"Local data file not found: {local_path}")
    
    if not net_id:
        raise ValueError("net_id is required when not in local_mode")
    
    # Priority paths
    paths = [
        f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/{data_type}/",
        f"s3a://{net_id}-dsan6000-datasets/reddit/parquet/{data_type}/"
    ]
    
    for path in paths:
        try:
            logger.info(f"Attempting read: {path}")
            df = spark.read.parquet(path)
            logger.info(f"Success! Loaded {df.count():,} rows.")
            return df
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            
    raise RuntimeError("Could not read data from any S3 path.")


def categorize_subreddit_udf(subreddit: str) -> str:
    """
    Categorize based on specific Subreddit names from the database.
    """
    if not subreddit:
        return "Other"
    
    sub = str(subreddit).lower().strip()
    
    # 1. Text AI Group
    if sub in ['chatgpt', 'openai', 'claudeai', 'gpt4', 'localllama']:
        return 'Text AI'
        
    # 2. Creative AI Group
    elif sub in ['stablediffusion']:
        return 'Creative AI'
        
    # 3. Research & Tech Group
    elif sub in ['machinelearning', 'datascience', 'computerscience', 
                 'programming', 'singularity', 'futurology']:
        return 'Research/Tech'
        
    # 4. Explicit Other Group (Non-AI) & Default
    # Includes: askreddit, neoliberal
    else:
        return 'Other'


def engineer_features_spark(df: DataFrame) -> DataFrame:
    """
    Apply categorization and create text/numerical features.
    """
    logger.info("Engineering features...")
    
    # 1. Create target label
    categorize_func = udf(categorize_subreddit_udf, StringType())
    df_features = df.withColumn("ai_category", categorize_func(col("subreddit")))
    
    # 2. Text Processing
    df_features = df_features.withColumn(
        "combined_text",
        concat_ws(" ", coalesce(col("title"), lit("")), coalesce(col("selftext"), lit("")))
    )
    
    # Remove URLs and basic cleaning
    df_features = df_features.withColumn(
        "processed_text",
        regexp_replace(col("combined_text"), r"http\S+|www\S+|\[.*?\]", "")
    )
    
    # 3. Numerical Features
    df_features = df_features.withColumn("text_length", length(col("processed_text")))
    df_features = df_features.withColumn("word_count", size(split(col("processed_text"), "\\s+")))
    df_features = df_features.withColumn("hour_of_day", hour(from_unixtime(col("created_utc"))))
    
    # Metadata
    df_features = df_features.withColumn("has_url", when(col("url").isNotNull(), 1).otherwise(0))
    df_features = df_features.withColumn("score", coalesce(col("score"), lit(0)))
    df_features = df_features.withColumn("num_comments", coalesce(col("num_comments"), lit(0)))
    
    return df_features


def balance_dataset(df: DataFrame) -> DataFrame:
    """
    Perform Smart Sampling: Downsample 'Other', Upsample 'Creative AI'.
    """
    logger.info("Performing Smart Balancing...")
    
    # Cache input to avoid re-computing previous steps
    df.cache()
    
    # Split by category
    df_text = df.filter(col("ai_category") == "Text AI")
    df_creative = df.filter(col("ai_category") == "Creative AI")
    df_research = df.filter(col("ai_category") == "Research/Tech")
    df_other = df.filter(col("ai_category") == "Other")
    
    counts = {
        "Text AI": df_text.count(),
        "Creative AI": df_creative.count(),
        "Research/Tech": df_research.count(),
        "Other": df_other.count()
    }
    
    logger.info(f"Original Counts: {counts}")
    
    if counts["Other"] == 0:
        logger.warning("No 'Other' data found. Skipping balancing.")
        return df

    # --- Strategy ---
    # 1. Downsample 'Other' to be roughly 1.5x the size of 'Text AI' (approx 25k-30k)
    # 2. Upsample 'Creative AI' to be closer to 'Research/Tech' (approx 12k)
    
    target_other = 25000
    fraction_other = min(1.0, target_other / counts["Other"])
    
    target_creative = 12000
    fraction_creative = max(1.0, target_creative / max(1, counts["Creative AI"]))
    
    logger.info(f"Sampling 'Other' with fraction: {fraction_other:.4f}")
    logger.info(f"Sampling 'Creative AI' with fraction: {fraction_creative:.4f}")
    
    df_other_sampled = df_other.sample(withReplacement=False, fraction=fraction_other, seed=42)
    df_creative_sampled = df_creative.sample(withReplacement=True, fraction=fraction_creative, seed=42)
    
    # Combine
    df_balanced = df_text.union(df_creative_sampled).union(df_research).union(df_other_sampled)
    
    new_counts = df_balanced.groupBy("ai_category").count().collect()
    logger.info("Balanced Counts:")
    for row in new_counts:
        logger.info(f"  {row['ai_category']}: {row['count']:,}")
        
    # Unpersist original df
    df.unpersist()
        
    return df_balanced


def create_ml_pipeline(df: DataFrame, max_tfidf_features: int = 1000) -> Tuple[Pipeline, List[str]]:
    """
    Create pipeline with TF-IDF and Scaling.
    """
    logger.info(f"Creating ML pipeline (TF-IDF features: {max_tfidf_features})...")
    
    stages = []
    
    # 1. Text Features (TF-IDF)
    tokenizer = Tokenizer(inputCol="processed_text", outputCol="words")
    hashing_tf = HashingTF(inputCol="words", outputCol="tf_features", numFeatures=max_tfidf_features)
    idf = IDF(inputCol="tf_features", outputCol="text_features")
    stages.extend([tokenizer, hashing_tf, idf])
    
    # 2. Numerical Features Assembly
    num_cols = ["hour_of_day", "text_length", "word_count", "has_url", "score", "num_comments"]
    assembler_inputs = [c for c in num_cols if c in df.columns] + ["text_features"]
    
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")
    stages.append(assembler)
    
    # 3. Scaling
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
    stages.append(scaler)
    
    # 4. Label Indexing
    indexer = StringIndexer(inputCol="ai_category", outputCol="label", handleInvalid="skip")
    stages.append(indexer)
    
    return Pipeline(stages=stages), assembler_inputs


def train_models(pipeline_model, df_train):
    """
    Train 4 different models.
    """
    logger.info("Training models...")
    
    # Transform data once
    train_data = pipeline_model.transform(df_train).select("scaled_features", "label")
    train_data.cache()
    
    models = {}
    
    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="label", maxIter=50)
    models['LogisticRegression'] = lr.fit(train_data)
    
    # 2. Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label", 
                                numTrees=50, maxDepth=10, seed=42)
    models['RandomForest'] = rf.fit(train_data)
    
    # 3. SVM (OneVsRest)
    # Fixed: Using OneVsRest to handle multi-class labels (0,1,2,3)
    logger.info("Training LinearSVC (OneVsRest)...")
    lsvc = LinearSVC(featuresCol="scaled_features", labelCol="label", maxIter=10)
    ovr = OneVsRest(classifier=lsvc, featuresCol="scaled_features", labelCol="label")
    models['SVM_OneVsRest'] = ovr.fit(train_data)
    
    # 4. Neural Network (MLP)
    logger.info("Training Neural Network (MLP)...")
    # Get input size from first row
    input_dim = len(train_data.first()["scaled_features"])
    num_classes = train_data.select("label").distinct().count()
    
    # Layers: Input -> 128 -> 64 -> Output (Deep Network)
    layers = [input_dim, 128, 64, num_classes]
    mlp = MultilayerPerceptronClassifier(layers=layers, featuresCol="scaled_features", 
                                         labelCol="label", maxIter=100, seed=42)
    models['NeuralNetwork'] = mlp.fit(train_data)
    
    train_data.unpersist()
    return models


def evaluate_models(models, pipeline_model, df_test) -> pd.DataFrame:
    """
    Evaluate models and return metrics dataframe.
    """
    logger.info("Evaluating models...")
    
    test_data = pipeline_model.transform(df_test)
    test_data.cache()
    
    results = []
    
    evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1")
    evaluator_prec = MulticlassClassificationEvaluator(metricName="weightedPrecision")
    evaluator_rec = MulticlassClassificationEvaluator(metricName="weightedRecall")
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        predictions = model.transform(test_data)
        
        metrics = {
            "Model": name,
            "Accuracy": evaluator_acc.evaluate(predictions),
            "F1 Score": evaluator_f1.evaluate(predictions),
            "Precision": evaluator_prec.evaluate(predictions),
            "Recall": evaluator_rec.evaluate(predictions)
        }
        results.append(metrics)
        logger.info(f"  {name} F1: {metrics['F1 Score']:.4f}")
    
    test_data.unpersist()
    return pd.DataFrame(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ML Classification - Optimized")
    
    # Essential arguments
    parser.add_argument("--net-id", type=str, required=True, help="Student Net ID")
    
    # Optional arguments (including those needed for compatibility)
    parser.add_argument("--output-dir", type=str, default=None, help="S3 Output Path")
    parser.add_argument("--master", type=str, default="local[*]", help="Spark Master URL")
    parser.add_argument("--plots-dir", type=str, default="./plots", help="Local directory for plots")
    
    # Parameters kept for script compatibility, even if not strictly used in this optimized version
    parser.add_argument("--local-mode", action="store_true", default=False)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--max-tfidf-features", type=int, default=1000)
    parser.add_argument("--use-cv", action="store_true", default=False)
    parser.add_argument("--fast-mode", action="store_true", default=True)

    args = parser.parse_args()
    
    # Initialize Spark using the passed master URL
    spark = create_spark_session(master_url=args.master, local_mode=False)
    
    try:
        # 1. Read Data
        df = read_data_from_local_or_s3(spark, net_id=args.net_id, data_type="submissions", local_mode=False)
        
        # 2. Engineer Features (Categorization)
        df = engineer_features_spark(df)
        
        # 3. Smart Balancing (Downsample Other, Upsample Creative)
        df_balanced = balance_dataset(df)
        
        # 4. Pipeline Setup
        pipeline, _ = create_ml_pipeline(df_balanced, max_tfidf_features=args.max_tfidf_features)
        
        # 5. Split Data
        train_df, test_df = df_balanced.randomSplit([0.8, 0.2], seed=42)
        
        # 6. Fit Pipeline on Training Data
        logger.info("Fitting feature pipeline...")
        pipeline_model = pipeline.fit(train_df)
        
        # 7. Train Models
        models = train_models(pipeline_model, train_df)
        
        # 8. Evaluate
        results_df = evaluate_models(models, pipeline_model, test_df)
        
        # 9. Save Results
        output_path = args.output_dir or f"s3a://{args.net_id}-dsan6000-datasets/project/ml_results/"
        logger.info(f"Saving results to {output_path}...")
        
        # Convert pandas DF to Spark DF for saving to S3
        spark_results = spark.createDataFrame(results_df)
        spark_results.write.mode("overwrite").csv(os.path.join(output_path, "evaluation_metrics"), header=True)
        
        logger.info("Analysis Complete!")
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        sys.exit(1)
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    main()