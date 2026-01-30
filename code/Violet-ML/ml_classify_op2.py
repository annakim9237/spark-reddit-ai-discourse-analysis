#!/usr/bin/env python3
"""
ML Classification - Final Version with Visualization Data
Updates:
1. Replaced HashingTF with CountVectorizer to extract Feature Importance (Keywords).
2. Saves Test Predictions for Confusion Matrix.
3. Tracks Hyperparameters.
4. Performs Cross-Validation.
"""

import logging
import os
import sys
import json
import numpy as np
from typing import Dict, List, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, when, length, regexp_replace, hour, 
    from_unixtime, udf, lit, concat_ws, split, size, coalesce, count
)
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, Tokenizer, 
    CountVectorizer, IDF, StringIndexer, IndexToString
)
from pyspark.ml.classification import (
    RandomForestClassifier, LogisticRegression, 
    LinearSVC, MultilayerPerceptronClassifier, OneVsRest
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_spark_session(master_url: str = "local[*]", app_name: str = "ML_Final_Vis") -> SparkSession:
    logger.info(f"Creating Spark session (master: {master_url})...")
    builder = SparkSession.builder.appName(app_name).master(master_url)
    
    builder = (builder
        .config("spark.executor.memory", "6g")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")
    )
    return builder.getOrCreate()

def read_data(spark: SparkSession, net_id: str) -> DataFrame:
    paths = [
        f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/submissions/",
        f"s3a://{net_id}-dsan6000-datasets/reddit/parquet/submissions/"
    ]
    for path in paths:
        try:
            logger.info(f"Attempting read: {path}")
            df = spark.read.parquet(path)
            if df.limit(1).count() > 0:
                logger.info(f"Loaded {df.count():,} rows.")
                return df
        except: continue
    raise RuntimeError("Could not load data.")

def categorize_subreddit_udf(subreddit: str) -> str:
    if not subreddit: return "Other"
    sub = str(subreddit).lower().strip()
    if sub in ['chatgpt', 'openai', 'claudeai', 'gpt4', 'localllama']: return 'Text AI'
    elif sub in ['stablediffusion']: return 'Creative AI'
    elif sub in ['machinelearning', 'datascience', 'computerscience', 'programming', 'singularity', 'futurology']: return 'Research/Tech'
    else: return 'Other'

def engineer_features(df: DataFrame) -> DataFrame:
    logger.info("Engineering features...")
    cat_func = udf(categorize_subreddit_udf, StringType())
    df = df.withColumn("ai_category", cat_func(col("subreddit")))
    
    df = df.withColumn("combined_text", concat_ws(" ", coalesce(col("title"), lit("")), coalesce(col("selftext"), lit(""))))
    df = df.withColumn("processed_text", regexp_replace(col("combined_text"), r"http\S+|www\S+|\[.*?\]", ""))
    
    df = df.withColumn("text_length", length(col("processed_text")))
    df = df.withColumn("word_count", size(split(col("processed_text"), "\\s+")))
    df = df.withColumn("hour_of_day", hour(from_unixtime(col("created_utc"))))
    df = df.withColumn("has_url", when(col("url").isNotNull(), 1).otherwise(0))
    df = df.fillna(0, subset=["score", "num_comments"])
    
    return df

def balance_dataset(df: DataFrame) -> DataFrame:
    logger.info("Balancing dataset...")
    df.cache()
    counts = df.groupBy("ai_category").count().collect()
    count_map = {row['ai_category']: row['count'] for row in counts}
    
    dfs = {cat: df.filter(col("ai_category") == cat) for cat in count_map.keys()}
    
    if "Other" in count_map:
        frac = min(1.0, 25000 / count_map["Other"])
        dfs["Other"] = dfs["Other"].sample(False, frac, seed=42)
        
    if "Creative AI" in count_map:
        frac = max(1.0, 12000 / max(1, count_map["Creative AI"]))
        dfs["Creative AI"] = dfs["Creative AI"].sample(True, frac, seed=42)
    
    if not dfs: return df
    
    balanced_df = dfs[list(dfs.keys())[0]]
    for cat in list(dfs.keys())[1:]:
        balanced_df = balanced_df.union(dfs[cat])
        
    df.unpersist()
    return balanced_df

def create_pipeline_stages(df: DataFrame, max_tfidf: int = 1000) -> Tuple[List, List]:
    stages = []
    
    # NLP Stages - Changed HashingTF to CountVectorizer for Feature Importance
    tokenizer = Tokenizer(inputCol="processed_text", outputCol="words")
    # VocabSize matches max_tfidf
    cv = CountVectorizer(inputCol="words", outputCol="tf_features", vocabSize=max_tfidf, minDF=5.0)
    idf = IDF(inputCol="tf_features", outputCol="text_features")
    stages.extend([tokenizer, cv, idf])
    
    # Vector Assembly
    num_cols = ["hour_of_day", "text_length", "word_count", "has_url", "score", "num_comments"]
    assembler_inputs = [c for c in num_cols if c in df.columns] + ["text_features"]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")
    stages.append(assembler)
    
    # Scaling
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
    stages.append(scaler)
    
    # Label Indexing
    indexer = StringIndexer(inputCol="ai_category", outputCol="label", handleInvalid="skip")
    stages.append(indexer)
    
    return stages, assembler_inputs, num_cols

def extract_feature_importance(feature_model, trained_models, num_cols):
    """
    Extract feature importance (Coefficients/Importances) mapping back to words.
    """
    logger.info("Extracting Feature Importance...")
    try:
        # 1. Get Vocabulary from CountVectorizer (Stage 1)
        # Pipeline: Tokenizer(0) -> CountVectorizer(1) -> IDF(2) -> Assembler(3) ...
        cv_model = feature_model.stages[1] 
        vocab = cv_model.vocabulary
        
        # 2. Combine feature names: Numerical + Vocab
        # Note: VectorAssembler order is [num_cols, text_features]
        all_features = num_cols + vocab
        
        importance_data = []

        # Extract from Random Forest (Easiest)
        if 'RandomForest' in trained_models:
            rf_model = trained_models['RandomForest']
            importances = rf_model.featureImportances.toArray()
            
            # Create a list of (feature, importance)
            feat_imp = [(all_features[i], float(importances[i])) for i in range(len(importances)) if i < len(all_features)]
            # Sort by importance
            feat_imp.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top 20
            for feat, score in feat_imp[:20]:
                importance_data.append({"Model": "RandomForest", "Feature": feat, "Importance": score})

        # Extract from Logistic Regression (Coefficients)
        if 'LogisticRegression' in trained_models:
            lr_model = trained_models['LogisticRegression']
            # LR for multiclass returns a matrix (numClasses x numFeatures)
            # We will take the max coefficient absolute value across all classes for simplicity
            coeffs = lr_model.coefficientMatrix
            # Sum of absolute coefficients across classes to find overall important features
            importances = np.sum(np.abs(coeffs.toArray()), axis=0)
            
            feat_imp = [(all_features[i], float(importances[i])) for i in range(len(importances)) if i < len(all_features)]
            feat_imp.sort(key=lambda x: x[1], reverse=True)
            
            for feat, score in feat_imp[:20]:
                importance_data.append({"Model": "LogisticRegression", "Feature": feat, "Importance": score})

        return pd.DataFrame(importance_data)
        
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        return pd.DataFrame()

def train_and_tune_models(df_train: DataFrame, stages: List) -> Dict:
    logger.info("Setting up Pipeline and Cross-Validation...")
    
    feature_pipeline = Pipeline(stages=stages)
    feat_model = feature_pipeline.fit(df_train)
    train_data = feat_model.transform(df_train).select("scaled_features", "label")
    train_data.cache()
    
    trained_models = {}
    best_params_log = {}
    evaluator = MulticlassClassificationEvaluator(metricName="f1")

    # 1. Logistic Regression (Tuned)
    logger.info("Training LR...")
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="label")
    lr_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).build()
    cv_lr = CrossValidator(estimator=lr, estimatorParamMaps=lr_grid, evaluator=evaluator, numFolds=3)
    cv_lr_model = cv_lr.fit(train_data)
    trained_models['LogisticRegression'] = cv_lr_model.bestModel
    best_params_log['LogisticRegression'] = {"regParam": cv_lr_model.bestModel.getRegParam()}

    # 2. Random Forest (Tuned)
    logger.info("Training RF...")
    rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label", seed=42)
    rf_grid = ParamGridBuilder().addGrid(rf.numTrees, [20]).addGrid(rf.maxDepth, [5, 10]).build()
    cv_rf = CrossValidator(estimator=rf, estimatorParamMaps=rf_grid, evaluator=evaluator, numFolds=3)
    cv_rf_model = cv_rf.fit(train_data)
    trained_models['RandomForest'] = cv_rf_model.bestModel
    best_params_log['RandomForest'] = {"maxDepth": cv_rf_model.bestModel.getMaxDepth()}

    # 3. SVM
    logger.info("Training SVM...")
    lsvc = LinearSVC(featuresCol="scaled_features", labelCol="label", maxIter=10)
    trained_models['SVM_OneVsRest'] = OneVsRest(classifier=lsvc, featuresCol="scaled_features", labelCol="label").fit(train_data)

    # 4. MLP
    logger.info("Training MLP...")
    input_dim = len(train_data.first()["scaled_features"])
    num_classes = train_data.select("label").distinct().count()
    layers = [input_dim, 64, 32, num_classes]
    mlp = MultilayerPerceptronClassifier(layers=layers, featuresCol="scaled_features", labelCol="label", seed=42, maxIter=50)
    trained_models['NeuralNetwork'] = mlp.fit(train_data)
    
    train_data.unpersist()
    return trained_models, feat_model, best_params_log

def evaluate_and_track(models, feature_model, df_test, best_params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Evaluating models...")
    test_data = feature_model.transform(df_test)
    
    # Get Label Map (Index to String) to save readable predictions
    label_indexer = feature_model.stages[-1] # StringIndexer is last
    labels = label_indexer.labels
    
    results = []
    all_predictions_pd = pd.DataFrame()
    
    evaluators = {
        "Accuracy": MulticlassClassificationEvaluator(metricName="accuracy"),
        "F1": MulticlassClassificationEvaluator(metricName="f1"),
        "Precision": MulticlassClassificationEvaluator(metricName="weightedPrecision"),
        "Recall": MulticlassClassificationEvaluator(metricName="weightedRecall")
    }
    
    for name, model in models.items():
        logger.info(f"Predicting with {name}...")
        predictions = model.transform(test_data)
        
        # Save Metrics
        row = {"Model": name}
        for metric_name, evaluator in evaluators.items():
            row[metric_name] = evaluator.evaluate(predictions)
            
        if name in best_params: row["Best_Params"] = json.dumps(best_params[name])
        else: row["Best_Params"] = "Fixed"
        results.append(row)
        
        # Save Predictions for Confusion Matrix (Sample to save memory)
        # Select True Label and Prediction
        preds_subset = predictions.select("label", "prediction").sample(False, 0.5, seed=42).toPandas()
        preds_subset["Model"] = name
        # Map indices back to class names
        preds_subset["True_Label"] = preds_subset["label"].apply(lambda x: labels[int(x)])
        preds_subset["Predicted_Label"] = preds_subset["prediction"].apply(lambda x: labels[int(x)])
        
        all_predictions_pd = pd.concat([all_predictions_pd, preds_subset])
        
    return pd.DataFrame(results), all_predictions_pd

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--net-id", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--master", type=str, default="local[*]")
    # Compat
    parser.add_argument("--plots-dir", type=str, default=".")
    parser.add_argument("--local-mode", action="store_true")
    parser.add_argument("--data-dir", type=str)
    
    args = parser.parse_args()
    
    spark = create_spark_session(args.master)
    
    try:
        df = read_data(spark, args.net_id)
        df = engineer_features(df)
        df_balanced = balance_dataset(df)
        
        stages, _, num_cols = create_pipeline_stages(df_balanced, max_tfidf=1000)
        train_df, test_df = df_balanced.randomSplit([0.8, 0.2], seed=42)
        
        # Train
        models, feature_model, best_params = train_and_tune_models(train_df, stages)
        
        # Evaluate & Get Predictions
        metrics_df, predictions_df = evaluate_and_track(models, feature_model, test_df, best_params)
        
        # Extract Feature Importance
        importance_df = extract_feature_importance(feature_model, models, num_cols)
        
        # Save Outputs
        out_path = args.output_dir or f"s3a://{args.net_id}-dsan6000-datasets/project/ml_results/"
        logger.info(f"Saving results to {out_path}...")
        
        # 1. Metrics
        spark.createDataFrame(metrics_df).write.mode("overwrite").csv(os.path.join(out_path, "metrics"), header=True)
        # 2. Predictions (For Confusion Matrix)
        spark.createDataFrame(predictions_df).write.mode("overwrite").csv(os.path.join(out_path, "predictions"), header=True)
        # 3. Feature Importance (For Bar Charts)
        if not importance_df.empty:
            spark.createDataFrame(importance_df).write.mode("overwrite").csv(os.path.join(out_path, "feature_importance"), header=True)
        
        logger.info("Done! Download 'metrics', 'predictions', and 'feature_importance' folders to plot locally.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()