#!/usr/bin/env python3
"""
BINARY CLASSIFICATION VERSION — WITHOUT r/neoliberal

Task:
- Predict whether a Reddit comment will have "high" engagement (score >= 9)
  vs "normal/low" engagement (score < 9), for AI-related subreddits
  EXCLUDING r/neoliberal.

Features:
- TF-IDF word features
- topicDistribution (precomputed LDA topics)
- day_of_week
- hour
- text_length

Model:
- Logistic Regression (binary classification)

Outputs:
- Metrics printed: Accuracy, F1, ROC-AUC, PR-AUC
- Confusion matrix (printed + CSV)
- Label distribution (printed + CSV)
- Predictions:
    - Parquet: data/parquet/ml_binary_without_neoliberal_predictions/
    - CSV   : data/csv/ml_binary_without_neoliberal_predictions.csv
"""

import time
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_unixtime, dayofweek, hour, length, when
)

from pyspark.ml import Pipeline

from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    VectorAssembler,
    CountVectorizerModel,    
)
from pyspark.ml.classification import (
    LogisticRegression,
    LogisticRegressionModel,   
)

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.sql import functions as F
from pyspark.ml.linalg import VectorUDT
# --------------------------------------------------------------------
# Make sure local output dirs exist
# --------------------------------------------------------------------
os.makedirs("data/parquet", exist_ok=True)
os.makedirs("data/csv", exist_ok=True)

print("=" * 80)
print("BINARY RUN (WITHOUT r/neoliberal): Logistic Regression")
print("=" * 80)

overall_start = time.time()

spark = (
    SparkSession.builder
    .appName("Anna-Reddit-ML-Binary-LogReg-without-neoliberal")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider",
    )
    .getOrCreate()
)

# --------------------------------------------------------------------
# 1) Load comments (WITHOUT r/neoliberal)
# --------------------------------------------------------------------
AI_CORE_SUBREDDITS = [
    "ChatGPT", "OpenAI", "GPT4", "ClaudeAI", "PerplexityAI",
    "StableDiffusion", "MidJourney", "Sora", "AIArt",
    "GenerativeAI", "ArtificialIntelligence", "MachineLearning",
    "computerscience", "datascience", "programming",
    "Futurology", "singularity",
    "LocalLLaMA", "OpenAI_Dev",
]

print("\n[1/8] Loading comments from S3 (FULL, without r/neoliberal)...")
comments = spark.read.parquet(
    "s3a://hk1505-dsan6000-datasets/project/reddit/parquet/comments/"
)

comments_filt = (
    comments.filter(col("subreddit").isin(AI_CORE_SUBREDDITS))
            .select("id", "subreddit", "created_utc", "score", "body")
            .withColumnRenamed("body", "text")
            .filter(col("text").isNotNull() & (length(col("text")) > 30))
)

full_count = comments_filt.count()
print(f"   Rows after filter (FULL): {full_count}")

if full_count == 0:
    print("ERROR: No rows after filter. Check subreddit list / path.")
    spark.stop()
    raise SystemExit

# --------------------------------------------------------------------
# 2) Load topicDistribution WITHOUT neoliberal
# --------------------------------------------------------------------
print("\n[2/8] Loading topicDistribution parquet from S3 (without_neo)...")
doc_topics = spark.read.parquet(
    "s3a://hk1505-dsan6000-datasets/project/nlp_without_neo/parquet/NLPQ1_Anna_doc_topic_dist_ver2.parquet/"
)
print(f"   Topic rows (FULL): {doc_topics.count()}")

# --------------------------------------------------------------------
# 3) Join comments + topicDistribution on 'id'
# --------------------------------------------------------------------
print("\n[3/8] Joining FULL comments with topicDistribution on 'id'...")
df = (
    comments_filt.alias("c")
    .join(
        doc_topics.select("id", "topicDistribution").alias("t"),
        on="id",
        how="inner",
    )
)

joined_count = df.count()
print(f"   Rows after join (FULL): {joined_count}")

df = df.filter(col("score").isNotNull())
if df.count() == 0:
    print("ERROR: Joined FULL df is empty after filtering null scores.")
    spark.stop()
    raise SystemExit

# --------------------------------------------------------------------
# 4) Add metadata features + binary label
# --------------------------------------------------------------------
print("\n[4/8] Adding metadata features and binary label...")

df = df.withColumn("day_of_week", dayofweek(from_unixtime("created_utc")))
df = df.withColumn("hour",       hour(from_unixtime("created_utc")))
df = df.withColumn("text_length", length(col("text")))

# Binary label: high engagement = score >= 9
# df = df.withColumn(
#     "label_binary",
#     when(col("score") >= 9, 1.0).otherwise(0.0)
# )

# Binary label: high engagement = score >= 8
df = df.withColumn(
    "label_binary",
    when(col("score") >= 8, 1.0).otherwise(0.0)
)

print("   Label distribution (0 = low/normal, 1 = high):")
label_dist_df = df.groupBy("label_binary").count().orderBy("label_binary")
label_dist_df.show()

# --------------------------------------------------------------------
# 5) Text → TF-IDF
# --------------------------------------------------------------------
def extract_cv_and_lr(pipeline_model):
    """
    Given a fitted PipelineModel, extract the CountVectorizerModel
    and LogisticRegressionModel from its stages.
    """
    cv_model = None
    lr_model = None

    for stage in pipeline_model.stages:
        if isinstance(stage, CountVectorizerModel):
            cv_model = stage
        if isinstance(stage, LogisticRegressionModel):
            lr_model = stage

    if cv_model is None or lr_model is None:
        raise ValueError(
            "Could not find both CountVectorizerModel and LogisticRegressionModel in the pipeline."
        )

    return cv_model, lr_model


def save_word_coefs_from_pipeline(pipeline_model, out_csv_path):
    """
    Extract word-level coefficients from a fitted PipelineModel
    (with CountVectorizer + LogisticRegression) and save them as a CSV.

    The output CSV will contain:
        - term: the vocabulary token
        - coef: the logistic regression coefficient for that term
        - abs_coef: absolute value of the coefficient (for ranking)

    NOTE:
    In this pipeline, the feature vector is:
        [tfidf_features, topicDistribution, day_of_week, hour, text_length]
    We only use the first len(vocab) coefficients, which correspond to the
    TF-IDF word features.
    """
    # 1) Get CountVectorizer and LogisticRegression from the pipeline
    cv_model, lr_model = extract_cv_and_lr(pipeline_model)

    # 2) Extract vocabulary and coefficients
    vocab = cv_model.vocabulary                       # list of tokens
    all_coefs = lr_model.coefficients.toArray()       # shape = (num_features,)

    # Word-level coefficients correspond to the first len(vocab) entries
    word_coefs = all_coefs[: len(vocab)]

    if len(all_coefs) < len(vocab):
        print(
            f"Warning: coef vector is shorter ({len(all_coefs)}) than vocab size ({len(vocab)}). "
            "Check the feature assembler ordering."
        )

    # 3) Build a pandas DataFrame
    size = min(len(vocab), len(word_coefs))
    data = pd.DataFrame({
        "term": vocab[:size],
        "coef": word_coefs[:size],
    })
    data["abs_coef"] = np.abs(data["coef"])

    # 4) Sort by absolute coefficient (largest impact first)
    data = data.sort_values("abs_coef", ascending=False)

    # 5) Ensure directory exists
    out_dir = os.path.dirname(out_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 6) Save to CSV
    data.to_csv(out_csv_path, index=False)

    print(f"Saved word coefficients to: {out_csv_path}")


def save_single_csv(df, tmp_dir, final_file):
    """
    Save a Spark DataFrame as a single CSV file.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to save.
    tmp_dir : str
        Temporary directory where Spark will write its partitioned CSV.
        (This directory will contain part-*.csv and _SUCCESS.)
    final_file : str
        Final CSV file path (single file) to move the part file to.

        Example:
            tmp_dir    = "data/csv/ml_binary_with_neoliberal_predictions_tmp"
            final_file = "data/csv/ml_binary_with_neoliberal_predictions.csv"
    """
    # Ensure directories exist
    os.makedirs(tmp_dir, exist_ok=True)
    final_dir = os.path.dirname(final_file)
    if final_dir:
        os.makedirs(final_dir, exist_ok=True)

    # 1) Cast all vector-type columns to string so that CSV writer can handle them
    vector_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, VectorUDT)
    ]

    for col_name in vector_cols:
        df = df.withColumn(col_name, F.col(col_name).cast("string"))

    # 2) Write to a temporary folder as a single-part CSV
    (
        df
        .coalesce(1)  # force a single CSV part file
        .write
        .mode("overwrite")
        .csv(tmp_dir, header=True)
    )

    # 3) Find the generated CSV part file inside the temporary folder
    part_file = None
    for fname in os.listdir(tmp_dir):
        if fname.endswith(".csv"):
            part_file = fname
            break

    if part_file is None:
        raise FileNotFoundError(f"No CSV part file found in {tmp_dir}")

    # 4) Move the part file to the final location with the desired name
    src = os.path.join(tmp_dir, part_file)
    os.replace(src, final_file)

    # clean up the temporary folder completely
    shutil.rmtree(tmp_dir)

    print(f"Saved single CSV to: {final_file}")


print("\n[5/8] Building TF-IDF feature pipeline...")

tokenizer = RegexTokenizer(
    inputCol="text",
    outputCol="words",
    pattern="\\W+",
    toLowercase=True,
)

remover = StopWordsRemover(
    inputCol="words",
    outputCol="words_clean",
)
default_stops = remover.getStopWords()

extra_stops = [
    "http", "https", "www", "com", "net", "org", "io",
    "jpg", "jpeg", "png", "gif", "html",
    "link", "links", "url", "site", "page",
    "video", "videos", "youtube", "yt", "preview", "format",
    "amp", "lt", "gt",
]
remover = remover.setStopWords(default_stops + extra_stops)

cv = CountVectorizer(
    inputCol="words_clean",
    outputCol="tf_features",
    vocabSize=8000,
    minDF=50,
)

idf = IDF(
    inputCol="tf_features",
    outputCol="tfidf_features",
)

# --------------------------------------------------------------------
# 6) Assemble features
# --------------------------------------------------------------------
print("\n[6/8] Assembling feature vector...")

feature_cols = [
    "tfidf_features",
    "topicDistribution",
    "day_of_week",
    "hour",
    "text_length",
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
)

# --------------------------------------------------------------------
# 7) Logistic Regression (binary)
# --------------------------------------------------------------------
print("\n[7/8] Setting up Logistic Regression classifier...")

lr_clf = LogisticRegression(
    featuresCol="features",
    labelCol="label_binary",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    maxIter=50,
    regParam=0.1,
    elasticNetParam=0.0,   # 0.0 = L2 regularization
)

pipeline = Pipeline(stages=[
    tokenizer,
    remover,
    cv,
    idf,
    assembler,
    lr_clf,
])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print("\nFitting FULL binary pipeline (without neoliberal)...")
fit_start = time.time()
model = pipeline.fit(train_df)
print(f"   Training time: {time.time() - fit_start:.1f}s")

# Save word-level coefficients for the WITHOUT-neoliberal model
save_word_coefs_from_pipeline(
    pipeline_model=model,
    out_csv_path="data/csv/ml_full_without_neoliberal_word_coefs.csv",
)


# --------------------------------------------------------------------
# 8) Evaluation + SAVE locally (parquet + CSV)
# --------------------------------------------------------------------
print("\n[8/8] Evaluating binary classifier on held-out set...")

preds = model.transform(test_df)
preds.cache()

# =========================
# Binary classification metrics
# =========================
bin_eval_roc = BinaryClassificationEvaluator(
    labelCol="label_binary",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC",
)
bin_eval_pr = BinaryClassificationEvaluator(
    labelCol="label_binary",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR",
)

auc_roc = bin_eval_roc.evaluate(preds)
auc_pr  = bin_eval_pr.evaluate(preds)

acc_eval = MulticlassClassificationEvaluator(
    labelCol="label_binary",
    predictionCol="prediction",
    metricName="accuracy",
)
f1_eval = MulticlassClassificationEvaluator(
    labelCol="label_binary",
    predictionCol="prediction",
    metricName="f1",
)

accuracy = acc_eval.evaluate(preds)
f1_score = f1_eval.evaluate(preds)

print(f"   ROC-AUC : {auc_roc:.4f}")
print(f"   PR-AUC  : {auc_pr:.4f}")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   F1-score: {f1_score:.4f}")

# =========================
# Confusion matrix
# =========================
print("\nConfusion matrix (label_binary vs prediction):")
confusion_df = (
    preds.groupBy("label_binary", "prediction")
         .count()
         .orderBy("label_binary", "prediction")
)
confusion_df.show()

# =========================
# Save metrics as CSV
# =========================
metrics_rows = [
    ("roc_auc", float(auc_roc)),
    ("pr_auc", float(auc_pr)),
    ("accuracy", float(accuracy)),
    ("f1", float(f1_score)),
]
metrics_df = spark.createDataFrame(metrics_rows, ["metric", "value"])
save_single_csv(
    df=metrics_df,
    tmp_dir="data/csv/ml_binary_without_neoliberal_metrics_tmp",
    final_file="data/csv/ml_binary_without_neoliberal_metrics.csv",
)

# =========================
# Save label distribution as CSV
# =========================
save_single_csv(
    df=label_dist_df,
    tmp_dir="data/csv/ml_binary_without_neoliberal_label_dist_tmp",
    final_file="data/csv/ml_binary_without_neoliberal_label_dist.csv",
)

# =========================
# Save confusion matrix as CSV
# =========================
save_single_csv(
    df=confusion_df,
    tmp_dir="data/csv/ml_binary_without_neoliberal_confusion_matrix_tmp",
    final_file="data/csv/ml_binary_without_neoliberal_confusion_matrix.csv",
)

# =========================
# Save predictions (parquet + CSV)
# =========================
pred_path_parquet = "data/parquet/ml_binary_without_neoliberal_predictions/"
print(f"\nSaving predictions to {pred_path_parquet} (parquet) ...")

preds_out = preds.select(
    "id",
    "subreddit",
    "score",
    "label_binary",
    "prediction",
    "probability",
)

preds_out.write.mode("overwrite").parquet(pred_path_parquet)

save_single_csv(
    df=preds_out,
    tmp_dir="data/csv/ml_binary_without_neoliberal_predictions_tmp",
    final_file="data/csv/ml_binary_without_neoliberal_predictions.csv",
)

print("\nSaved CSV files:")
print("  data/csv/ml_binary_without_neoliberal_metrics.csv")
print("  data/csv/ml_binary_without_neoliberal_label_dist.csv")
print("  data/csv/ml_binary_without_neoliberal_confusion_matrix.csv")
print("  data/csv/ml_binary_without_neoliberal_predictions.csv")

print("\nTotal runtime: {:.1f}s".format(time.time() - overall_start))
print("=" * 80)
print("BINARY Logistic Regression (WITHOUT r/neoliberal) COMPLETE")
print("=" * 80)

spark.stop()



