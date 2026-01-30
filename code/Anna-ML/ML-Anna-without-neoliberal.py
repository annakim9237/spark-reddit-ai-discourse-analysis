#!/usr/bin/env python3
"""
FULL VERSION — ML Regression on full AI-related Reddit comments

- Reads comments + topicDistribution from S3
- Uses full filtered dataset (no sampling)
- Builds TF-IDF + LDA-topic + metadata features
- Trains Lasso regression to predict score
- Saves locally:
  * Predictions → data/parquet/ml_full_without_neoliberal_predictions/
  * Word coefficients → data/csv/ml_full_without_neoliberal_word_coefs/
  * Topic coefficients → data/csv/ml_full_without_neoliberal_topic_coefs/
  * Meta feature coefficients → data/csv/ml_full_without_neoliberal_meta_coefs/
"""

import time
import os

from pathlib import Path
import shutil


from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_unixtime, dayofweek, hour, length
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    VectorAssembler
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# make sure local dirs exist
os.makedirs("data/parquet", exist_ok=True)
os.makedirs("data/csv", exist_ok=True)

print("=" * 80)
print("FULL RUN: ML Regression (TF-IDF + LDA topics → score)")
print("=" * 80)

overall_start = time.time()

spark = (
    SparkSession.builder
    .appName("Anna-Reddit-ML-Regression-FULL")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider",
    )
    .getOrCreate()
)

AI_CORE_SUBREDDITS = [
    "ChatGPT","OpenAI","GPT4","ClaudeAI","PerplexityAI",
    "StableDiffusion","MidJourney","Sora","AIArt",
    "GenerativeAI","ArtificialIntelligence","MachineLearning",
    "computerscience","datascience","programming",
    "Futurology","singularity",
    "LocalLLaMA","OpenAI_Dev",
]

# -------------------------------------------------------------
# 1) Load comments from S3
# -------------------------------------------------------------
print("\n[1/8] Loading comments from S3 (FULL)...")
comments = spark.read.parquet(
    "s3a://hk1505-dsan6000-datasets/project/reddit/parquet/comments/"
)

comments_filt = (
    comments.filter(col("subreddit").isin(AI_CORE_SUBREDDITS))
            .select("id","subreddit","created_utc","score","body")
            .withColumnRenamed("body","text")
            .filter(col("text").isNotNull() & (length(col("text")) > 30))
)

full_count = comments_filt.count()
print(f"   Rows after filter (FULL): {full_count}")

if full_count == 0:
    print("ERROR: No rows after filter. Check subreddit list / path.")
    spark.stop()
    raise SystemExit

# -------------------------------------------------------------
# 2) Load topicDistribution from S3
# -------------------------------------------------------------
print("\n[2/8] Loading topicDistribution parquet from S3...")
doc_topics = spark.read.parquet(
    "s3a://hk1505-dsan6000-datasets/project/nlp_without_neo/parquet/NLPQ1_Anna_doc_topic_dist_ver2.parquet/"
)
print(f"   Topic rows (FULL): {doc_topics.count()}")

# -------------------------------------------------------------
# 3) Join FULL comments + topicDistribution
# -------------------------------------------------------------
print("\n[3/8] Joining FULL comments with topicDistribution on 'id'...")
df = (
    comments_filt.alias("c")
    .join(
        doc_topics.select("id","topicDistribution").alias("t"),
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

# -------------------------------------------------------------
# 4) Add metadata features
# -------------------------------------------------------------
print("\n[4/8] Adding metadata features...")
df = df.withColumn("day_of_week", dayofweek(from_unixtime("created_utc")))
df = df.withColumn("hour",       hour(from_unixtime("created_utc")))
df = df.withColumn("text_length", length(col("text")))

# -------------------------------------------------------------
# 5) Text → TF-IDF (FULL settings)
# -------------------------------------------------------------
print("\n[5/8] Building TF-IDF pipeline (FULL settings)...")

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
    "http","https","www","com","net","org","io",
    "jpg","jpeg","png","gif","html",
    "link","links","url","site","page",
    "video","videos","youtube","yt","preview","format",
    "amp","lt","gt",
]
remover = remover.setStopWords(default_stops + extra_stops)

cv = CountVectorizer(
    inputCol="words_clean",
    outputCol="tf_features",
    vocabSize=8000,   # larger than test (e.g., 8000–10000)
    minDF=50,
)

idf = IDF(
    inputCol="tf_features",
    outputCol="tfidf_features",
)

# -------------------------------------------------------------
# 6) Assemble features
# -------------------------------------------------------------
print("\n[6/8] Assembling feature vector (FULL)...")

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

# -------------------------------------------------------------
# 7) Lasso Regression (FULL)
# -------------------------------------------------------------
print("\n[7/8] Setting up Lasso Regression (FULL)...")

lr = LinearRegression(
    featuresCol="features",
    labelCol="score",
    predictionCol="prediction",
    elasticNetParam=1.0,  # L1
    regParam=0.1,
    maxIter=50,
)

pipeline = Pipeline(stages=[
    tokenizer,
    remover,
    cv,
    idf,
    assembler,
    lr,
])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print("\nFitting FULL pipeline...")
fit_start = time.time()
model = pipeline.fit(train_df)
print(f"   FULL training time: {time.time() - fit_start:.1f}s")

# -------------------------------------------------------------
# 8) Evaluation + SAVE locally
# -------------------------------------------------------------
print("\n[8/8] Evaluating FULL model on held-out set...")
preds = model.transform(test_df)

for metric in ["rmse","mae","r2"]:
    evaluator = RegressionEvaluator(
        labelCol="score",
        predictionCol="prediction",
        metricName=metric,
    )
    val = evaluator.evaluate(preds)
    print(f"   {metric.upper()} (FULL): {val:.4f}")

# ---------- Save predictions locally ----------
print("\nSaving FULL predictions to local data/parquet/ml_full_without_neoliberal_predictions/ ...")

preds.select("id","subreddit","score","prediction") \
     .write.mode("overwrite") \
     .parquet("data/parquet/ml_full_without_neoliberal_predictions/")

# ---------- Extract & save coefficients ----------
print("Extracting and saving FULL model coefficients...")

stages = model.stages
cv_model = stages[2]          # CountVectorizerModel
lr_model = stages[5]          # LinearRegressionModel (Lasso)

vocab = cv_model.vocabulary
coeffs = lr_model.coefficients

vocab_size  = len(vocab)
topic_count = len(df.select("topicDistribution").first()["topicDistribution"])


# ============================================================
# Helper: write Spark DataFrame as a single clean CSV file
#         instead of a folder with part-0000-*.csv
# ============================================================
def save_single_csv(df, tmp_dir, final_file):
    """
    df         : Spark DataFrame to export
    tmp_dir    : temporary directory where Spark writes part-*.csv
    final_file : final single CSV filepath (e.g., data/csv/xxx.csv)

    Steps:
      1) Write df to tmp_dir as a single-part CSV
      2) Find the part-*.csv inside tmp_dir
      3) Move/rename it to final_file
      4) Remove tmp_dir (including _SUCCESS)
    """
    # 1) Spark write
    df.coalesce(1).write.mode("overwrite").csv(tmp_dir, header=True)

    tmp_path = Path(tmp_dir)
    final_path = Path(final_file)

    # 2) Find the part file
    part_file = None
    for f in tmp_path.iterdir():
        if f.name.startswith("part-") and f.suffix == ".csv":
            part_file = f
            break
    if part_file is None:
        raise RuntimeError(f"No part CSV found in {tmp_dir}")

    # 3) Ensure parent directory exists and remove existing file if any
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        final_path.unlink()

    # Move part file to final filename
    shutil.move(str(part_file), str(final_path))

    # 4) Remove temporary directory
    shutil.rmtree(tmp_path)


# ===========================
# Save Word Coefficients CSV
# ===========================
word_rows = [(vocab[i], float(coeffs[i])) for i in range(vocab_size)]
word_df = spark.createDataFrame(word_rows, ["word", "coef"])

save_single_csv(
    df=word_df,
    tmp_dir="data/csv/ml_full_without_neoliberal_word_coefs_tmp",
    final_file="data/csv/ml_full_without_neoliberal_word_coefs.csv"
)


# ============================
# Save Topic Coefficients CSV
# ============================
topic_base = vocab_size
topic_rows = [
    (int(i), float(coeffs[topic_base + i]))
    for i in range(topic_count)
]
topic_df = spark.createDataFrame(topic_rows, ["topic_id", "coef"])

save_single_csv(
    df=topic_df,
    tmp_dir="data/csv/ml_full_without_neoliberal_topic_coefs_tmp",
    final_file="data/csv/ml_full_without_neoliberal_topic_coefs.csv"
)


# ===========================
# Save Metadata Coefficients
# ===========================
meta_base = vocab_size + topic_count
meta_rows = [
    ("day_of_week", float(coeffs[meta_base + 0])),
    ("hour",        float(coeffs[meta_base + 1])),
    ("text_length", float(coeffs[meta_base + 2])),
]
meta_df = spark.createDataFrame(meta_rows, ["feature_name", "coef"])

save_single_csv(
    df=meta_df,
    tmp_dir="data/csv/ml_full_without_neoliberal_meta_coefs_tmp",
    final_file="data/csv/ml_full_without_neoliberal_meta_coefs.csv"
)

print("\nTotal FULL runtime: {:.1f}s".format(time.time() - overall_start))
print("=" * 80)
print("FULL ML Regression COMPLETE (saved to data/parquet + data/csv)")
print("=" * 80)
