#!/usr/bin/env python3
"""
TEST VERSION — ML Regression (Small Sample, Without r/neoliberal, Multi-regParam)

This script runs three Lasso regression models on a small sample of AI-related
Reddit comments (excluding r/neoliberal) to compare how different levels of
L1 regularization affect feature selection and engagement prediction.

Workflow:
  1) Load Reddit AI-related comments from S3 (excluding r/neoliberal)
  2) Load precomputed LDA topicDistribution from:
       s3a://hk1505-dsan6000-datasets/project/nlp_without_neo/parquet/NLPQ1_Anna_doc_topic_dist_ver2.parquet/
  3) Take a small random sample (~3%) for fast experimentation
  4) Text processing:
       - RegexTokenizer (lowercased)
       - StopWordsRemover (Spark defaults + URL/HTML artifacts)
       - CountVectorizer → tf_features
       - IDF → tfidf_features
  5) Add LDA topics + metadata:
       - topicDistribution
       - day_of_week, hour (from created_utc)
       - text_length (character count)
  6) Run L1-Lasso Regression with three different regParam values:
       • regParam = 0.10  → baseline regularization
       • regParam = 0.05  → weaker regularization (more features kept)
       • regParam = 0.20  → stronger regularization (only strongest signals)

Artifacts saved locally (separate folders per run, tagged by regParam):

  Predictions (id, subreddit, score, prediction):
    • data/parquet/ml_test_without_neoliberal_reg005_predictions/
    • data/parquet/ml_test_without_neoliberal_reg010_predictions/
    • data/parquet/ml_test_without_neoliberal_reg020_predictions/

  Word coefficients (TF-IDF feature weights):
    • data/csv/ml_test_without_neoliberal_reg005_word_coefs/
    • data/csv/ml_test_without_neoliberal_reg010_word_coefs/
    • data/csv/ml_test_without_neoliberal_reg020_word_coefs/

  Topic coefficients (LDA topic weights):
    • data/csv/ml_test_without_neoliberal_reg005_topic_coefs/
    • data/csv/ml_test_without_neoliberal_reg010_topic_coefs/
    • data/csv/ml_test_without_neoliberal_reg020_topic_coefs/

  Metadata coefficients (day_of_week, hour, text_length):
    • data/csv/ml_test_without_neoliberal_reg005_meta_coefs/
    • data/csv/ml_test_without_neoliberal_reg010_meta_coefs/
    • data/csv/ml_test_without_neoliberal_reg020_meta_coefs/

Goal:
   Understand which words, topics, and posting-time features matter most
     for predicting engagement in AI-related subreddits, when r/neoliberal
     is excluded, and how these signals change as we adjust L1 strength.
"""


import time
import os

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

# Make sure local output dirs exist
os.makedirs("data/parquet", exist_ok=True)
os.makedirs("data/csv", exist_ok=True)

print("=" * 80)
print("TEST RUN: ML Regression (TF-IDF + LDA topics → score)")
print("=" * 80)

overall_start = time.time()

spark = (
    SparkSession.builder
    .appName("Anna-Reddit-ML-Regression-TEST")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider",
    )
    .getOrCreate()
)

# ------------------------------------------------------------------
# 1) Load comments + topicDistribution from S3
# ------------------------------------------------------------------
AI_CORE_SUBREDDITS = [
    "ChatGPT","OpenAI","GPT4","ClaudeAI","PerplexityAI",
    "StableDiffusion","MidJourney","Sora","AIArt",
    "GenerativeAI","ArtificialIntelligence","MachineLearning",
    "computerscience","datascience","programming",
    "Futurology","singularity",
    "LocalLLaMA","OpenAI_Dev",
]

print("\n[1/8] Loading comments from S3...")
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
print(f"   Rows after filter (full): {full_count}")

# Small random sample for quick test
SAMPLE_FRACTION = 0.03
comments_sample = comments_filt.sample(False, SAMPLE_FRACTION, seed=42)
sample_count = comments_sample.count()
print(f"   Rows in TEST sample ({SAMPLE_FRACTION*100:.1f}%): {sample_count}")

if sample_count == 0:
    print("ERROR: Sample is empty. Increase SAMPLE_FRACTION.")
    spark.stop()
    raise SystemExit

print("\n[2/8] Loading topicDistribution parquet from S3...")
doc_topics = spark.read.parquet(
    "s3a://hk1505-dsan6000-datasets/project/nlp_without_neo/parquet/NLPQ1_Anna_doc_topic_dist_ver2.parquet/"
)
print(f"   Topic rows (full): {doc_topics.count()}")

# ------------------------------------------------------------------
# 3) Join sample comments + topicDistribution
# ------------------------------------------------------------------
print("\n[3/8] Joining TEST sample with topicDistribution on 'id'...")
df = (
    comments_sample.alias("c")
    .join(
        doc_topics.select("id","topicDistribution").alias("t"),
        on="id",
        how="inner",
    )
)

joined_count = df.count()
print(f"   Rows after join (TEST): {joined_count}")

df = df.filter(col("score").isNotNull())
if df.count() == 0:
    print("ERROR: Joined TEST df is empty after filtering null scores.")
    spark.stop()
    raise SystemExit

# ------------------------------------------------------------------
# 4) Add metadata features
# ------------------------------------------------------------------
print("\n[4/8] Adding metadata features...")
df = df.withColumn("day_of_week", dayofweek(from_unixtime("created_utc")))
df = df.withColumn("hour",       hour(from_unixtime("created_utc")))
df = df.withColumn("text_length", length(col("text")))

# ------------------------------------------------------------------
# 5) Text → TF-IDF (TEST settings)
# ------------------------------------------------------------------
print("\n[5/8] Building TF-IDF pipeline (TEST settings)...")

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
    vocabSize=3000,   # smaller for test
    minDF=50,
)

idf = IDF(
    inputCol="tf_features",
    outputCol="tfidf_features",
)

# ------------------------------------------------------------------
# 6) Assemble features
# ------------------------------------------------------------------
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
# -------------------------------------------------------------
# 7) Train/test split (모든 run이 공유)
# -------------------------------------------------------------
print("\n[7/8] Creating train/test split (shared across regParam runs)...")
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# -------------------------------------------------------------
# 8) Multiple Lasso runs with different regParam + SAVE locally
# -------------------------------------------------------------
experiments = [
    (0.10, "reg010"),
    (0.05, "reg005"),
    (0.20, "reg020"),
]

for reg_value, tag in experiments:
    print("\n" + "=" * 80)
    print(f"FULL RUN: Lasso Regression with regParam={reg_value}  (tag={tag})")
    print("=" * 80)

    lr = LinearRegression(
        featuresCol="features",
        labelCol="score",
        predictionCol="prediction",
        elasticNetParam=1.0,  # L1
        regParam=reg_value,
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

    print("\nFitting FULL pipeline...")
    fit_start = time.time()
    model = pipeline.fit(train_df)
    print(f"   FULL training time (regParam={reg_value}): {time.time() - fit_start:.1f}s")

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    print("\nEvaluating FULL model on held-out set...")
    preds = model.transform(test_df)

    for metric in ["rmse","mae","r2"]:
        evaluator = RegressionEvaluator(
            labelCol="score",
            predictionCol="prediction",
            metricName=metric,
        )
        val = evaluator.evaluate(preds)
        print(f"   {metric.upper()} (FULL, {tag}): {val:.4f}")

    # ---------------------------------------------------------
    # Save predictions locally 
    # ---------------------------------------------------------
    pred_path = f"data/parquet/ml_full_without_neoliberal_{tag}_predictions/"
    print(f"\nSaving FULL predictions to {pred_path} ...")

    preds.select("id","subreddit","score","prediction") \
         .write.mode("overwrite") \
         .parquet(pred_path)

    # ---------------------------------------------------------
    # Extract & save coefficients (words / topics / meta)
    # ---------------------------------------------------------
    print("Extracting and saving FULL model coefficients...")

    stages = model.stages
    cv_model = stages[2]          # CountVectorizerModel
    lr_model = stages[5]          # LinearRegressionModel (Lasso)

    vocab = cv_model.vocabulary
    coeffs = lr_model.coefficients

    vocab_size  = len(vocab)
    topic_count = len(df.select("topicDistribution").first()["topicDistribution"])

    # Word coefficients
    word_weights = [(vocab[i], float(coeffs[i])) for i in range(vocab_size)]
    word_df = spark.createDataFrame(word_weights, ["word","coef"])
    word_out = f"data/csv/ml_full_without_neoliberal_{tag}_word_coefs/"
    word_df.coalesce(1).write.mode("overwrite").csv(
        word_out,
        header=True
    )

    # Topic coefficients
    topic_base = vocab_size
    topic_rows = [
        (int(i), float(coeffs[topic_base + i]))
        for i in range(topic_count)
    ]
    topic_df = spark.createDataFrame(topic_rows, ["topic_id","coef"])
    topic_out = f"data/csv/ml_full_without_neoliberal_{tag}_topic_coefs/"
    topic_df.coalesce(1).write.mode("overwrite").csv(
        topic_out,
        header=True
    )

    # Meta feature coefficients
    meta_base = vocab_size + topic_count
    meta_rows = [
        ("day_of_week", float(coeffs[meta_base + 0])),
        ("hour",        float(coeffs[meta_base + 1])),
        ("text_length", float(coeffs[meta_base + 2])),
    ]
    meta_df = spark.createDataFrame(meta_rows, ["feature_name","coef"])
    meta_out = f"data/csv/ml_full_without_neoliberal_{tag}_meta_coefs/"
    meta_df.coalesce(1).write.mode("overwrite").csv(
        meta_out,
        header=True
    )

print("\nTotal FULL runtime: {:.1f}s".format(time.time() - overall_start))
print("=" * 80)
print("ALL FULL ML Regression RUNS COMPLETE (saved to data/parquet + data/csv)")
print("=" * 80)
