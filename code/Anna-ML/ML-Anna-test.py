#!/usr/bin/env python3
"""
TEST VERSION — ML Regression (Small Sample, Multi-regParam)

This script runs **three Lasso regression models** on a small Reddit dataset sample
to compare feature selection strength under different regularization levels.

Workflow:
  1) Load Reddit AI-related comments from S3
  2) Load LDA topicDistribution parquet from S3
  3) Random sample (≈3%) for fast experiment
  4) Vectorize text → TF-IDF (CountVectorizer + IDF)
  5) Add LDA topics + metadata (day/hour/text_length)
  6) Run L1-Lasso Regression with **three regParam values**:
        • regParam = 0.10 (baseline)
        • regParam = 0.05 (more features retained — less shrinkage)
        • regParam = 0.20 (stronger feature filtering — only top signals remain)

Artifacts saved locally (separate folders per run):
  • Predictions → data/parquet/ml_test_reg005_predictions/
                → data/parquet/ml_test_reg010_predictions/
                → data/parquet/ml_test_reg020_predictions/

  • Word Feature Coefficients → data/csv/ml_test_reg005_word_coefs/
                              → data/csv/ml_test_reg010_word_coefs/
                              → data/csv/ml_test_reg020_word_coefs/

  • Topic Coefficients → data/csv/ml_test_reg005_topic_coefs/
                       → data/csv/ml_test_reg010_topic_coefs/
                       → data/csv/ml_test_reg020_topic_coefs/

  • Metadata Coefficients → data/csv/ml_test_reg005_meta_coefs/
                          → data/csv/ml_test_reg010_meta_coefs/
                          → data/csv/ml_test_reg020_meta_coefs/

Outcome:
   Compare how L1 regularization strength changes which words,
     topics, and metadata matter most for predicting Reddit engagement.
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
    "Futurology","singularity","neoliberal",
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
    "s3a://hk1505-dsan6000-datasets/project/nlp/parquet/NLPQ1_Anna_doc_topic_dist_ver2.parquet/"
)
print(f"   Topic rows (full): {doc_topics.count()}")

# ------------------------------------------------------------------
# 2) Join sample comments + topicDistribution
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
# 3) Add metadata features
# ------------------------------------------------------------------
print("\n[4/8] Adding metadata features...")
df = df.withColumn("day_of_week", dayofweek(from_unixtime("created_utc")))
df = df.withColumn("hour",       hour(from_unixtime("created_utc")))
df = df.withColumn("text_length", length(col("text")))

# ------------------------------------------------------------------
# 4) Text → TF-IDF (TEST settings)
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
# 5) Assemble features
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

# ------------------------------------------------------------------
# 6) Train/test split 
# ------------------------------------------------------------------
print("\n[7/8] Creating train/test split (shared across regParam runs)...")
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# ------------------------------------------------------------------
# 7) Multiple Lasso runs with different regParam + SAVE (local)
# ------------------------------------------------------------------
experiments = [
    (0.10, "reg010"),
    (0.05, "reg005"),
    (0.20, "reg020"),
]

for reg_value, tag in experiments:
    print("\n" + "=" * 80)
    print(f"TEST RUN: Lasso Regression with regParam={reg_value}  (tag={tag})")
    print("=" * 80)

    lr = LinearRegression(
        featuresCol="features",
        labelCol="score",
        predictionCol="prediction",
        elasticNetParam=1.0,  # L1
        regParam=reg_value,
        maxIter=30,
    )

    pipeline = Pipeline(stages=[
        tokenizer,
        remover,
        cv,
        idf,
        assembler,
        lr,
    ])

    print("\nFitting TEST pipeline on sample...")
    fit_start = time.time()
    model = pipeline.fit(train_df)
    print(f"   TEST training time (regParam={reg_value}): {time.time() - fit_start:.1f}s")

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------
    print("\nEvaluating TEST model on held-out sample...")
    preds = model.transform(test_df)

    for metric in ["rmse","mae","r2"]:
        evaluator = RegressionEvaluator(
            labelCol="score",
            predictionCol="prediction",
            metricName=metric,
        )
        val = evaluator.evaluate(preds)
        print(f"   {metric.upper()} (TEST, {tag}): {val:.4f}")

    # --------------------------------------------------------------
    # Save predictions → local parquet 
    # --------------------------------------------------------------
    pred_path = f"data/parquet/ml_test_{tag}_predictions/"
    print(f"\nSaving TEST predictions to {pred_path} ...")

    preds.select("id","subreddit","score","prediction") \
         .write.mode("overwrite") \
         .parquet(pred_path)

    # --------------------------------------------------------------
    # Extract & save coefficients (words / topics / meta)
    # --------------------------------------------------------------
    print("Extracting and saving TEST model coefficients...")

    stages = model.stages
    cv_model = stages[2]
    lr_model = stages[5]

    vocab = cv_model.vocabulary
    coeffs = lr_model.coefficients

    vocab_size  = len(vocab)
    topic_count = len(df.select("topicDistribution").first()["topicDistribution"])

    # Word coefficients
    word_weights = [(vocab[i], float(coeffs[i])) for i in range(vocab_size)]
    word_df = spark.createDataFrame(word_weights, ["word","coef"])
    word_out = f"data/csv/ml_test_{tag}_word_coefs/"
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
    topic_out = f"data/csv/ml_test_{tag}_topic_coefs/"
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
    meta_out = f"data/csv/ml_test_{tag}_meta_coefs/"
    meta_df.coalesce(1).write.mode("overwrite").csv(
        meta_out,
        header=True
    )

print("\nTotal TEST runtime: {:.1f}s".format(time.time() - overall_start))
print("=" * 80)
print("ALL TEST ML Regression RUNS COMPLETE (saved locally in data/parquet + data/csv)")
print("=" * 80)
