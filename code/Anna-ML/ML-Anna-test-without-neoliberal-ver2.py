#!/usr/bin/env python3
"""
FULL DATA–FRIENDLY VERSION — GBT Regression (Without TF-IDF)
Small sample or full dataset, Without r/neoliberal

Task: Score prediction regression (score_log)

Features:
  • Word2Vec sentence embedding (w2v_features)
  • LDA topicDistribution
  • One-hot encoded time (day_of_week, hour)
  • text_length_log
  • has_question, has_url, exclamation_count, uppercase_ratio

Model:
  • GBTRegressor (Gradient Boosted Trees) on assembled features.

Use USE_SAMPLE flag to switch between ~3% sample (for testing) and full data.
"""

import time
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    from_unixtime,
    dayofweek,
    hour,
    length,
    log1p,
    expm1,
    when,
    instr,
    regexp_replace,
)
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    Word2Vec,
    VectorAssembler,
    OneHotEncoder,
    StringIndexer,
)
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# ------------------------ helper: uppercase ratio ------------------------
def uppercase_ratio_fn(text: str) -> float:
    if text is None:
        return 0.0
    total_letters = 0
    upper_letters = 0
    for ch in text:
        if ch.isalpha():
            total_letters += 1
            if ch.isupper():
                upper_letters += 1
    if total_letters == 0:
        return 0.0
    return float(upper_letters) / float(total_letters)

uppercase_ratio_udf = udf(uppercase_ratio_fn, DoubleType())

# ------------------------ config ------------------------
USE_SAMPLE = True   # True: ~3% sample for quick testing; False: full data
SAMPLE_FRACTION = 0.03

MIN_SCORE = -20
MAX_SCORE = 200

os.makedirs("data/parquet", exist_ok=True)

print("=" * 80)
print("LIGHTWEIGHT FULL RUN: GBT Regression (W2V + topics + time + simple text features)")
print("=" * 80)

overall_start = time.time()

spark = (
    SparkSession.builder
    .appName("Anna-Reddit-ML-Regression-LIGHT-FULL")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider",
    )
    .getOrCreate()
)

# ------------------------ 1) load comments & topics ------------------------
AI_CORE_SUBREDDITS = [
    "ChatGPT", "OpenAI", "GPT4", "ClaudeAI", "PerplexityAI",
    "StableDiffusion", "MidJourney", "Sora", "AIArt",
    "GenerativeAI", "ArtificialIntelligence", "MachineLearning",
    "computerscience", "datascience", "programming",
    "Futurology", "singularity",
    "LocalLLaMA", "OpenAI_Dev",
]

print("\n[1/8] Loading comments from S3...")
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
print(f"   Rows after filter (full, AI core only): {full_count}")

print("\n[2/8] Loading topicDistribution parquet from S3...")
doc_topics = spark.read.parquet(
    "s3a://hk1505-dsan6000-datasets/project/nlp_without_neo/parquet/NLPQ1_Anna_doc_topic_dist_ver2.parquet/"
)
topic_full_count = doc_topics.count()
print(f"   Topic rows (full): {topic_full_count}")

# ------------------------ 3) join and (maybe) sample ------------------------
print("\n[3/8] Joining comments with topicDistribution on 'id'...")
df_full = (
    comments_filt.alias("c")
    .join(
        doc_topics.select("id", "topicDistribution").alias("t"),
        on="id",
        how="inner",
    )
)

joined_count = df_full.count()
print(f"   Rows after join (full): {joined_count}")

if joined_count == 0:
    print("ERROR: Joined df_full is empty.")
    spark.stop()
    raise SystemExit

if USE_SAMPLE:
    print(f"\n[3b/8] Taking TEST sample (~{SAMPLE_FRACTION*100:.1f}% of joined data)...")
    df = df_full.sample(False, SAMPLE_FRACTION, seed=42)
    sample_count = df.count()
    print(f"   Rows in TEST sample: {sample_count}")
    if sample_count == 0:
        print("ERROR: TEST sample is empty. Increase SAMPLE_FRACTION.")
        spark.stop()
        raise SystemExit
else:
    print("\n[3b/8] Using FULL joined data (no sampling).")
    df = df_full

df = df.filter(col("score").isNotNull())
if df.count() == 0:
    print("ERROR: df is empty after filtering null scores.")
    spark.stop()
    raise SystemExit

# ------------------------ 4) metadata + label + extra features ------------------------
print("\n[4/8] Adding metadata features, label transforms, and extra text features...")

df = df.withColumn("day_of_week", dayofweek(from_unixtime("created_utc")))
df = df.withColumn("hour",       hour(from_unixtime("created_utc")))
df = df.withColumn("text_length", length(col("text")))

df = df.withColumn(
    "score_clipped",
    when(col("score") > MAX_SCORE, MAX_SCORE)
    .when(col("score") < MIN_SCORE, MIN_SCORE)
    .otherwise(col("score")),
)

df = df.withColumn("score_shifted", col("score_clipped") - MIN_SCORE)
df = df.withColumn("score_log", log1p(col("score_shifted")))
df = df.withColumn("text_length_log", log1p(col("text_length")))

df = df.withColumn("has_question",
                   when(instr(col("text"), "?") > 0, 1.0).otherwise(0.0))

df = df.withColumn(
    "has_url",
    when((instr(col("text"), "http") > 0) | (instr(col("text"), "www") > 0), 1.0)
    .otherwise(0.0),
)

df = df.withColumn(
    "exclamation_count",
    (length(col("text")) - length(regexp_replace(col("text"), "!", ""))).cast("double"),
)

df = df.withColumn("uppercase_ratio", uppercase_ratio_udf(col("text")))

# ------------------------ 5) train/test split ------------------------
print("\n[5/8] Creating train/test split...")
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df = train_df.cache()
test_df = test_df.cache()
print(f"   Train size: {train_df.count()}, Test size: {test_df.count()}")

# ------------------------ 6) featurizer pipeline (NO TF-IDF) ------------------------
print("\n[6/8] Building featurizer pipeline (Word2Vec + one-hot time + assembler)...")

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

w2v = Word2Vec(
    inputCol="words_clean",
    outputCol="w2v_features",
    vectorSize=100,
    minCount=20,   # full 데이터면 20 정도로 두는게 적당
    maxIter=5,
    seed=42,
)

indexer_dow = StringIndexer(
    inputCol="day_of_week",
    outputCol="day_of_week_idx",
    handleInvalid="keep",
)
indexer_hour = StringIndexer(
    inputCol="hour",
    outputCol="hour_idx",
    handleInvalid="keep",
)

encoder = OneHotEncoder(
    inputCols=["day_of_week_idx", "hour_idx"],
    outputCols=["day_of_week_oh", "hour_oh"],
    dropLast=False,
)

feature_cols = [
    "w2v_features",
    "topicDistribution",
    "day_of_week_oh",
    "hour_oh",
    "text_length_log",
    "has_question",
    "has_url",
    "exclamation_count",
    "uppercase_ratio",
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
)

featurizer_pipeline = Pipeline(stages=[
    tokenizer,
    remover,
    w2v,
    indexer_dow,
    indexer_hour,
    encoder,
    assembler,
])

print("\n[7/8] Fitting featurizer pipeline on TRAIN...")
feat_start = time.time()
featurizer_model = featurizer_pipeline.fit(train_df)
print(f"   Featurizer training time: {time.time() - feat_start:.1f}s")

train_feat = featurizer_model.transform(train_df).cache()
test_feat  = featurizer_model.transform(test_df).cache()
print(f"   train_feat count: {train_feat.count()}, test_feat count: {test_feat.count()}")

# ------------------------ 7) GBTRegressor ------------------------
print("\n[8/8] Training GBTRegressor (light config)...")

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="score_log",
    predictionCol="prediction",
    maxDepth=5,     # full 데이터 생각해서 너무 깊지 않게
    maxIter=30,     # 50 → 30으로 줄임
    stepSize=0.1,
    seed=42,
)

gbt_start = time.time()
gbt_model = gbt.fit(train_feat)
print(f"   GBT training time: {time.time() - gbt_start:.1f}s")

# ------------------------ evaluation ------------------------
print("\nEvaluating model on held-out set...")

preds = gbt_model.transform(test_feat)
preds = preds.withColumn("score_pred_shifted", expm1(col("prediction")))
preds = preds.withColumn("prediction_score", col("score_pred_shifted") + MIN_SCORE)

for metric in ["rmse", "mae", "r2"]:
    evaluator = RegressionEvaluator(
        labelCol="score_clipped",
        predictionCol="prediction_score",
        metricName=metric,
    )
    val = evaluator.evaluate(preds)
    print(f"   {metric.upper()} (GBT, clipped score scale): {val:.4f}")

TAG = "gbt_light_full"
pred_path = f"data/parquet/ml_test_without_neoliberal_{TAG}_predictions/"
print(f"\nSaving predictions to {pred_path} ...")

preds.select(
    "id",
    "subreddit",
    "score",
    "score_clipped",
    "score_log",
    "prediction",
    "prediction_score",
).write.mode("overwrite").parquet(pred_path)

print("\nTotal runtime: {:.1f}s".format(time.time() - overall_start))
print("=" * 80)
print("LIGHTWEIGHT FULL RUN COMPLETE.")
print("=" * 80)

spark.stop()
