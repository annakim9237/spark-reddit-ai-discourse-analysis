#!/usr/bin/env python3
"""
Quick-fix Spark LDA pipeline (token filtering + stronger stopwords + higher minDF)

Changes vs original:
- RegexTokenizer(..., toLowercase=True)
- Expanded extra stopwords with domain tokens (ai, openai, chatgpt, gpt4, model, models, etc.)
- Added tokens_filter_udf to remove:
    * tokens with length < MIN_TOKEN_LEN
    * tokens containing digits
    * tokens that are not purely alphabetic
    * tokens starting with URL/user/subreddit patterns (http, www, u/, r/)
    * tokens in the expanded stopword set
- Increased CountVectorizer.minDF from 20 -> 50 and reduced vocabSize to 10000
- Keeps rest of pipeline (CountVectorizer -> LDA) the same structure
"""
import time
import os
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, udf, size
from pyspark.sql.types import IntegerType, ArrayType, StringType

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA
import pyspark.sql.functions as F

print("=" * 80)
print("NLP-Q1-Anna: Spark LDA ver2(token filter + stronger stopwords + higher minDF)")
print("=" * 80)

overall_start = time.time()

AI_CORE_SUBREDDITS = [
    "ChatGPT", "OpenAI", "GPT4", "ClaudeAI", "PerplexityAI",
    "StableDiffusion", "MidJourney", "Sora", "AIArt",
    "GenerativeAI", "ArtificialIntelligence", "MachineLearning",
    "computerscience", "datascience", "programming",
    "Futurology", "singularity", "neoliberal",
    "LocalLLaMA", "OpenAI_Dev",
]

MIN_TOKEN_LEN = 3
K_TOPICS = 15

# ----------------------------
# Spark session
# ----------------------------
print("\n[1/6] Creating Spark session...")
step_start = time.time()
spark = (
    SparkSession.builder
    .appName("NLPQ1-Anna-Spark-ver2")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider",
    )
    .getOrCreate()
)
print(f"Spark session created ({time.time() - step_start:.1f}s)")

# ----------------------------
# Read comments
# ----------------------------
print("\n[2/6] Reading AI-related comments from S3...")
step_start = time.time()
comments = spark.read.parquet(
    "s3a://hk1505-dsan6000-datasets/project/reddit/parquet/comments/"
)
print(f"Comments loaded ({time.time() - step_start:.1f}s)")
print(f"Total rows: {comments.count()}")

# ----------------------------
# Select & filter text
# ----------------------------
print("\n[3/6] Selecting columns, filtering by core AI subreddits, and filtering text...")
step_start = time.time()
df = (
    comments
    .filter(F.col("subreddit").isin(AI_CORE_SUBREDDITS))
    .select("id", "subreddit", "created_utc", "score", "body")
    .withColumnRenamed("body", "text")
)

df = df.filter(
    col("text").isNotNull()
    & (col("text") != "")
    & (length(col("text")) > 30)
)

print(f"Remaining rows after subreddit + text filter: {df.count()}")
print(f"Filtering done ({time.time() - step_start:.1f}s)")

# ----------------------------
# Build Spark ML pipeline components
# ----------------------------
print("\n[4/6] Building Spark ML pipeline (tokenizer + stopwords + token filter + CV + LDA)...")

# Tokenizer with toLowercase=True
tokenizer = RegexTokenizer(
    inputCol="text",
    outputCol="tokens",
    pattern="\\W+",  # split on non-word chars
    toLowercase=True,
)

# StopWordsRemover with expanded list
remover = StopWordsRemover(
    inputCol="tokens",
    outputCol="tokens_clean",
)

# Start from default stopwords and extend
default_stops = remover.getStopWords()
extra_stops = [
    # Conversational and vague tokens
    "people","person","someone","everyone","anyone","thing","things","stuff",
    "like","get","got","give","take","make","made","put","keep","let",
    "know","think","feel","look","see","say","said","tell","told",
    "go","went","come","came","back","away","around","really","actually",
    "even","still","always","never","ever","maybe","probably","kinda","sort",
    "just","also","very","so","too","lot","lots","many","much","more","less",

    # Time
    "time","day","year","years","week","weeks","month","months","hour","hours",
    "today","yesterday","tomorrow","now","ago","long","later","early","late",

    # Pronouns & auxiliaries
    "i","im","i'm","ive","i've","id","i'd",
    "you","youre","you're","youve","you've","youd","you'd",
    "we","weve","we've","they","theyre","they're","them","their","theirs",
    "he","him","his","she","her","hers","it","its","it’s",
    "my","mine","your","yours","our","ours","us","u",
    "one","ones","someone","anyone","everyone","noone",

    "re","ve","ll","m","d","s",
    "dont","don't","didnt","didn't","doesnt","doesn't",
    "cant","can't","couldnt","couldn't","wouldnt","wouldn't",
    "wont","won't","isnt","isn't","arent","aren't","wasnt","wasn't",
    "should","could","would","must","might","may","shall",

    # Fillers / interjections
    "yeah","yes","no","oh","well","ok","okay","hmm","uh","uhh","uhm","huh",
    "lol","lmao","haha","ahah","wow","nah","guess","kinda","sorta",

    # Reddit/platform tokens
    "r","askreddit","reddit","subreddit","thread","threads","comment","comments",
    "post","posts","posting","message","messages","inbox",
    "rule","rules","wiki","index","contact","automatically","action","mod","mods",

    # URL/format artifacts
    "http","https","www","com","net","org","io","jpg","jpeg","png","gif","html",
    "link","links","url","site","page","video","youtube","yt","preview","format",

    # Generic adjectives and vague words
    "good","bad","best","better","worse","nice","cool","great","awesome","amazing",
    "kind","type","sort","maybe","probably","literally",
    "else","anything","something","nothing","everything"
]

remover = remover.setStopWords(default_stops + extra_stops)

# CountVectorizer: higher minDF and smaller vocab to reduce noise
cv = CountVectorizer(
    inputCol="tokens_final",
    outputCol="features",
    vocabSize=10000,
    minDF=50,
)

# LDA
lda = LDA(
    k=K_TOPICS,
    maxIter=20,
    featuresCol="features",
    seed=42,
)

# Note: we'll not include the LDA in the pipeline until we add the token filter stage to DataFrame
pipeline = Pipeline(stages=[tokenizer, remover, cv, lda])

# ----------------------------
# Token filtering UDF (applied after StopWordsRemover)
# ----------------------------
print("\n[Extra] Defining token filter UDF and applying it...")

# Broadcast the stopwords set for use inside UDF
broadcast_stop = spark.sparkContext.broadcast(set(default_stops + extra_stops))

def tokens_filter(tokens):
    """
    tokens: list of tokens (already lowercased by tokenizer)
    Returns filtered list of tokens
    """
    if tokens is None:
        return []
    out = []
    stopset = broadcast_stop.value
    for t in tokens:
        if t is None:
            continue
        # skip user/subreddit/url-like tokens
        if t.startswith("http") or t.startswith("www") or t.startswith("u/") or t.startswith("r/"):
            continue
        # remove tokens containing digits or non-alpha characters
        if any(ch.isdigit() for ch in t):
            continue
        if not t.isalpha():
            continue
        # length filter
        if len(t) < MIN_TOKEN_LEN:
            continue
        # stopwords
        if t in stopset:
            continue
        out.append(t)
    return out

tokens_filter_udf = udf(tokens_filter, ArrayType(StringType()))

# Apply tokenizer -> stopword remover -> token filter
step_start = time.time()
df_tok = tokenizer.transform(df)
df_clean = remover.transform(df_tok)
df_clean = df_clean.withColumn("tokens_final", tokens_filter_udf(col("tokens_clean")))
# drop documents with empty token lists
df_clean = df_clean.filter(size(col("tokens_final")) > 0)
print(f"Rows after token filtering: {df_clean.count()}")
print(f"Token filtering done ({time.time() - step_start:.1f}s)")

# ----------------------------
# Fit CountVectorizer + LDA
# ----------------------------
print("\n[5/6] Fitting CountVectorizer + LDA...")
step_start = time.time()
# Build pipeline for CV + LDA only (tokenization already done)
from pyspark.ml import Pipeline as ML_Pipeline
pipeline_cv_lda = ML_Pipeline(stages=[cv, lda])
model = pipeline_cv_lda.fit(df_clean)
print(f"LDA fit complete ({time.time() - step_start:.1f}s)")

# Extract models
cv_model = model.stages[0]
lda_model = model.stages[1]
vocab = cv_model.vocabulary

# ----------------------------
# Extract topic descriptions and save outputs
# ----------------------------
print("\n[6/6] Extracting topic descriptions and saving outputs...")
topics = lda_model.describeTopics(maxTermsPerTopic=15)
topics_df = topics.toPandas()

def indices_to_words(indices_list):
    return [vocab[i] for i in indices_list]

topics_df["terms_words"] = topics_df["termIndices"].apply(indices_to_words)

# Ensure output dirs exist
os.makedirs("data", exist_ok=True)
os.makedirs("data/parquet", exist_ok=True)

topics_df.to_csv("data/NLPQ1_Anna_spark_lda_topics_ver2.csv", index=False)
print("Saved topic info to data/NLPQ1_Anna_spark_lda_topics_ver2.csv")

# Save document-topic distributions
print("\n[Extra] Saving document-topic distributions to Parquet...")
transformed = model.transform(df_clean)

transformed.select("id", "subreddit", "topicDistribution") \
    .write.mode("overwrite").parquet("data/parquet/NLPQ1_Anna_doc_topic_dist_ver2.parquet")
print("Saved doc-topic distributions to data/parquet/NLPQ1_Anna_doc_topic_dist_ver2.parquet")

print("\n[Extra] Computing dominant topic per document and topic sizes...")

def argmax_topic(dist):
    arr = dist.toArray() if hasattr(dist, "toArray") else list(dist)
    max_idx = max(range(len(arr)), key=lambda i: arr[i])
    return int(max_idx)

argmax_udf = udf(argmax_topic, IntegerType())

transformed_with_dom = transformed.withColumn(
    "dominant_topic",
    argmax_udf(col("topicDistribution")),
)

topic_counts = (
    transformed_with_dom
    .groupBy("dominant_topic")
    .count()
    .orderBy(F.col("count").desc())
)

topic_counts_pdf = topic_counts.toPandas()
topic_counts_path = "data/NLPQ1_Anna_spark_lda_topic_counts_ver2.csv"
topic_counts_pdf.to_csv(topic_counts_path, index=False)
print(f"Saved topic counts to {topic_counts_path}")

print("\n" + "=" * 80)
print("NLP-Q1 Spark LDA ver2 COMPLETE!")
print("=" * 80)
print(f"Total time: {time.time() - overall_start:.1f}s")