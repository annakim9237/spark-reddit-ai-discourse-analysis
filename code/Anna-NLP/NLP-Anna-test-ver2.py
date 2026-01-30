#!/usr/bin/env python3
"""
RQ0 (SMALL TEST VERSION, ver2): Dominant Topics in AI-related Reddit Comments (Spark LDA)

This is the SMALL test script updated with the "ver2" preprocessing:
- RegexTokenizer with toLowercase=True
- Expanded stopwords (including domain tokens)
- Token filtering UDF to remove short tokens, tokens with digits, non-alpha tokens,
  url/user/subreddit-like tokens, and tokens in the expanded stopword list
- CountVectorizer uses the filtered token column ("tokens_final")
- Keep SMALL settings: limited docs, small vocab, few topics for fast runs
"""
import time
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, udf, size
from pyspark.sql.types import IntegerType, ArrayType, StringType

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA
import pyspark.sql.functions as F

print("=" * 80)
print("NLP-Q1-Anna (SMALL ver2): Spark LDA Topic Modeling on AI-related Reddit Comments")
print("=" * 80)

overall_start = time.time()

# -------------------------------------------------------------------
# Core AI-related subreddits (AskReddit excluded)
# -------------------------------------------------------------------
AI_CORE_SUBREDDITS = [
    # AI model ecosystems
    "ChatGPT", "OpenAI", "GPT4", "ClaudeAI", "PerplexityAI",
    # image/video creative gen
    "StableDiffusion", "MidJourney", "Sora", "AIArt",
    # research / tech
    "GenerativeAI", "ArtificialIntelligence", "MachineLearning",
    "computerscience", "datascience", "programming",
    # social + policy + future
    "Futurology", "singularity", "neoliberal",
    # dev hardcore
    "LocalLLaMA", "OpenAI_Dev",
]

MIN_TOKEN_LEN = 3
K_TOPICS = 5  # SMALL
LIMIT_DOCS = 10000  # SMALL
CV_VOCAB_SIZE = 5000  # SMALL
CV_MIN_DF = 50  # SMALL

# 1. Spark Session
print("\n[1/6] Creating Spark session (SMALL, ver2)...")
step_start = time.time()
spark = (
    SparkSession.builder
    .appName("NLPQ1-Anna-SparkLDA-Comments-SMALL-ver2")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.InstanceProfileCredentialsProvider")
    .getOrCreate()
)
print(f"Spark session created ({time.time() - step_start:.1f}s)")

# 2. Read comments (already filtered for team 05 AI-related subreddits)
print("\n[2/6] Reading AI-related comments from S3...")
step_start = time.time()
comments = spark.read.parquet(
    "s3a://hk1505-dsan6000-datasets/project/reddit/parquet/comments/"
)
print(f"Comments loaded ({time.time() - step_start:.1f}s)")
print(f"Total rows (full): {comments.count()}")

# 3. Select & filter text (SMALL subset, core AI subreddits only)
print("\n[3/6] Selecting columns, filtering by core AI subreddits, and sampling (SMALL)...")
step_start = time.time()

df = (
    comments
    .filter(F.col("subreddit").isin(AI_CORE_SUBREDDITS))  # ⭐ core AI subreddits only
    .select("id", "subreddit", "created_utc", "score", "body")
    .withColumnRenamed("body", "text")
)

# Stricter length filter for test version
df = df.filter(
    col("text").isNotNull()
    & (col("text") != "")
    & (length(col("text")) > 50)   # SMALL: use longer comments only
)

# Take a subset to keep the run fast (e.g., 10k documents)
df = df.limit(LIMIT_DOCS)  # adjust up/down as needed

print(f"Remaining rows after subreddit filter + text filter + limit: {df.count()}")
print(f"Filtering done ({time.time() - step_start:.1f}s)")

# 4. Build Spark ML pipeline pieces: tokenizer -> remove stopwords -> token filter -> CV -> LDA
print("\n[4/6] Building Spark ML pipeline components (SMALL, ver2)...")

# Tokenizer with toLowercase=True
tokenizer = RegexTokenizer(
    inputCol="text",
    outputCol="tokens",
    pattern="\\W+",
    toLowercase=True,
)

remover = StopWordsRemover(
    inputCol="tokens",
    outputCol="tokens_clean",
)

# Expand stopwords (keep default + extras)
default_stops = remover.getStopWords()
extra_stops = [
    # 1) Very common conversational words (too general to define a topic)
    "people","person","someone","everyone","anyone","thing","things","stuff",
    "like","get","got","give","take","make","made","put","keep","let",
    "know","think","feel","look","see","say","said","tell","told",
    "go","went","come","came","back","away","around","really","actually",
    "even","still","always","never","ever","maybe","probably","kinda","sort",
    "just","also","very","so","too","lot","lots","many","much","more","less",

    # 2) Time-related words (appear everywhere, not informative)
    "time","day","year","years","week","weeks","month","months","hour","hours",
    "today","yesterday","tomorrow","now","ago","long","later","early","late",

    # 3) Pronouns (1st/2nd/3rd person), not useful for topics
    "i","im","i'm","ive","i've","id","i'd",
    "you","youre","you're","youve","you've","youd","you'd",
    "we","weve","we've","they","theyre","they're","them","their","theirs",
    "he","him","his","she","her","hers","it","its","it’s",
    "my","mine","your","yours","our","ours","us","u",
    "one","ones","someone","anyone","everyone","noone",

    # 4) Auxiliary verbs & fragmented tokens
    "re","ve","ll","m","d","s",
    "dont","don't","didnt","didn't","doesnt","doesn't",
    "cant","can't","couldnt","couldn't","wouldnt","wouldn't",
    "wont","won't","isnt","isn't","arent","aren't","wasnt","wasn't",
    "should","could","would","must","might","may","shall",

    # 5) Fillers, interjections, casual expressions
    "yeah","yes","no","oh","well","ok","okay","hmm","uh","uhh","uhm","huh",
    "lol","lmao","haha","ahah","wow","nah","guess","kinda","sorta",

    # 6) Reddit/platform-related terms (not semantic topics)
    "r","askreddit","reddit","subreddit","thread","threads","comment","comments",
    "post","posts","posting","message","messages","inbox",
    "rule","rules","wiki","index","contact","automatically","action","mod","mods",

    # 7) URL/formatting artifacts
    "http","https","www","com","net","org","io","jpg","jpeg","png","gif","html",
    "link","links","url","site","page","video","youtube","yt","preview","format",

    # 8) Misc vague-but-frequent adjectives and general words
    "good","bad","best","better","worse","nice","cool","great","awesome","amazing",
    "kind","type","sort","maybe","probably","literally",
    "else","anything","something","nothing","everything",

    # 9) Domain-specific noisy tokens (remove for quick test)
    "ai","openai","chatgpt","gpt4","gpt","model","models","llama","llms","huggingface",
    "hf","api","token","key","discord","bot","bots","prompt","prompts","dataset","datasets",
    "training","trained","fine","finetune","fine-tune","inference","inferences","results",
    "thanks","thank","thx","amp"
]

remover = remover.setStopWords(default_stops + extra_stops)

# CountVectorizer uses the filtered tokens column (tokens_final)
cv = CountVectorizer(
    inputCol="tokens_final",
    outputCol="features",
    vocabSize=CV_VOCAB_SIZE,
    minDF=CV_MIN_DF
)

# LDA (SMALL)
k_topics = K_TOPICS
lda = LDA(
    k=k_topics,
    maxIter=5,        # SMALL: fewer iterations
    featuresCol="features",
    seed=42
)

# We will apply tokenizer/remover + token filter first, then fit CV+LDA via a small pipeline
print("\n[Extra] Defining token filter UDF and applying it (SMALL, ver2)...")

# Broadcast the stopwords set for UDF usage
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

# Apply tokenizer -> stopword remover -> token filter (explicit transforms)
step_start = time.time()
df_tok = tokenizer.transform(df)
df_clean = remover.transform(df_tok)
df_clean = df_clean.withColumn("tokens_final", tokens_filter_udf(col("tokens_clean")))
# drop documents with empty token lists
df_clean = df_clean.filter(size(col("tokens_final")) > 0)
print(f"Rows after token filtering: {df_clean.count()}")
print(f"Token filtering done ({time.time() - step_start:.1f}s)")

# 5. Fit CountVectorizer + LDA (SMALL)
print("\n[5/6] Fitting CountVectorizer + LDA (SMALL, ver2)...")
step_start = time.time()
from pyspark.ml import Pipeline as ML_Pipeline
pipeline_cv_lda = ML_Pipeline(stages=[cv, lda])
model = pipeline_cv_lda.fit(df_clean)
print(f"LDA fit complete ({time.time() - step_start:.1f}s)")

# 6. Extract models and save outputs
cv_model = model.stages[0]
lda_model = model.stages[1]
vocab = cv_model.vocabulary

print("\n[6/6] Extracting topic descriptions and saving outputs (SMALL, ver2)...")
topics = lda_model.describeTopics(maxTermsPerTopic=15)
topics_df = topics.toPandas()

def indices_to_words(indices_list):
    return [vocab[i] for i in indices_list]

topics_df["terms_words"] = topics_df["termIndices"].apply(indices_to_words)

# Ensure output dirs exist
os.makedirs("data", exist_ok=True)
os.makedirs("data/parquet", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

# Use different file names for the small/test version
topics_csv_path = "data/NLPQ1_Anna_spark_lda_topics_small_ver2.csv"
topics_df.to_csv(topics_csv_path, index=False)
print(f"Saved SMALL (ver2) topic info to {topics_csv_path}")

print("\n[Extra] Saving SMALL document-topic distributions to Parquet...")
transformed = model.transform(df_clean)

doc_topic_path = "data/parquet/NLPQ1_Anna_doc_topic_dist_small_ver2.parquet"
transformed.select("id", "subreddit", "topicDistribution") \
    .write.mode("overwrite").parquet(doc_topic_path)
print(f"Saved SMALL doc-topic distributions to {doc_topic_path}")

print("\n[Extra] Computing dominant topic per document and topic sizes (SMALL)...")

def argmax_topic(dist):
    arr = dist.toArray() if hasattr(dist, "toArray") else list(dist)
    max_idx = max(range(len(arr)), key=lambda i: arr[i])
    return int(max_idx)

argmax_udf = udf(argmax_topic, IntegerType())

transformed_with_dom = transformed.withColumn(
    "dominant_topic",
    argmax_udf(col("topicDistribution"))
)

topic_counts = (
    transformed_with_dom
    .groupBy("dominant_topic")
    .count()
    .orderBy(F.col("count").desc())
)

topic_counts_pdf = topic_counts.toPandas()
topic_counts_path = "data/test/NLPQ1_Anna_spark_lda_topic_counts_small_ver2.csv"
topic_counts_pdf.to_csv(topic_counts_path, index=False)

print(f"Saved SMALL topic counts to {topic_counts_path}")

print("\n" + "=" * 80)
print("NLP-Q1 Spark LDA Topic Modeling (SMALL, ver2) COMPLETE!")
print("=" * 80)
print(f"Total time (SMALL, ver2): {time.time() - overall_start:.1f}s")