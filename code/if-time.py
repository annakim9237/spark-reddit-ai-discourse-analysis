"""
FINAL AUTHOR BEHAVIOR EDA — TECH/AI ECOSYSTEM ONLY (TOP 1%)

Outputs (to S3):
1. top1pct_posters_per_subreddit/
2. top1pct_global_tech_posters/
3. specialist_generalist_summary/
4. top_example_posts_per_subreddit/

Description:
Performs author-level EDA limited EXCLUSIVELY to tech/AI subreddits.
Includes:
- Top 1% posters per subreddit (one row per author, includes best post inside that subreddit)
- Top 1% global tech/AI posters (includes best tech post + best global tech post)
- Specialist vs generalist analysis (within tech ecosystem only)
- Example posts (top 5 per subreddit)

Author: Mandy Sun
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel
import sys


def main():

    if len(sys.argv) < 2:
        raise ValueError("Usage: spark-submit final_author_behavior_tech_only_top1pct.py <netid>")

    netid = sys.argv[1]

    spark = (
        SparkSession.builder
        .appName(f"FinalAuthorBehaviorEDA_TechOnly_Top1Pct_{netid}")
        .getOrCreate()
    )

    # ---------------------------------------------------------
    # TECH/AI ECOSYSTEM SUBREDDITS ONLY
    # ---------------------------------------------------------
    TECH_SUBS = [
        "ChatGPT", "ClaudeAI", "PerplexityAI", "OpenAI", "GPT4",
        "LocalLLaMA", "StableDiffusion",
        "MachineLearning", "datascience", "programming",
        "computerscience", "Futurology", "neoliberal", "singularity",
    ]

    # ---------------------------------------------------------
    # OUTPUT ROOT
    # ---------------------------------------------------------
    s3_output_root = "s3a://ms4821-dsan6000-datasets/project/author_eda_top1pct_tech_only"
    print(f"\nAll outputs will be written to:\n{s3_output_root}\n")

    # ---------------------------------------------------------
    # LOAD DATA, FILTER TO TECH ECOSYSTEM ONLY, REPARTITION
    # ---------------------------------------------------------
    comments_path = "s3a://ms4821-dsan6000-datasets/project/reddit/parquet/comments/"
    submissions_path = "s3a://ms4821-dsan6000-datasets/project/reddit/parquet/submissions/"

    comments = spark.read.parquet(comments_path).withColumn("data_type", F.lit("comment"))
    submissions = spark.read.parquet(submissions_path).withColumn("data_type", F.lit("submission"))

    df = (
        comments.unionByName(submissions, allowMissingColumns=True)
                .filter(F.col("subreddit").isin(TECH_SUBS))
                .repartition("subreddit")  # better for per-subreddit windows/aggregations
                .persist(StorageLevel.MEMORY_AND_DISK)
    )

    total_rows = df.count()
    print(f"Loaded {total_rows:,} rows (tech-only ecosystem).\n")

    # Minimal post table reused multiple times
    base_posts = (
        df.select("subreddit", "author", "title", "body", "score", "created_utc")
          .persist(StorageLevel.MEMORY_AND_DISK)
    )

    # ---------------------------------------------------------
    # GLOBAL (TECH-ONLY) AUTHOR STATS
    # ---------------------------------------------------------
    author_activity = (
        df.groupBy("author")
          .pivot("data_type")
          .count()
          .fillna(0)
          .withColumnRenamed("comment", "num_comments")
          .withColumnRenamed("submission", "num_submissions")
    )

    author_stats = (
        df.groupBy("author")
          .agg(
              F.count("*").alias("total_posts"),
              F.countDistinct("subreddit").alias("num_subreddits"),
              F.avg("score").alias("avg_score"),
              F.max("score").alias("max_score"),
              F.min("created_utc").alias("first_post"),
              F.max("created_utc").alias("last_post"),
          )
          .withColumn("active_span_seconds", F.col("last_post") - F.col("first_post"))
          .join(author_activity, "author", "left")
          .persist(StorageLevel.MEMORY_AND_DISK)
    )

    print(f"Total authors in tech ecosystem: {author_stats.count():,}")

    # ---------------------------------------------------------
    # GLOBAL BEST POST (TECH-ONLY)
    # ---------------------------------------------------------
    w_global_best = Window.partitionBy("author").orderBy(F.desc("score"))

    global_best_posts = (
        base_posts
        .withColumn("rn", F.row_number().over(w_global_best))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .withColumnRenamed("subreddit", "global_best_subreddit")
        .withColumnRenamed("title", "global_best_title")
        .withColumnRenamed("body", "global_best_body")
        .withColumnRenamed("score", "global_best_score")
        .withColumnRenamed("created_utc", "global_best_created_utc")
    )

    # ---------------------------------------------------------
    # 1. TOP 1% POSTERS PER SUBREDDIT (Option A)
    # ---------------------------------------------------------
    print("\n[1] Computing TOP 1% posters PER subreddit (tech-only, Option A)...")

    sub_activity = (
        df.groupBy("subreddit", "author")
          .agg(
              F.count("*").alias("total_posts"),
              F.avg("score").alias("avg_score"),
              F.max("score").alias("max_score"),
              F.min("created_utc").alias("first_post"),
              F.max("created_utc").alias("last_post"),
          )
          .withColumn("active_span_seconds", F.col("last_post") - F.col("first_post"))
    )

    sub_counts = (
        df.groupBy("subreddit", "author")
          .pivot("data_type")
          .count()
          .fillna(0)
          .withColumnRenamed("comment", "num_comments")
          .withColumnRenamed("submission", "num_submissions")
    )

    sub_activity = sub_activity.join(sub_counts, ["subreddit", "author"], "left")

    # Compute top 1% threshold per subreddit based on total_posts
    p99_posts_sub = (
        sub_activity.groupBy("subreddit")
                    .agg(F.expr("percentile_approx(total_posts, 0.99)").alias("p99_posts"))
    )

    top1_authors_sub = (
        sub_activity.join(p99_posts_sub, "subreddit")
                    .filter(F.col("total_posts") >= F.col("p99_posts"))
    )

    print("Rows of top 1% posters per subreddit:", top1_authors_sub.count())

    # Best post within that subreddit (highest score)
    w_sub_best = Window.partitionBy("subreddit", "author").orderBy(F.desc("score"))

    sub_best_posts = (
        base_posts
        .withColumn("rn", F.row_number().over(w_sub_best))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .withColumnRenamed("title", "sub_best_title")
        .withColumnRenamed("body", "sub_best_body")
        .withColumnRenamed("score", "sub_best_score")
        .withColumnRenamed("created_utc", "sub_best_created_utc")
    )

    top1pct_posters_per_sub = (
        top1_authors_sub.join(sub_best_posts, ["subreddit", "author"], "left")
    )

    top1pct_posters_per_sub.write.mode("overwrite").option("header", True).csv(
        f"{s3_output_root}/top1pct_posters_per_subreddit"
    )
    print("WROTE: top1pct_posters_per_subreddit")

    # ---------------------------------------------------------
    # 2. TOP 1% GLOBAL TECH/AI AUTHORS
    # ---------------------------------------------------------
    print("\n[2] Computing TOP 1% global tech authors...")

    p99_global = author_stats.agg(
        F.expr("percentile_approx(total_posts, 0.99)").alias("p99_posts")
    ).collect()[0]["p99_posts"]

    top1pct_global_tech = author_stats.filter(F.col("total_posts") >= p99_global)

    tech_author_subs = (
        df.groupBy("author")
          .agg(F.collect_set("subreddit").alias("tech_subreddits"))
          .withColumn("num_tech_subreddits", F.size("tech_subreddits"))
          .withColumn("tech_subreddits_str", F.concat_ws(", ", "tech_subreddits"))
          .drop("tech_subreddits")
    )

    # Best tech post per author (same universe as df/base_posts)
    w_tech_best = Window.partitionBy("author").orderBy(F.desc("score"))

    tech_best_posts = (
        base_posts
        .withColumn("rn", F.row_number().over(w_tech_best))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .withColumnRenamed("subreddit", "tech_best_subreddit")
        .withColumnRenamed("title", "tech_best_title")
        .withColumnRenamed("body", "tech_best_body")
        .withColumnRenamed("score", "tech_best_score")
        .withColumnRenamed("created_utc", "tech_best_created_utc")
    )

    top1pct_global_tech = (
        top1pct_global_tech
        .join(tech_author_subs, "author", "left")
        .join(tech_best_posts, "author", "left")
        .join(global_best_posts, "author", "left")
    )

    top1pct_global_tech.write.mode("overwrite").option("header", True).csv(
        f"{s3_output_root}/top1pct_global_tech_posters"
    )
    print("WROTE: top1pct_global_tech_posters")

    # ---------------------------------------------------------
    # 3. SPECIALIST VS GENERALIST (TECH-ONLY)
    # ---------------------------------------------------------
    print("\n[3] Computing specialist vs generalist (tech-only)...")

    author_types = (
        author_stats.withColumn(
            "author_category",
            F.when(F.col("num_subreddits") == 1, "specialist")
             .when(F.col("num_subreddits") <= 3, "broad")
             .otherwise("generalist")
        )
        .select("author", "author_category")
    )

    total_authors = author_types.count()

    spec_gen_summary = (
        author_types.groupBy("author_category")
                    .agg(F.count("*").alias("count"))
                    .withColumn(
                        "percent",
                        F.round((F.col("count") / F.lit(total_authors)) * 100, 2)
                    )
                    .orderBy("author_category")
    )

    spec_gen_summary.write.mode("overwrite").option("header", True).csv(
        f"{s3_output_root}/specialist_generalist_summary"
    )
    print("WROTE: specialist_generalist_summary")

    # ---------------------------------------------------------
    # 4. EXAMPLE POSTS PER SUBREDDIT (TOP 5 BY SCORE)
    # ---------------------------------------------------------
    print("\n[4] Extracting example posts per subreddit...")

    w_example = Window.partitionBy("subreddit").orderBy(F.desc("score"))

    example_posts = (
        base_posts
        .withColumn("rank", F.row_number().over(w_example))
        .filter(F.col("rank") <= 5)
        .drop("rank")
    )

    example_posts.write.mode("overwrite").option("header", True).csv(
        f"{s3_output_root}/top_example_posts_per_subreddit"
    )
    print("WROTE: top_example_posts_per_subreddit")

    print("\nFINAL TECH-ONLY AUTHOR EDA COMPLETED.\n")

    # Optional: free memory if this script ever grows more
    df.unpersist()
    base_posts.unpersist()
    author_stats.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()
