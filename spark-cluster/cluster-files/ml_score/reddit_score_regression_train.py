#!/usr/bin/env python3
"""
file: reddit_score_regression_train.py

Train a Reddit comment score regression model (Spark ML, cluster mode).

- Predicts continuous `score`
- Saves model + metrics + plots
- Optionally exports sklearn-friendly sample CSV
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.feature import (
    CountVectorizer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from reddit_nlp_cleaning import (
    build_spark,
    load_reddit_data,
    clean_reddit_text,
    build_preprocessing_pipeline,
)

# ================================================================
# Feature Engineering Helpers
# ================================================================
def add_ml_features(df):
    """Add: hour_of_day, day_of_week, is_weekend, body_length."""
    df = df.withColumn("created_ts", F.from_unixtime("created_utc").cast("timestamp"))
    df = df.withColumn("hour_of_day", F.hour("created_ts").cast("int"))
    df = df.withColumn("day_of_week", F.dayofweek("created_ts").cast("int"))
    df = df.withColumn(
        "is_weekend", F.col("day_of_week").isin([1, 7]).cast("int")
    )
    df = df.withColumn("body_length", F.length(F.col("body_cleaned")))
    return df


# ================================================================
# Preprocess Helper
# ================================================================
def preprocess_reddit(
    spark,
    s3_path,
    data_type,
    sample_fraction=None,
    remove_stopwords=True,
    apply_lemmatization=True,
):
    df = load_reddit_data(
        spark,
        s3_path=s3_path,
        data_type=data_type,
        sample_fraction=sample_fraction,
    )

    # Clean raw text
    df = clean_reddit_text(df, "body")

    # NLP pipeline
    pipeline = build_preprocessing_pipeline(
        input_col="body_cleaned",
        remove_stopwords=remove_stopwords,
        apply_lemmatization=apply_lemmatization,
    )
    model = pipeline.fit(df)
    df = model.transform(df)

    # Basic features
    df = df.withColumn("token_count", F.size(F.col("tokens")))
    df = df.withColumn("processed_text", F.array_join("tokens", " "))

    return df


# ================================================================
# ML Pipeline
# ================================================================
def build_regression_pipeline():
    text_cv = CountVectorizer(
        inputCol="tokens",
        outputCol="text_features",
        vocabSize=1 << 18,
        minDF=5,
    )

    sub_indexer = StringIndexer(
        inputCol="subreddit",
        outputCol="subreddit_index",
        handleInvalid="keep",
    )

    sub_ohe = OneHotEncoder(
        inputCols=["subreddit_index"],
        outputCols=["subreddit_ohe"],
    )

    numeric_cols = [
        "token_count",
        "controversiality",
        "gilded",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "body_length",
    ]

    assembler = VectorAssembler(
        inputCols=["text_features", "subreddit_ohe"] + numeric_cols,
        outputCol="features",
    )

    lr = LinearRegression(
        labelCol="score",
        featuresCol="features",
        maxIter=20,
        regParam=0.0,
    )

    return SparkPipeline(stages=[text_cv, sub_indexer, sub_ohe, assembler, lr])


# ================================================================
# Utility
# ================================================================
def ensure_required_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def plot_actual_vs_pred(y_true, y_pred, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.3)
    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], c="red")
    plt.title("Actual vs Predicted Comment Score")
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Reddit Score Regression (Cluster Mode)")

    parser.add_argument("master_url", nargs="?", default=None)
    parser.add_argument("--s3-input",
                        default="s3a://ea973-dsan6000-datasets-final/project/reddit/parquet/")
    parser.add_argument("--data-type", choices=["comments"], default="comments")
    parser.add_argument("--sample", type=float)

    parser.add_argument("--metrics-out", default="data/csv/reddit_score_regression_metrics.csv")
    parser.add_argument("--plots-dir",  default="data/plots")
    parser.add_argument("--model-dir",  default="code/ml/models/reddit_score_regressor")

    parser.add_argument("--sample-csv", type=str)
    parser.add_argument("--sample-size", type=int, default=50000)

    args = parser.parse_args()

    master_url = args.master_url or (
        f"spark://{os.getenv('MASTER_PRIVATE_IP')}:7077"
        if os.getenv("MASTER_PRIVATE_IP")
        else None
    )
    if not master_url:
        print("Error: no master URL provided")
        return 1

    spark = None
    try:
        spark = build_spark(master_url, app_name="Reddit_Score_Regression")

        # ---------------------------
        # Preprocess
        # ---------------------------
        df = preprocess_reddit(
            spark,
            args.s3_input,
            args.data_type,
            sample_fraction=args.sample,
        )

        # Add engineered ML features (FIXED)
        df = add_ml_features(df)

        ensure_required_cols(df, [
            "score", "tokens", "token_count", "controversiality", "gilded",
            "hour_of_day", "day_of_week", "is_weekend", "body_length"
        ])

        # ---------------------------
        # Train/test split
        # ---------------------------
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

        model = build_regression_pipeline().fit(train_df)
        pred_df = model.transform(test_df)

        # ---------------------------
        # Metrics
        # ---------------------------
        evaluator_rmse = RegressionEvaluator(
            metricName="rmse",
            labelCol="score",
            predictionCol="prediction",
        )
        evaluator_r2 = RegressionEvaluator(
            metricName="r2",
            labelCol="score",
            predictionCol="prediction",
        )

        rmse = evaluator_rmse.evaluate(pred_df)
        r2 = evaluator_r2.evaluate(pred_df)
        print(f"RMSE = {rmse:.4f}, R² = {r2:.4f}")

        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"rmse": rmse, "r2": r2}]).to_csv(args.metrics_out, index=False)

        # ---------------------------
        # Plot
        # ---------------------------
        Path(args.plots_dir).mkdir(parents=True, exist_ok=True)
        y_pred = np.array(pred_df.select("prediction").rdd.map(lambda r: r[0]).collect())
        y_true = np.array(pred_df.select("score").rdd.map(lambda r: r[0]).collect())
        plot_actual_vs_pred(
            y_true, y_pred,
            os.path.join(args.plots_dir, "score_actual_vs_pred.png"),
        )

        # ---------------------------
        # Save model
        # ---------------------------
        model.write().overwrite().save(args.model_dir)

        # ---------------------------
        # OPTIONAL: export sample CSV
        # ---------------------------
        if args.sample_csv:
            print(f"Exporting sample to {args.sample_csv} ...")
            sample_df = df.limit(args.sample_size).toPandas()

            out_path = Path(args.sample_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sample_df.to_csv(out_path, index=False)

            print(f"✓ Saved sklearn sample CSV at {out_path}")

        print("\n✓ Score regression completed successfully.")
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    finally:
        if spark:
            spark.stop()


if __name__ == "__main__":
    main()
