#!/usr/bin/env python3
"""
file: reddit_score_feature_importance.py

Driver-side (sklearn) feature importance analysis for Reddit comment score
regression.

EXPECTED INPUT CSV (from Spark sample) should contain at least:
    body_cleaned      (string, cleaned comment text)
    subreddit         (string)
    token_count       (numeric)
    controversiality  (numeric)
    gilded            (numeric)
    hour_of_day       (numeric)
    day_of_week       (numeric)
    is_weekend        (numeric)
    body_length       (numeric)
    score             (numeric target)

Typical source:
    reddit_df.limit(50000).toPandas().to_csv("reddit_score_sample.csv", index=False)

OUTPUT (when called with --output-dir data):
    data/csv/reddit_score_feature_importance.csv
    data/plots/reddit_score_feature_importance.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -------------------------------------------------------------------
# Helper: Build preprocessing pipeline
# -------------------------------------------------------------------
def build_preprocess_pipeline(df: pd.DataFrame):
    """
    Build a ColumnTransformer using the columns actually present in df.

    Text: TF-IDF on body_cleaned
    Categorical: OneHotEncoder on subreddit
    Numeric: StandardScaler on numeric feature columns
    """

    # Text column (from Spark cleaning)
    text_col = "body_cleaned"
    if text_col not in df.columns:
        raise ValueError(f"Expected text column '{text_col}' not found in input CSV.")

    # Categorical columns
    categorical_cols = [c for c in ["subreddit"] if c in df.columns]

    # Numeric feature candidates – use intersection with df
    numeric_candidates = [
        "token_count",
        "controversiality",
        "gilded",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "body_length",
    ]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]

    if not numeric_cols:
        raise ValueError("No numeric feature columns found in input CSV.")

    tfidf = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        sublinear_tf=True,
    )

    transformers = [
        ("txt", tfidf, text_col),
        ("num", StandardScaler(with_mean=False), numeric_cols),
    ]

    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )

    pre = ColumnTransformer(transformers=transformers)
    return pre, text_col, numeric_cols, categorical_cols


# -------------------------------------------------------------------
# Helper: Extract expanded feature names
# -------------------------------------------------------------------
def extract_feature_names(
    preprocess: ColumnTransformer,
    text_col: str,
    numeric_cols,
    categorical_cols,
):
    names = []

    # 1. TF-IDF feature names
    tfidf = preprocess.named_transformers_["txt"]
    tfidf_names = tfidf.get_feature_names_out().tolist()
    names.extend(tfidf_names)

    # 2. Numeric feature names (already known)
    names.extend(numeric_cols)

    # 3. OHE category names (if we had categorical columns)
    if categorical_cols:
        ohe = preprocess.named_transformers_["cat"]
        ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
        names.extend(ohe_names)

    return names


# -------------------------------------------------------------------
# Helper: Plot feature importance
# -------------------------------------------------------------------
def plot_feature_importance(df_imp, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 12))
    sns.barplot(
        data=df_imp,
        x="coefficient",
        y="feature",
        palette=["#239b56" if c > 0 else "#cb4335" for c in df_imp["coefficient"]],
    )
    plt.title("Top Positive & Negative Coefficients (Score Regression)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Feature importance for Reddit comment score regression (sklearn)"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV exported from Spark (reddit_score_sample.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Base output directory. CSV → <output-dir>/csv/, plot → <output-dir>/plots/",
    )

    args = parser.parse_args()

    base_out = Path(args.output_dir)
    csv_out = base_out / "csv" / "reddit_score_feature_importance.csv"
    plot_out = base_out / "plots" / "reddit_score_feature_importance.png"

    print("=" * 70)
    print("REDDIT SCORE FEATURE IMPORTANCE (SKLEARN REGRESSION)")
    print("=" * 70)
    print(f"Input CSV        : {args.input}")
    print(f"Output CSV       : {csv_out}")
    print(f"Output Plot      : {plot_out}")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------
    df = pd.read_csv(args.input)

    if "score" not in df.columns:
        raise ValueError("Input CSV is missing required target column 'score'.")

    # Build preprocessing based on available columns
    preprocess, text_col, numeric_cols, categorical_cols = build_preprocess_pipeline(df)

    feature_cols = [text_col] + numeric_cols + categorical_cols
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns in input CSV: {missing_feats}")

    X = df[feature_cols]
    y = df["score"]

    # ------------------------------------------------------------
    # 2. Train/test split
    # ------------------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------
    # 3. Build sklearn regression pipeline
    # ------------------------------------------------------------
    model = SKPipeline(
        steps=[
            ("pre", preprocess),
            ("reg", Ridge(alpha=1.0)),
        ]
    )

    model.fit(X_tr, y_tr)

    # ------------------------------------------------------------
    # 4. Extract coefficients
    # ------------------------------------------------------------
    coef = model.named_steps["reg"].coef_
    feat_names = extract_feature_names(
        model.named_steps["pre"],
        text_col,
        numeric_cols,
        categorical_cols,
    )

    df_imp = pd.DataFrame({
        "feature": feat_names,
        "coefficient": coef,
    }).sort_values("coefficient")

    # Top +/- 25 features
    k = 25
    df_top = pd.concat([df_imp.head(k), df_imp.tail(k)])

    # ------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_top.to_csv(csv_out, index=False)
    print(f"✓ Saved feature importance CSV → {csv_out}")

    plot_feature_importance(df_top, plot_out)
    print(f"✓ Saved feature importance plot → {plot_out}")

    print("=" * 70)
    print("✅ Score regression feature importance complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
