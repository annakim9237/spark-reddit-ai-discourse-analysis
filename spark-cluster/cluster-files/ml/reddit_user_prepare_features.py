#!/usr/bin/env python3
"""
file: reddit_user_prepare_features.py

Task:
    - Read cleaned Reddit comments (parquet or CSV)
    - Fit LDA topic model on comments
    - Compute per-comment topic distributions
    - Aggregate to user-level topic vectors (mean over comments)
    - Optionally derive a 'primary_tool' per user from subreddit
    - Save user feature matrix to CSV
    - Optionally save topic -> top words mapping to CSV for interpretation

Assumptions (for typical DSAN6000 cluster run):
    - Input parquet from reddit_nlp_cleaning.py has at least:
        * author
        * processed_text
        * subreddit (optional but useful)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build user topic features from Reddit comments (LDA + user aggregation)"
    )

    parser.add_argument(
        "--input-parquet",
        help="Path to cleaned parquet (directory or file), e.g. data/parquet/reddit_cleaned.parquet",
    )
    parser.add_argument(
        "--input-csv",
        help="Alternative: input CSV with cleaned comments.",
    )

    parser.add_argument(
        "--text-column",
        default="processed_text",
        help="Name of text column with cleaned text (default: processed_text)",
    )
    parser.add_argument(
        "--author-column",
        default="author",
        help="Name of author column (default: author)",
    )
    parser.add_argument(
        "--subreddit-column",
        default="subreddit",
        help="Name of subreddit column (if present, used to derive primary_tool).",
    )

    parser.add_argument(
        "--num-topics",
        type=int,
        default=15,
        help="Number of LDA topics (default: 15)",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=200_000,
        help="Max # of comments to use for LDA (for speed). Use -1 for all.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=5,
        help="Min document frequency for CountVectorizer (default: 5)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
        help="Max vocabulary size (default: 50k)",
    )

    parser.add_argument(
        "--output-user-features",
        required=True,
        help="Output CSV with user-level topic features.",
    )

    parser.add_argument(
        "--output-topic-words",
        help="Optional: CSV path to save topic_id -> top words for each topic.",
    )
    parser.add_argument(
        "--top-words-per-topic",
        type=int,
        default=15,
        help="Number of top words to export per topic (default: 15)",
    )

    args = parser.parse_args()

    if not args.input_parquet and not args.input_csv:
        parser.error("You must supply --input-parquet OR --input-csv")

    return args


# -------------------------------------------------------------------
# Helper: map subreddit -> primary_tool label
# -------------------------------------------------------------------
def map_subreddit_to_tool(subreddit: str | float) -> str | None:
    """
    Map subreddit name to a canonical AI tool 'primary_tool' label.
    Returns None if subreddit is not one of the AI tools of interest.
    """
    if pd.isna(subreddit):
        return None

    s = str(subreddit).strip().upper()
    mapping = {
        "CHATGPT": "ChatGPT",
        "OPENAI": "OpenAI",
        "GPT4": "GPT4",
        "CLAUDEAI": "ClaudeAI",
        "PERPLEXITYAI": "PerplexityAI",
    }
    return mapping.get(s, None)


def majority_label(series: pd.Series) -> str | None:
    """Get the most frequent non-null label in a series, or None if none."""
    s = series.dropna()
    if s.empty:
        return None
    return s.value_counts().idxmax()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    args = parse_args()

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------
    if args.input_parquet:
        print(f"🔹 Loading parquet: {args.input_parquet}")
        df = pd.read_parquet(args.input_parquet)
    else:
        print(f"🔹 Loading CSV: {args.input_csv}")
        df = pd.read_csv(args.input_csv)

    print(f"🔹 Loaded {len(df):,} rows with columns: {list(df.columns)}")

    if args.text_column not in df.columns:
        raise ValueError(
            f"text column '{args.text_column}' not found in input data. "
            f"Available columns: {list(df.columns)}"
        )
    if args.author_column not in df.columns:
        raise ValueError(
            f"author column '{args.author_column}' not found in input data. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only relevant columns
    keep_cols = [args.author_column, args.text_column]
    if args.subreddit_column in df.columns:
        keep_cols.append(args.subreddit_column)

    df = df[keep_cols].dropna(subset=[args.author_column, args.text_column])
    print(f"🔹 After dropna(author, text): {len(df):,} comments")

    # Optional subsample for speed
    if args.max_comments > 0 and len(df) > args.max_comments:
        df = df.sample(n=args.max_comments, random_state=42)
        print(f"🔹 Subsampled to {len(df):,} comments for LDA")

    # -------------------------------------------------------------
    # Optional: derive per-comment primary_tool from subreddit
    # -------------------------------------------------------------
    if args.subreddit_column in df.columns:
        print(f"🔹 Deriving per-comment primary_tool from '{args.subreddit_column}'...")
        df["primary_tool_comment"] = df[args.subreddit_column].apply(map_subreddit_to_tool)
        has_any_tool = df["primary_tool_comment"].notna().any()
        print(f"    Any AI-tool subreddit rows? {has_any_tool}")
    else:
        print("⚠️ No subreddit column found; primary_tool will not be available.")
        df["primary_tool_comment"] = None

    # -------------------------------------------------------------
    # Step 1: Vectorize text
    # -------------------------------------------------------------
    print("🔹 Vectorizing text with CountVectorizer...")
    vect = CountVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        stop_words="english",
    )
    X = vect.fit_transform(df[args.text_column])
    print(f"    Shape of document-term matrix: {X.shape}")

    # -------------------------------------------------------------
    # Step 2: Fit LDA
    # -------------------------------------------------------------
    print(f"🔹 Fitting LDA with {args.num_topics} topics...")
    lda = LatentDirichletAllocation(
        n_components=args.num_topics,
        learning_method="batch",
        max_iter=20,
        random_state=42,
        n_jobs=-1,
    )
    doc_topic = lda.fit_transform(X)  # shape: (n_docs, num_topics)
    print("    LDA finished.")

    topic_cols = [f"topic_{k}" for k in range(args.num_topics)]

    # -------------------------------------------------------------
    # Step 3: Build per-comment topic DF
    # -------------------------------------------------------------
    doc_topic_df = pd.DataFrame(doc_topic, columns=topic_cols)
    doc_topic_df[args.author_column] = df[args.author_column].values
    doc_topic_df["primary_tool_comment"] = df["primary_tool_comment"].values

    # -------------------------------------------------------------
    # Step 4: Aggregate to user-level topic features
    # -------------------------------------------------------------
    print("🔹 Aggregating to user-level topic vectors (mean topic dist + comment_count + primary_tool)...")

    # Group by author (as_index=True, then reset_index where needed)
    group = doc_topic_df.groupby(args.author_column)

    # Mean topic distribution per user
    user_topics = (
        group[topic_cols]
        .mean()
        .reset_index()  # bring author column back as a regular column
    )

    # Comment count per user
    comment_count = (
        group.size()
        .reset_index(name="comment_count")
    )

    # Merge comment_count
    user_topics = user_topics.merge(comment_count, on=args.author_column, how="left")

    # Majority primary_tool per user (if any tools present)
    if doc_topic_df["primary_tool_comment"].notna().any():
        primary_tool = (
            group["primary_tool_comment"]
            .agg(majority_label)
            .reset_index(name="primary_tool")
        )
        user_topics = user_topics.merge(primary_tool, on=args.author_column, how="left")
    else:
        user_topics["primary_tool"] = None

    # Sort by comment_count (optional)
    user_topics = user_topics.sort_values("comment_count", ascending=False)

    out_path = Path(args.output_user_features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    user_topics.to_csv(out_path, index=False)
    print(f"✅ Saved user topic features → {out_path} (rows: {len(user_topics):,})")

    # -------------------------------------------------------------
    # Optional: export topic → top words for interpretability
    # -------------------------------------------------------------
    if args.output_topic_words:
        print(f"🔹 Exporting top {args.top_words_per_topic} words per topic → {args.output_topic_words}")
        inv_vocab = {idx: term for term, idx in vect.vocabulary_.items()}

        rows = []
        for topic_idx, topic_weights in enumerate(lda.components_):
            top_indices = np.argsort(topic_weights)[::-1][: args.top_words_per_topic]
            top_terms = [inv_vocab[i] for i in top_indices]
            rows.append(
                {
                    "topic_id": topic_idx,
                    "top_words": ", ".join(top_terms),
                }
            )

        topic_words_df = pd.DataFrame(rows)
        Path(args.output_topic_words).parent.mkdir(parents=True, exist_ok=True)
        topic_words_df.to_csv(args.output_topic_words, index=False)
        print(f"✅ Saved topic word summaries → {args.output_topic_words}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
