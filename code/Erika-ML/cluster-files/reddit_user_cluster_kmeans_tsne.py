#!/usr/bin/env python3
"""
file: reddit_user_cluster_kmeans_tsne.py

Task:
    - Read user-level topic features (CSV)
    - Run KMeans clustering on topic columns
    - Compute 2D t-SNE for visualization
    - Attach human-readable cluster labels (personas)
    - Save:
        * CSV with user + topics + cluster_id + cluster_label
        * PNG scatter plot (t-SNE colored by cluster & annotated with labels)
        * Cluster summary CSV (n_users, comment stats, primary_tool mode)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# -------------------------------------------------------------------
# Human-readable cluster labels (personas) for AI tools / ads story
# -------------------------------------------------------------------
cluster_labels = {
    0: "Power devs: coding help & debugging",
    1: "Prompt experimenters & jailbreak crowd",
    2: "Career & job search / upskilling",
    3: "Tool comparisons & which AI should I use?",
    4: "Product feedback, reliability & bug reports",
    5: "Creators & marketers (content generation)",
    6: "Students & learners (homework / studying)",
    7: "Ethics, safety & policy debates",
    8: "API / integrations & plugins builders",
    9: "General chat / off-topic exploration",
    # If n_clusters != 10, unknown IDs will fall back to "Cluster k"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster Reddit users via LDA topic vectors + t-SNE visualization"
    )
    parser.add_argument(
        "--input-user-features",
        required=True,
        help="CSV with user-level topic features (output of reddit_user_prepare_features.py)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of KMeans clusters (default: 10)",
    )
    parser.add_argument(
        "--output-clusters-csv",
        required=True,
        help="Output CSV path with user features + cluster labels",
    )
    parser.add_argument(
        "--output-tsne-plot",
        required=True,
        help="Output PNG path for 2D t-SNE scatter plot",
    )
    return parser.parse_args()


def persona_label_for_cluster(c: int) -> str:
    """Return a human-readable label for cluster c."""
    return cluster_labels.get(c, f"Cluster {c}")


def main():
    args = parse_args()

    # -------------------------------------------------------------
    # Load user features
    # -------------------------------------------------------------
    print(f"🔹 Loading user features from: {args.input_user_features}")
    df = pd.read_csv(args.input_user_features)
    print(f"   Rows: {len(df):,}, columns: {list(df.columns)}")

    # Identify topic columns
    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    if not topic_cols:
        raise ValueError("No columns starting with 'topic_' found in input user features.")

    print(f"🔹 Found {len(topic_cols)} topic columns: {topic_cols[0]} ... {topic_cols[-1]}")
    X = df[topic_cols].values

    # -------------------------------------------------------------
    # 1) KMeans clustering
    # -------------------------------------------------------------
    print(f"🔹 Running KMeans with k={args.n_clusters} ...")
    km = KMeans(
        n_clusters=args.n_clusters,
        n_init=10,
        random_state=42,
    )
    df["cluster"] = km.fit_predict(X)
    print("    KMeans done.")

    # Attach human-readable persona labels
    df["cluster_label"] = df["cluster"].map(persona_label_for_cluster)
    print("🔹 Example cluster labels:")
    print(df[["author", "cluster", "cluster_label"]].head())

    # -------------------------------------------------------------
    # 2) 2D t-SNE
    # -------------------------------------------------------------
    print("🔹 Computing 2D t-SNE embedding...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="random",
        learning_rate="auto",
    )
    X_embedded = tsne.fit_transform(X)
    df["tsne_1"] = X_embedded[:, 0]
    df["tsne_2"] = X_embedded[:, 1]
    print("    t-SNE done.")

    # -------------------------------------------------------------
    # 3) Save clustered CSV
    # -------------------------------------------------------------
    out_csv = Path(args.output_clusters_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved clustered user features → {out_csv} (rows: {len(df):,})")

    # -------------------------------------------------------------
    # 4) Cluster-level summary for business analysis
    # -------------------------------------------------------------
    print("🔹 Building cluster summary (n_users, comment stats, primary_tool mode)...")

    summary_rows = []
    has_comment_count = "comment_count" in df.columns
    has_primary_tool = "primary_tool" in df.columns

    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        row = {
            "cluster": int(c),
            "cluster_label": persona_label_for_cluster(int(c)),
            "n_users": int(len(sub)),
        }

        if has_comment_count:
            row["mean_comment_count"] = float(sub["comment_count"].mean())
            row["median_comment_count"] = float(sub["comment_count"].median())
        else:
            row["mean_comment_count"] = np.nan
            row["median_comment_count"] = np.nan

        if has_primary_tool:
            tools = sub["primary_tool"].dropna()
            row["top_primary_tool"] = tools.value_counts().idxmax() if not tools.empty else None
        else:
            row["top_primary_tool"] = None

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_csv.with_name(out_csv.stem + "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved cluster summary → {summary_path}")

    # -------------------------------------------------------------
    # 5) Plot t-SNE with cluster colors + persona labels at centroids
    # -------------------------------------------------------------
    out_plot = Path(args.output_tsne_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df["tsne_1"],
        df["tsne_2"],
        c=df["cluster"],
        s=8,
        alpha=0.7,
    )
    plt.title("Reddit User Segmentation (t-SNE of Topic Vectors)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Compute centroid per cluster and annotate with persona label
    centers = (
        df.groupby("cluster")[["tsne_1", "tsne_2"]]
          .mean()
          .reset_index()
    )

    for _, row in centers.iterrows():
        c = int(row["cluster"])
        x, y = row["tsne_1"], row["tsne_2"]
        label = persona_label_for_cluster(c)
        plt.text(
            x,
            y,
            f"{c}: {label}",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.6),
        )

    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()

    print(f"✅ Saved t-SNE plot → {out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
