"""
LOCAL VISUALIZATION PIPELINE USING MERGED CSVs

MERGED SCHEMAS (confirmed from user):

posters_merged:
    author,total_posts,num_subreddits,avg_score,max_score,
    first_post,last_post,active_span_seconds,
    num_tech_subreddits,tech_subreddits_str

examples_merged:
    subreddit,author,title,body,score,created_utc

overlap_merged:
    subreddit,subreddit.1,shared_authors

Outputs → data/plots/
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def ensure_dirs():
    os.makedirs("data/plots", exist_ok=True)


def main():

    ensure_dirs()
    sns.set(style="whitegrid")

    # ---------------------------------------------------------
    # LOAD CSVs WITH CORRECT SCHEMAS
    # ---------------------------------------------------------
    posters = pd.read_csv("data/csv/posters_merged.csv")
    examples = pd.read_csv("data/csv/examples_merged.csv")
    overlap = pd.read_csv("data/csv/overlap_merged.csv")

    print("\nLoaded merged CSVs:")
    print("posters:", posters.shape)
    print("examples:", examples.shape)
    print("overlap:", overlap.shape)

    # ---------------------------------------------------------
    # SPECIALISTS VS GENERALISTS
    # ---------------------------------------------------------
    print("\n=== Visualization 1: Specialist / Generalist Breakdown ===")

    def classify(n):
        if n == 1:
            return "Specialist"
        elif n <= 3:
            return "Broad (2–3)"
        else:
            return "Generalist (4+)"

    posters["category"] = posters["num_tech_subreddits"].apply(classify)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=posters, x="category", palette="magma", order=["Specialist", "Broad (2–3)", "Generalist (4+)"])
    plt.title("Specialists vs Broad vs Generalist (Top 1% Contributors)")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("data/plots/specialization_breakdown.png")
    plt.close()

    # ---------------------------------------------------------
    # TOP SUBREDDIT OVERLAP PAIRS
    # ---------------------------------------------------------
    print("\n=== Visualization 2: Top Subreddit Co-Engagement Pairs ===")

    # Your actual column names:
    sub1 = "subreddit"
    sub2 = "subreddit.1"

    overlap_sorted = overlap.sort_values("shared_authors", ascending=False).head(25)

    overlap_sorted["pair"] = overlap_sorted[sub1] + " ↔ " + overlap_sorted[sub2]

    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=overlap_sorted,
        x="shared_authors",
        y="pair",
        palette="viridis"
    )
    plt.title("Top 25 Subreddit Co-Engagement Pairs")
    plt.xlabel("Shared Authors")
    plt.ylabel("Subreddit Pair")
    plt.tight_layout()
    plt.savefig("data/plots/subreddit_overlap_top_pairs.png")
    plt.close()

    # ---------------------------------------------------------
    #  ACTIVE SPAN DISTRIBUTION
    # ---------------------------------------------------------
    print("\n=== Visualization 3: Active Span Distribution ===")

    posters["active_span_years"] = posters["active_span_seconds"] / (86400 * 365)

    plt.figure(figsize=(10, 6))
    sns.histplot(posters["active_span_years"], bins=40, kde=True, color="purple")
    plt.title("Distribution of Active Years for Top Contributors")
    plt.xlabel("Years Active")
    plt.tight_layout()
    plt.savefig("data/plots/active_span_distribution.png")
    plt.close()

    # ---------------------------------------------------------
    # TOP POSTS DISTRIBUTION (VIBE CHECK)
    # ---------------------------------------------------------
    print("\n=== Visualization 4: Top Posts (Vibe Check) ===")

    top_posts = examples.sort_values(by="score", ascending=False).head(30)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_posts,
        x="score",
        y="subreddit",
        palette="rocket"
    )
    plt.title("Highest-Scoring Posts — Subreddit Distribution")
    plt.xlabel("Score")
    plt.ylabel("Subreddit")
    plt.tight_layout()
    plt.savefig("data/plots/top_post_scores_distribution.png")
    plt.close()

    # ---------------------------------------------------------
    # SUBREDDIT OVERLAP HEATMAP
    # ---------------------------------------------------------
    print("\n=== Visualization 5: Subreddit Overlap Heatmap ===")

    try:
        heatmap_df = overlap.pivot(
            index=sub1,
            columns=sub2,
            values="shared_authors"
        ).fillna(0)

        plt.figure(figsize=(14, 12))
        sns.heatmap(heatmap_df, cmap="Blues", linewidths=.5)
        plt.title("Subreddit Overlap Heatmap — AI/Tech Ecosystem")
        plt.tight_layout()
        plt.savefig("data/plots/subreddit_overlap_heatmap.png")
        plt.close()

    except Exception as e:
        print("Skipping heatmap — pivot error:", e)

    print("\n🎉 All visualizations saved in data/plots/\n")


if __name__ == "__main__":
    main()
