#!/usr/bin/env python3
"""
NLP-Q1-3 Visualization: Dominant Topics in AI-related Reddit Comments

Plots included:
1. Topic size bar chart
2. Labeled top topics horizontal bar chart
3. Word cloud per topic (optional)
4. Topic-term heatmap
5. Subreddit x Topic heatmap
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional wordcloud support
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("[INFO] WordCloud package not installed. Word clouds will be skipped.")

print("=" * 80)
print("NLP-Q1-Anna: Topic Visualization")
print("=" * 80)

# ---------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------
topics_csv = "./data/NLPQ1_Anna_spark_lda_topics_ver2.csv"
counts_csv = "./data/NLPQ1_Anna_spark_lda_topic_counts_ver2.csv"
sub_topic_csv = "./data/NLPQ1_Anna_subreddit_topic_counts_ver2.csv"

topics_df = pd.read_csv(topics_csv)
counts_df = pd.read_csv(counts_csv)
sub_topic_df = pd.read_csv(sub_topic_csv)

print(f"Loaded topics: {len(topics_df)} rows")
print(f"Loaded topic counts: {len(counts_df)} rows")
print(f"Loaded subreddit-topic counts: {len(sub_topic_df)} rows")

os.makedirs("data/plots", exist_ok=True)

# Normalize column names
if "topic" in topics_df.columns:
    topics_df = topics_df.rename(columns={"topic": "topic_id"})
if "dominant_topic" in counts_df.columns:
    counts_df = counts_df.rename(columns={"dominant_topic": "topic_id"})

# Merge topic info with counts
merged = topics_df.merge(counts_df, on="topic_id", how="left")
merged["count"] = merged["count"].fillna(0).astype(int)

# Parse terms list safely
def parse_terms_words(x):
    if isinstance(x, str):
        cleaned = x.strip("[]").replace("'", "").replace('"', "")
        terms = [t.strip() for t in cleaned.split(",") if t.strip()]
        return terms
    elif isinstance(x, list):
        return x
    return []

merged["terms_list"] = merged["terms_words"].apply(parse_terms_words)

def label_from_terms(terms, top_n=4):
    return ", ".join(terms[:top_n])

merged["topic_label"] = merged["terms_list"].apply(label_from_terms)

# ---------------------------------------------------------------------
# 2. Topic size bar chart
# ---------------------------------------------------------------------
print("\n[1/5] Creating topic size bar chart...")

merged_sorted = merged.sort_values("topic_id")

plt.figure(figsize=(10, 5))
plt.bar(merged_sorted["topic_id"].astype(str), merged_sorted["count"])
plt.xlabel("Topic ID", fontsize=12)
plt.ylabel("Number of Comments", fontsize=12)
plt.title("Topic Prevalence (Number of Comments per Topic)", fontsize=14, fontweight="bold")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("data/plots/NLPQ1_Anna_topic_sizes.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# 3. Labeled top topics horizontal bar chart
# ---------------------------------------------------------------------
print("\n[2/5] Creating labeled top topics chart...")

TOP_N = 10
top_n = merged.sort_values("count", ascending=False).head(TOP_N)

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_n)), top_n["count"])
plt.yticks(range(len(top_n)), top_n["topic_label"])
plt.xlabel("Number of Comments", fontsize=12)
plt.title(f"Top {TOP_N} Topics in AI-related Reddit Comments", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("data/plots/NLPQ1_Anna_top_topics_labeled.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# 4. Word cloud per topic-
# ---------------------------------------------------------------------
if HAS_WORDCLOUD:
    print("\n[3/5] Generating word clouds per topic...")

    n_topics = len(merged)
    ncols = 4
    nrows = (n_topics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, row in merged.iterrows():
        ax = axes[idx]
        terms = row["terms_list"]
        text = " ".join(terms * 3)
        wc = WordCloud(width=400, height=400, background_color="white").generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Topic {row['topic_id']}", fontsize=10)

    # Remove unused axes
    for j in range(len(merged), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Word Clouds per Topic", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/plots/NLPQ1_Anna_topic_wordclouds.png", dpi=300, bbox_inches="tight")
    plt.close()

else:
    print("[INFO] Skipping word clouds due to missing WordCloud package.")

# ---------------------------------------------------------------------
# 5. Topic-term heatmap
# ---------------------------------------------------------------------
print("\n[4/5] Creating topic-term heatmap...")

def parse_weights(x):
    if isinstance(x, str):
        cleaned = x.strip("[]")
        vals = [float(t.strip()) for t in cleaned.split(",") if t.strip()]
        return vals
    elif isinstance(x, list):
        return x
    return []

topics_df["weights_list"] = topics_df["termWeights"].apply(parse_weights)

MAX_TERMS = 10
heat_data = []

for _, row in topics_df.merge(merged[["topic_id", "terms_list"]], on="topic_id").iterrows():
    weights = row["weights_list"][:MAX_TERMS]
    if len(weights) < MAX_TERMS:
        weights += [0.0] * (MAX_TERMS - len(weights))
    heat_data.append(weights)

heat_df = pd.DataFrame(
    heat_data,
    index=[f"T{tid}" for tid in merged_sorted["topic_id"]],
    columns=[f"Rank{i+1}" for i in range(MAX_TERMS)]
)

plt.figure(figsize=(12, 8))
sns.heatmap(heat_df, cmap="YlOrRd", cbar_kws={"label": "Term Weight"}, annot=False)
plt.title("Topic-Term Heatmap (Top Ranked Terms per Topic)", fontsize=14, fontweight="bold")
plt.xlabel("Term Rank", fontsize=12)
plt.ylabel("Topic", fontsize=12)
plt.tight_layout()
plt.savefig("data/plots/NLPQ1_Anna_topic_term_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# 6. Subreddit × Topic heatmap
# ---------------------------------------------------------------------
print("\n[5/5] Creating Subreddit × Topic heatmap...")

sub_topic_pivot = sub_topic_df.pivot_table(
    index="subreddit",
    columns="dominant_topic",
    values="count",
    aggfunc="sum",
    fill_value=0
)

sub_topic_pivot.columns = [f"T{int(c)}" for c in sub_topic_pivot.columns]

plt.figure(figsize=(12, max(6, 0.5 * len(sub_topic_pivot))))
sns.heatmap(sub_topic_pivot, cmap="Blues", linewidths=0.5)
plt.title("Subreddit × Dominant Topic Heatmap", fontsize=14, fontweight="bold")
plt.xlabel("Topic ID", fontsize=12)
plt.ylabel("Subreddit", fontsize=12)
plt.tight_layout()
plt.savefig("data/plots/NLPQ1_Anna_subreddit_topic_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nVisualization COMPLETE!")
print("=" * 80)
