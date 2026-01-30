import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
csv_dir = "../data/csv"
plot_dir = "../data/plots"
os.makedirs(plot_dir, exist_ok=True)

# -------------------------------------------------------------
# Load CSVs
# -------------------------------------------------------------
missing_df = pd.read_csv(f"{csv_dir}/missingness_by_subreddit.csv")
deleted_df = pd.read_csv(f"{csv_dir}/deleted_removed_subreddit.csv")
score_df = pd.read_csv(f"{csv_dir}/score_stats_by_subreddit.csv")
text_df = pd.read_csv(f"{csv_dir}/text_quality_by_subreddit.csv")
sub_stats = pd.read_csv(f"{csv_dir}/subreddit_statistics.csv")  # needed for null normalization

# Keep only subreddit + total row count
sub_stats = sub_stats[["subreddit", "total_rows"]]

# -------------------------------------------------------------
# Normalize missingness → convert raw null counts → percent null
# -------------------------------------------------------------
df = missing_df.merge(sub_stats, on="subreddit", how="left")

null_cols = [c for c in df.columns if c.endswith("_nulls")]

for c in null_cols:
    df[c.replace("_nulls", "_null_pct")] = df[c] / df["total_rows"] * 100

pct_cols = [c for c in df.columns if c.endswith("_null_pct")]
missing_pct_df = df.set_index("subreddit")[pct_cols]

# -------------------------------------------------------------
# 1. Score Distribution Boxplot (using quartiles properly)
# -------------------------------------------------------------
plt.figure(figsize=(14,6))

score_long = score_df.melt(
    id_vars="subreddit",
    value_vars=["min_score", "q1", "median", "q3", "max_score"],
    var_name="stat",
    value_name="score"
)

sns.boxplot(data=score_long, x="subreddit", y="score")

plt.yscale("symlog", linthresh=10)   # <<< shows negatives + compresses large values
plt.xticks(rotation=90)
plt.title("Score Summary Stats per Subreddit (Symlog Scale)")
plt.tight_layout()
plt.savefig(f"{plot_dir}/score_boxplot_symlog.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# 2. Missingness Heatmap (percentage version — more meaningful)
# -------------------------------------------------------------
plt.figure(figsize=(16, 10))
sns.heatmap(missing_pct_df, cmap="Reds", vmin=0, vmax=100)
plt.title("Percent Missing per Column per Subreddit")
plt.tight_layout()
plt.savefig(f"{plot_dir}/missingness_pct_heatmap.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# 3. Deleted Content Barplot
# -------------------------------------------------------------
delete_cols = ["pct_author_deleted", "pct_body_removed", "pct_selftext_removed"]
available_cols = [c for c in delete_cols if c in deleted_df.columns]

plt.figure(figsize=(14, 6))
deleted_df.plot(
    x="subreddit",
    y=available_cols,
    kind="bar",
    figsize=(14, 6)
)
plt.xticks(rotation=90)
plt.ylabel("Percentage (%)")
plt.title("Deleted / Removed Content (%) per Subreddit")
plt.tight_layout()
plt.savefig(f"{plot_dir}/deleted_content_bar.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# 4. Text Length Histogram
# -------------------------------------------------------------
plt.figure(figsize=(14, 6))
sns.histplot(text_df["avg_body_len"], bins=30)
plt.title("Histogram of Average Body Length per Subreddit")
plt.xlabel("Average Body Length (characters)")
plt.tight_layout()
plt.savefig(f"{plot_dir}/text_length_histogram.png", dpi=300)
plt.close()

print("\nAll plots saved to:", plot_dir)
