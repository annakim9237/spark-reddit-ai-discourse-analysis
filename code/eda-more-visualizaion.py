#!/usr/bin/env python3
"""
Generate bar plots for:
1) total_rows by subreddit
2) avg_score by subreddit
3) monthly total_rows over time

Inputs (relative to project root):
- data/csv/subreddit_statistics.csv
- data/csv/temporal_distribution.csv

Outputs (saved under project_root/data/plots):
- subreddit_total_rows.png
- subreddit_avg_score.png
- total_rows_by_month.png

Assumption:
- This script is located in ./code at the project root level.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# -------------------------------------------------------------------
# Resolve paths
# -------------------------------------------------------------------
# This file is in ./code, so project_root is one level up
this_file = Path(__file__).resolve()
project_root = this_file.parents[1]

csv_dir = project_root / "data" / "csv"
plots_dir = project_root / "data" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

subreddit_csv = csv_dir / "subreddit_statistics.csv"
temporal_csv = csv_dir / "temporal_distribution.csv"

# -------------------------------------------------------------------
# 1) Subreddit statistics plots
# -------------------------------------------------------------------
df_sub = pd.read_csv(subreddit_csv)

# Sort for nicer plotting
df_total = df_sub.sort_values("total_rows", ascending=False)
df_score = df_sub.sort_values("avg_score", ascending=False)

# --- (1) Total rows by subreddit ---
plt.figure(figsize=(10, 6))
ax = plt.gca()

ax.bar(df_total["subreddit"], df_total["total_rows"])
ax.set_title("Total Rows by Subreddit")
ax.set_xlabel("Subreddit")
ax.set_ylabel("Total Rows")
plt.xticks(rotation=45, ha="right")

# Turn off scientific notation & offset, format with comma separators
ax.ticklabel_format(axis="y", style="plain", useOffset=False)
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

plt.tight_layout()
plt.savefig(plots_dir / "subreddit_total_rows.png", dpi=300)
plt.close()

# --- (2) Average score by subreddit ---
plt.figure(figsize=(10, 6))
ax = plt.gca()

ax.bar(df_score["subreddit"], df_score["avg_score"])
ax.set_title("Average Score by Subreddit")
ax.set_xlabel("Subreddit")
ax.set_ylabel("Average Score")
plt.xticks(rotation=45, ha="right")

# Scores are small, but we still disable scientific notation just in case
ax.ticklabel_format(axis="y", style="plain", useOffset=False)

plt.tight_layout()
plt.savefig(plots_dir / "subreddit_avg_score.png", dpi=300)
plt.close()

# -------------------------------------------------------------------
# 2) Temporal distribution plot (monthly total_rows)
# -------------------------------------------------------------------
df_temp = pd.read_csv(temporal_csv)

# Ensure months are sorted in chronological order
df_temp = df_temp.sort_values("year_month")

plt.figure(figsize=(10, 6))
ax = plt.gca()

ax.bar(df_temp["year_month"], df_temp["total_rows"])
ax.set_title("Total Rows by Month")
ax.set_xlabel("Year-Month")
ax.set_ylabel("Total Rows")
plt.xticks(rotation=45, ha="right")

# Again, turn off scientific notation and use comma formatting
ax.ticklabel_format(axis="y", style="plain", useOffset=False)
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

plt.tight_layout()
plt.savefig(plots_dir / "total_rows_by_month.png", dpi=300)
plt.close()

print(f"Saved plots to: {plots_dir}")
