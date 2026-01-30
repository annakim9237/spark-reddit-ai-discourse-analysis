#!/usr/bin/env python3
"""
EDA: Optimal Posting Time Analysis for AI-related Subreddits
When is the optimal time to post content to maximize engagement?

Visualize optimal posting times using heatmaps, bar charts, and scatter plots

The csv files used are generated from EDA-Anna-csv.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory
os.makedirs("data/plots", exist_ok=True)

print("=" * 80)
print("POSTING TIME VISUALIZATION")
print("=" * 80)

# Load data
print("\n[1/5] Loading CSV files...")
overall_df = pd.read_csv("data/csv/EDA_Anna_posting_time_analysis.csv")
subreddit_df = pd.read_csv("data/csv/EDA_Anna_subreddit_time_analysis.csv")
print(f"Loaded overall data: {len(overall_df)} rows")
print(f"Loaded subreddit data: {len(subreddit_df)} rows")

# Day of week mapping (1=Sunday, 7=Saturday)
day_map = {1: 'Sun', 2: 'Mon', 3: 'Tue', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat'}
overall_df['day_name'] = overall_df['day_of_week'].map(day_map)
subreddit_df['day_name'] = subreddit_df['day_of_week'].map(day_map)

# 1. Heatmap: Posting Volume (Overall)
print("\n[2/5] Creating heatmap for posting volume...")

plt.figure(figsize=(12, 6))
heatmap_data = overall_df.pivot(index='day_name', columns='hour', values='num_posts')
# Reorder rows to Monday-Sunday
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
heatmap_data = heatmap_data.reindex(day_order)

sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='g', cbar_kws={'label': 'Number of Posts'})
plt.title('Posting Activity by Day and Hour', fontsize=16, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Day of Week', fontsize=12)
plt.tight_layout()
plt.savefig('data/plots/EDA_Anna_heatmap_posting_volume.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: data/plots/EDA_Anna_heatmap_posting_volume.png")

# 2.2. Heatmap: Average Score (Overall)
print("\n[3/5] Creating heatmap for average score...")

plt.figure(figsize=(12, 6))
heatmap_score = overall_df.pivot(index='day_name', columns='hour', values='avg_score')
heatmap_score = heatmap_score.reindex(day_order)

sns.heatmap(heatmap_score, cmap='coolwarm', annot=False, fmt='.1f', cbar_kws={'label': 'Average Score'})
plt.title('Average Score by Day and Hour', fontsize=16, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Day of Week', fontsize=12)
plt.tight_layout()
plt.savefig('data/plots/EDA_Anna_heatmap_avg_score.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: data/plots/EDA_Anna_heatmap_avg_score.png")

# 3. Bar Chart: Hourly Activity
print("\n[4/5] Creating bar charts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Hourly posting volume
hourly_posts = overall_df.groupby('hour')['num_posts'].sum().reset_index()
axes[0].bar(hourly_posts['hour'], hourly_posts['num_posts'], color='steelblue')
axes[0].set_xlabel('Hour of Day', fontsize=12)
axes[0].set_ylabel('Total Posts', fontsize=12)
axes[0].set_title('Total Posts by Hour', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Hourly average score
hourly_score = overall_df.groupby('hour')['avg_score'].mean().reset_index()
axes[1].bar(hourly_score['hour'], hourly_score['avg_score'], color='coral')
axes[1].set_xlabel('Hour of Day (UTC)', fontsize=12)
axes[1].set_ylabel('Average Score', fontsize=12)
axes[1].set_title('Average Score by Hour', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data/plots/EDA_Anna_bar_hourly_activity.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: data/plots/EDA_Anna_bar_hourly_activity.png")


# 4. Bar Chart: Daily Activity
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Daily posting volume
daily_posts = overall_df.groupby('day_name')['num_posts'].sum().reindex(day_order).reset_index()
axes[0].bar(daily_posts['day_name'], daily_posts['num_posts'], color='teal')
axes[0].set_xlabel('Day of Week', fontsize=12)
axes[0].set_ylabel('Total Posts', fontsize=12)
axes[0].set_title('Total Posts by Day', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Daily average score
daily_score = overall_df.groupby('day_name')['avg_score'].mean().reindex(day_order).reset_index()
axes[1].bar(daily_score['day_name'], daily_score['avg_score'], color='salmon')
axes[1].set_xlabel('Day of Week', fontsize=12)
axes[1].set_ylabel('Average Score', fontsize=12)
axes[1].set_title('Average Score by Day', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data/plots/EDA_Anna_bar_daily_activity.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: data/plots/bar_daily_activity.png")


# 5. Scatter Plot: Volume vs Score
print("\n[5/5] Creating scatter plot...")

plt.figure(figsize=(10, 6))
plt.scatter(overall_df['num_posts'], overall_df['avg_score'], 
            alpha=0.6, s=100, c=overall_df['hour'], cmap='viridis')
plt.colorbar(label='Hour of Day')
plt.xlabel('Number of Posts', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.title('Posting Volume vs Average Score', fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('data/plots/EDA_Anna_scatter_volume_vs_score.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: data/plots/EDA_Anna_scatter_volume_vs_score.png")


# 6. Subreddit Comparison (All Subreddits)
print("\n[6/7] Creating subreddit comparison heatmaps...")

subreddits = subreddit_df['subreddit'].unique()
n_subreddits = len(subreddits)

# Calculate grid size
ncols = 4
nrows = (n_subreddits + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3))
axes = axes.flatten() if n_subreddits > 1 else [axes]

for idx, subreddit in enumerate(subreddits):
    sub_data = subreddit_df[subreddit_df['subreddit'] == subreddit]
    pivot_data = sub_data.pivot(index='day_name', columns='hour', values='num_posts')
    pivot_data = pivot_data.reindex(day_order)
    
    sns.heatmap(pivot_data, cmap='YlOrRd', annot=False, cbar=True, 
                ax=axes[idx], cbar_kws={'label': 'Posts'})
    axes[idx].set_title(f'r/{subreddit}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Hour', fontsize=10)
    axes[idx].set_ylabel('Day', fontsize=10)

# Hide unused subplots
for idx in range(n_subreddits, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Posting Activity by Subreddit', fontsize=18, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('data/plots/EDA_Anna_subreddit_comparison_heatmaps.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: data/plots/EDA_Anna_subreddit_comparison_heatmaps.png")

# 7. Summary Statistics
print("\n[7/7] Creating summary statistics...")

# Top 10 time slots by volume
top_volume = overall_df.nlargest(10, 'num_posts')[['day_name', 'hour', 'num_posts', 'avg_score']]
print("\nTop 10 Time Slots by Posting Volume:")
print(top_volume.to_string(index=False))

# Top 10 time slots by score
top_score = overall_df.nlargest(10, 'avg_score')[['day_name', 'hour', 'num_posts', 'avg_score']]
print("\nTop 10 Time Slots by Average Score:")
print(top_score.to_string(index=False))

# Subreddit summary
sub_summary = subreddit_df.groupby('subreddit').agg({
    'num_posts': 'sum',
    'avg_score': 'mean'
}).sort_values('num_posts', ascending=False).reset_index()
print("\n Subreddit Summary (by total posts):")
print(sub_summary.to_string(index=False))

# Save summary to CSV
top_volume.to_csv('data/csv/EDA_Anna_top_volume_timeslots.csv', index=False)
top_score.to_csv('data/csv/EDA_Anna_top_score_timeslots.csv', index=False)
sub_summary.to_csv('data/csv/EDA_Anna_subreddit_summary.csv', index=False)

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print("\nGenerated plots:")
print("  - data/plots/EDA_Anna_heatmap_posting_volume.png")
print("  - data/plots/EDA_Anna_heatmap_avg_score.png")
print("  - data/plots/EDA_Anna_bar_hourly_activity.png")
print("  - data/plots/EDA_Anna_bar_daily_activity.png")
print("  - data/plots/EDA_Anna_scatter_volume_vs_score.png")
print("  - data/plots/EDA_Anna_subreddit_comparison_heatmaps.png")
print("\nGenerated summaries:")
print("  - data/csv/EDA_Anna_top_volume_timeslots.csv")
print("  - data/csv/EDA_Anna_top_score_timeslots.csv")
print("  - data/csv/EDA_Anna_subreddit_summary.csv")
