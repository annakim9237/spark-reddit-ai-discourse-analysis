# Milestone 1: Exploratory Data Analysis

**Due:** Week 3
**Git Tag:** `v0.1-eda`

## Overview

In this milestone, you will perform comprehensive exploratory data analysis (EDA) on your filtered Reddit dataset. The goal is to understand the characteristics of your data, identify patterns and trends, and answer 3-4 of your EDA business questions.

## Learning Objectives

- Apply distributed computing with Apache Spark for large-scale data analysis
- Perform statistical analysis on hundreds of millions of rows
- Identify temporal patterns and trends
- Create meaningful visualizations
- Derive insights from data exploration

## Deliverables

### 1. Code (`code/eda/` directory)

Create well-organized PySpark scripts for your EDA:
- Data quality assessment and cleaning
- Statistical calculations (counts, averages, distributions)
- Temporal analysis (time-based patterns)
- Grouping and aggregation operations
- Any data transformations needed

**Best practices:**
- Modularize your code (separate scripts for different analyses)
- Use functions to avoid code duplication
- Add clear comments explaining your approach
- Include logging to track progress

### 2. Results (`data/csv/` directory)

Save key statistical results as CSV files:
- Summary statistics by subreddit
- Temporal aggregations (daily/weekly/monthly patterns)
- Top posts/comments by various metrics
- Distribution statistics
- Correlation analysis results

**Naming convention:** Use descriptive names like `eda_temporal_patterns.csv`, `eda_subreddit_statistics.csv`

### 3. Visualizations (`data/plots/` directory)

Create publication-quality visualizations:
- Temporal patterns (time series plots)
- Distribution plots (histograms, box plots)
- Comparison charts (bar charts, scatter plots)
- Heatmaps (correlations, patterns)

**Requirements:**
- Clear titles and axis labels
- Legends when needed
- High resolution (300 DPI for static images)
- Consistent color scheme across plots

**Naming convention:** `eda_[description].png` (e.g., `eda_posting_patterns_by_hour.png`)

### 4. EDA Findings Document

Create a document (markdown or PDF) summarizing:
- Data quality observations
- Key statistical findings
- Patterns and trends discovered
- Answers to 3-4 EDA business questions
- Unexpected findings or anomalies

### 5. Website Page (`website/eda.html`)

Create your EDA page for the website including:
- Overview of your EDA approach
- Key visualizations with interpretations
- Statistical insights
- Clear answers to your EDA business questions
- Implications for NLP and ML stages

## Example EDA Questions

These are generic examples - your questions should be specific to your subreddits and problem statement:

1. **Temporal patterns**: How does posting activity vary by time of day, day of week, and month across your subreddits?
2. **Engagement analysis**: What post characteristics (title length, posting time, flair) correlate with high scores?
3. **Community comparison**: How do engagement patterns differ across your selected subreddits?
4. **Content analysis**: What is the distribution of post types, lengths, and NSFW content?

## Analysis Techniques to Consider

### Statistical Analysis
- Descriptive statistics (mean, median, std, percentiles)
- Distributions (post scores, comment counts, text lengths)
- Correlations between variables
- Outlier detection

### Temporal Analysis
- Posting frequency over time
- Peak activity hours/days
- Seasonal trends
- Growth or decline patterns

### Comparative Analysis
- Subreddit-to-subreddit comparisons
- Comment vs. submission patterns
- Author activity patterns

## Technical Tips

### Spark Performance
```python
# Cache frequently accessed dataframes
df_comments.cache()

# Use partition pruning when possible
df_filtered = df_comments.filter(
    (col("created_utc") >= start_date) &
    (col("created_utc") <= end_date)
)

# Repartition for balanced workload
df = df.repartition(200)
```

### Visualization Best Practices
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style consistently
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Create plot
fig, ax = plt.subplots()
# ... plotting code ...

# Save high quality
plt.savefig('data/plots/eda_plot.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Aggregations Example
```python
from pyspark.sql.functions import hour, dayofweek, count, avg

# Temporal patterns
hourly_activity = df.groupBy(
    hour("created_utc").alias("hour")
).agg(
    count("*").alias("post_count"),
    avg("score").alias("avg_score")
).orderBy("hour")
```

## Common Pitfalls to Avoid

1. **Not caching data**: Cache DataFrames you use multiple times
2. **Too many small files**: Use `coalesce()` before saving results
3. **Unclear visualizations**: Always label axes and add titles
4. **Missing documentation**: Explain what each analysis shows
5. **Not answering questions**: Remember to explicitly answer your business questions
6. **Ignoring data quality**: Address missing values, outliers, anomalies

## Getting Help

- Review the PySpark documentation for aggregation functions
- Check the course discussion board for common issues
- Attend office hours for technical questions
- Review example EDA notebooks in the course materials

## Next Steps

After completing Milestone 1:
- Use EDA insights to inform your NLP analysis (Milestone 2)
- Identify which text fields to focus on for sentiment/topic analysis
- Determine if you need to filter data further for ML
- Begin planning your NLP approach based on data characteristics
