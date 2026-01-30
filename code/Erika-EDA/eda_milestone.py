#!/usr/bin/env python3
"""
Reddit EDA: Title Characteristics vs. Engagement Analysis
Analyzes how post title features (length, question format) relate to engagement
across different AI subreddit categories.

Author: Erika
"""

import sys
import logging
from typing import List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, length, lower, upper, when, count, avg, stddev, sum,
    from_unixtime, to_date, date_format, regexp_extract, size, split,
    percentile_approx, min as spark_min, max as spark_max
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Subreddit categories
CATEGORY_MAPPING = {
    'chatbot_focused': ['ChatGPT', 'OpenAI', 'GPT4', 'ClaudeAI', 'PerplexityAI'],
    'creative_generation': ['StableDiffusion', 'MidJourney', 'Sora', 'AIArt'],
    'research_technical': ['GenerativeAI', 'ArtificialIntelligence', 'MachineLearning', 
                           'computerscience', 'datascience', 'programming'],
    'social_policy': ['Futurology', 'singularity', 'neoliberal'],
    'dev_hardcore': ['LocalLLaMA', 'OpenAI_Dev'],
    'baseline_control': ['AskReddit']
}


def create_spark_session() -> SparkSession:
    """Create and configure Spark session for local analysis."""
    spark = (
        SparkSession.builder
        .appName("Reddit_Title_Engagement_EDA")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    
    logger.info("Spark session created successfully")
    return spark


def load_submissions(spark: SparkSession, data_path: str) -> DataFrame:
    """Load submissions data from local parquet files."""
    logger.info(f"Loading submissions from: {data_path}")
    
    df = spark.read.parquet(data_path)
    
    total_rows = df.count()
    logger.info(f"Loaded {total_rows:,} submissions")
    
    # Show schema
    logger.info("Submissions schema:")
    df.printSchema()
    
    return df


def add_category_column(df: DataFrame) -> DataFrame:
    """Add category column based on subreddit mapping."""
    logger.info("Adding category labels to submissions")
    
    # Create category mapping expression
    category_expr = None
    for category, subreddits in CATEGORY_MAPPING.items():
        condition = col("subreddit").isin(subreddits)
        if category_expr is None:
            category_expr = when(condition, category)
        else:
            category_expr = category_expr.when(condition, category)
    
    df_with_category = df.withColumn("category", category_expr)
    
    # Show category distribution
    logger.info("Category distribution:")
    df_with_category.groupBy("category").count().show()
    
    return df_with_category


def extract_title_features(df: DataFrame) -> DataFrame:
    """Extract features from post titles."""
    logger.info("Extracting title features")
    
    df_features = (
        df
        # Basic length features
        .withColumn("title_length", length(col("title")))
        .withColumn("title_word_count", size(split(col("title"), "\\s+")))
        
        # Question detection
        .withColumn("is_question", 
                    when(col("title").rlike("\\?"), 1).otherwise(0))
        .withColumn("ends_with_question",
                    when(col("title").rlike("\\?$"), 1).otherwise(0))
        
        # Question word patterns (case-insensitive)
        .withColumn("starts_with_how",
                    when(lower(col("title")).rlike("^how "), 1).otherwise(0))
        .withColumn("starts_with_what",
                    when(lower(col("title")).rlike("^what "), 1).otherwise(0))
        .withColumn("starts_with_why",
                    when(lower(col("title")).rlike("^why "), 1).otherwise(0))
        .withColumn("starts_with_when",
                    when(lower(col("title")).rlike("^when "), 1).otherwise(0))
        .withColumn("starts_with_where",
                    when(lower(col("title")).rlike("^where "), 1).otherwise(0))
        .withColumn("starts_with_who",
                    when(lower(col("title")).rlike("^who "), 1).otherwise(0))
        
        # Content indicators
        .withColumn("has_exclamation",
                    when(col("title").rlike("!"), 1).otherwise(0))
        .withColumn("is_all_caps",
                    when(col("title") == upper(col("title")), 1).otherwise(0))
        
        # Add date for temporal analysis
        .withColumn("date", to_date(from_unixtime(col("created_utc"))))
        .withColumn("year_month", date_format(col("date"), "yyyy-MM"))
    )
    
    logger.info("Title features extracted successfully")
    return df_features


def generate_summary_statistics(df: DataFrame) -> pd.DataFrame:
    """Generate summary statistics by category."""
    logger.info("Generating summary statistics by category")
    
    summary = (
        df
        .groupBy("category")
        .agg(
            count("*").alias("total_posts"),
            
            # Title characteristics
            avg("title_length").alias("avg_title_length"),
            stddev("title_length").alias("std_title_length"),
            avg("title_word_count").alias("avg_word_count"),
            
            # Question patterns
            (sum(col("is_question")) / count("*") * 100).alias("question_pct"),
            (sum(col("starts_with_how")) / count("*") * 100).alias("how_pct"),
            (sum(col("starts_with_what")) / count("*") * 100).alias("what_pct"),
            (sum(col("starts_with_why")) / count("*") * 100).alias("why_pct"),
            
            # Engagement metrics
            avg("num_comments").alias("avg_comments"),
            stddev("num_comments").alias("std_comments"),
            percentile_approx("num_comments", 0.5).alias("median_comments"),
            
            avg("score").alias("avg_score"),
            stddev("score").alias("std_score"),
            percentile_approx("score", 0.5).alias("median_score"),
            
            # Engagement extremes
            spark_max("num_comments").alias("max_comments"),
            spark_max("score").alias("max_score")
        )
        .orderBy(col("avg_comments").desc())
    )
    
    df_summary = summary.toPandas()
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS BY CATEGORY")
    print("="*100)
    print(df_summary.to_string(index=False))
    print("="*100 + "\n")
    
    return df_summary


def analyze_title_length_engagement(df: DataFrame, output_dir: str) -> None:
    """Analyze relationship between title length and engagement."""
    logger.info("Analyzing title length vs engagement relationship")
    
    # Sample data for visualization (to avoid memory issues)
    df_sample = (
        df
        .filter(col("category").isNotNull())
        .filter(col("num_comments") < 500)  # Remove extreme outliers
        .filter(col("title_length") < 300)  # Remove extremely long titles
        .sample(fraction=0.3, seed=42)
        .select("category", "title_length", "num_comments", "score", "is_question")
        .toPandas()
    )
    
    logger.info(f"Sampled {len(df_sample):,} posts for visualization")
    
    # Create scatter plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    categories = sorted(df_sample['category'].unique())
    
    # Define colors
    colors = {
        'chatbot_focused': '#E74C3C',
        'creative_generation': '#3498DB',
        'research_technical': '#2ECC71',
        'social_policy': '#F39C12',
        'dev_hardcore': '#9B59B6',
        'baseline_control': '#95A5A6'
    }
    
    for idx, category in enumerate(categories):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        data = df_sample[df_sample['category'] == category]
        
        # Scatter plot
        scatter = ax.scatter(
            data['title_length'],
            data['num_comments'],
            c=data['score'],
            s=30,
            alpha=0.5,
            cmap='YlOrRd',
            edgecolors='black',
            linewidth=0.3
        )
        
        # Add regression line
        if len(data) > 10:
            z = np.polyfit(data['title_length'], data['num_comments'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data['title_length'].min(), 
                                data['title_length'].max(), 100)
            ax.plot(x_line, p(x_line), 
                   color=colors.get(category, '#34495E'),
                   linestyle='--', linewidth=2, alpha=0.8,
                   label=f'Trend (r={np.corrcoef(data["title_length"], data["num_comments"])[0,1]:.2f})')
        
        ax.set_xlabel('Title Length (characters)', fontsize=10)
        ax.set_ylabel('Number of Comments', fontsize=10)
        ax.set_title(category.replace('_', ' ').title(), 
                    fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add colorbar for score
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Post Score', fontsize=8)
    
    # Remove extra subplots
    for idx in range(len(categories), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Title Length vs. Engagement Across AI Subreddit Categories\n(Color intensity = post score)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = f"{output_dir}/title_length_engagement_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved scatter plot to: {output_path}")
    plt.close()


def analyze_question_format_impact(df: DataFrame, output_dir: str) -> None:
    """Analyze impact of question format on engagement."""
    logger.info("Analyzing question format impact on engagement")
    
    # Aggregate statistics
    question_stats = (
        df
        .filter(col("category").isNotNull())
        .groupBy("category", "is_question")
        .agg(
            count("*").alias("post_count"),
            avg("num_comments").alias("avg_comments"),
            avg("score").alias("avg_score")
        )
        .toPandas()
    )
    
    # Pivot for visualization
    pivot_comments = question_stats.pivot(
        index='category',
        columns='is_question',
        values='avg_comments'
    ).fillna(0)
    
    pivot_comments.columns = ['Non-Question', 'Question']
    
    # Create grouped bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Comments comparison
    pivot_comments.plot(kind='bar', ax=ax1, color=['#95A5A6', '#3498DB'], alpha=0.8)
    ax1.set_xlabel('Subreddit Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Comments per Post', fontsize=12, fontweight='bold')
    ax1.set_title('Question Posts vs. Non-Question Posts: Comment Engagement',
                  fontsize=14, fontweight='bold')
    ax1.legend(title='Post Type', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels([x.replace('_', ' ').title() for x in pivot_comments.index],
                        rotation=45, ha='right')
    
    # Score comparison
    pivot_score = question_stats.pivot(
        index='category',
        columns='is_question',
        values='avg_score'
    ).fillna(0)
    pivot_score.columns = ['Non-Question', 'Question']
    
    pivot_score.plot(kind='bar', ax=ax2, color=['#95A5A6', '#E74C3C'], alpha=0.8)
    ax2.set_xlabel('Subreddit Category', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Score per Post', fontsize=12, fontweight='bold')
    ax2.set_title('Question Posts vs. Non-Question Posts: Score',
                  fontsize=14, fontweight='bold')
    ax2.legend(title='Post Type', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticklabels([x.replace('_', ' ').title() for x in pivot_score.index],
                        rotation=45, ha='right')
    
    plt.tight_layout()
    
    output_path = f"{output_dir}/question_format_impact.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved question format analysis to: {output_path}")
    plt.close()


def analyze_optimal_title_length(df: DataFrame, output_dir: str) -> None:
    """Find optimal title length ranges for engagement."""
    logger.info("Analyzing optimal title length ranges")
    
    # Create title length bins
    length_analysis = (
        df
        .filter(col("category").isNotNull())
        .withColumn("length_bin",
                   when(col("title_length") < 50, "Very Short (<50)")
                   .when((col("title_length") >= 50) & (col("title_length") < 100), "Short (50-100)")
                   .when((col("title_length") >= 100) & (col("title_length") < 150), "Medium (100-150)")
                   .when((col("title_length") >= 150) & (col("title_length") < 200), "Long (150-200)")
                   .otherwise("Very Long (200+)"))
        .groupBy("category", "length_bin")
        .agg(
            count("*").alias("post_count"),
            avg("num_comments").alias("avg_comments"),
            avg("score").alias("avg_score")
        )
        .filter(col("post_count") >= 10)  # Minimum sample size
        .toPandas()
    )
    
    # Create heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Comments heatmap
    pivot_comments = length_analysis.pivot(
        index='length_bin',
        columns='category',
        values='avg_comments'
    )
    
    # Reorder rows
    bin_order = ["Very Short (<50)", "Short (50-100)", "Medium (100-150)", 
                 "Long (150-200)", "Very Long (200+)"]
    pivot_comments = pivot_comments.reindex([b for b in bin_order if b in pivot_comments.index])
    
    sns.heatmap(pivot_comments, annot=True, fmt='.1f', cmap='RdYlGn',
                ax=axes[0], cbar_kws={'label': 'Avg Comments'},
                linewidths=1, linecolor='white')
    axes[0].set_xlabel('Subreddit Category', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Title Length Range', fontsize=11, fontweight='bold')
    axes[0].set_title('Average Comments by Title Length',
                     fontsize=13, fontweight='bold')
    axes[0].set_xticklabels([x.get_text().replace('_', ' ').title() 
                             for x in axes[0].get_xticklabels()],
                            rotation=45, ha='right')
    
    # Score heatmap
    pivot_score = length_analysis.pivot(
        index='length_bin',
        columns='category',
        values='avg_score'
    )
    pivot_score = pivot_score.reindex([b for b in bin_order if b in pivot_score.index])
    
    sns.heatmap(pivot_score, annot=True, fmt='.1f', cmap='RdYlGn',
                ax=axes[1], cbar_kws={'label': 'Avg Score'},
                linewidths=1, linecolor='white')
    axes[1].set_xlabel('Subreddit Category', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Title Length Range', fontsize=11, fontweight='bold')
    axes[1].set_title('Average Score by Title Length',
                     fontsize=13, fontweight='bold')
    axes[1].set_xticklabels([x.get_text().replace('_', ' ').title() 
                             for x in axes[1].get_xticklabels()],
                            rotation=45, ha='right')
    
    plt.suptitle('Optimal Title Length for Maximum Engagement',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = f"{output_dir}/optimal_title_length_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved optimal length heatmap to: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("="*100)
    print("REDDIT EDA: TITLE CHARACTERISTICS VS. ENGAGEMENT ANALYSIS")
    print("="*100)
    
    # Configuration
    submissions_path = "data/reddit_output/submissions/"
    output_dir = "output/eda_visualizations"
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Spark
    spark = create_spark_session()
    
    try:
        # Load data
        df_submissions = load_submissions(spark, submissions_path)
        
        # Add categories
        df_with_category = add_category_column(df_submissions)
        
        # Extract title features
        df_features = extract_title_features(df_with_category)
        
        # Cache for multiple operations
        df_features.cache()
        
        # Generate analyses
        summary_stats = generate_summary_statistics(df_features)
        
        analyze_title_length_engagement(df_features, output_dir)
        
        analyze_question_format_impact(df_features, output_dir)
        
        analyze_optimal_title_length(df_features, output_dir)
        
        # Save summary stats
        summary_stats.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
        logger.info(f"Saved summary statistics to: {output_dir}/summary_statistics.csv")
        
        print("\n" + "="*100)
        print("✅ EDA ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📊 Visualizations saved to: {output_dir}/")
        print("="*100)
        
    except Exception as e:
        logger.exception(f"Error during analysis: {str(e)}")
        return 1
    
    finally:
        spark.stop()
        logger.info("Spark session stopped")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())