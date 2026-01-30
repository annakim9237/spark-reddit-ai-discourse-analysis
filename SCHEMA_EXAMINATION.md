# Reddit Data Schema Examination

This guide explains how to examine the Reddit dataset schema before beginning your analysis.

## Purpose

The `examine_reddit_schema.py` script helps you understand:
- Column names and data types for comments and submissions
- Sample data to see what the content looks like
- Which columns are most useful for analysis

## Prerequisites

- AWS CLI configured with your credentials
- Access to the Reddit dataset in S3

## Step 1: Download Sample Files

Download one sample file each for comments and submissions to examine locally:

```bash
# Download a sample comments file (~108 MB)
aws s3 cp \
  s3://dsan6000-datasets/reddit/parquet/comments/yyyy=2024/mm=01/comments_RC_2024-01.zst_97.parquet \
  /tmp/reddit_comments_sample.parquet \
  --request-payer requester

# Download a sample submissions file (~141 MB)
aws s3 cp \
  s3://dsan6000-datasets/reddit/parquet/submissions/yyyy=2024/mm=01/submissions_RS_2024-01.zst_1.parquet \
  /tmp/reddit_submissions_sample.parquet \
  --request-payer requester
```

**Note:** These downloads will take a few minutes depending on your connection speed.

## Step 2: Run the Schema Examination Script

```bash
# Make sure you're in the project root directory
cd bigdata-project

# Run the examination script
uv run python examine_reddit_schema.py
```

## What You'll See

The script will display:

### 1. Total Rows and Columns
```
Total rows: 1,000,000
Total columns: 17
```

### 2. Complete Schema
```
SCHEMA:
--------------------------------------------------------------------------------
  author                         String
  author_flair_css_class         String
  author_flair_text              String
  body                           String
  controversiality               Int64
  created_utc                    Int64
  distinguished                  String
  edited                         Float64
  gilded                         Int64
  id                             String
  link_id                        String
  parent_id                      String
  retrieved_on                   Int64
  score                          Int64
  stickied                       Boolean
  subreddit                      String
  subreddit_id                   String
```

### 3. Sample Data Rows
Formatted display of the first 3 rows showing actual data.

### 4. Interesting Columns for Analysis
A curated list of columns typically useful for analysis:

**For Comments:**
- `subreddit` - Which subreddit the comment was posted in
- `author` - Username of commenter
- `body` - The comment text content
- `score` - Upvotes minus downvotes
- `created_utc` - Unix timestamp of when comment was created
- `parent_id` - ID of parent comment/submission
- `link_id` - ID of the submission this comment belongs to
- `controversiality` - Whether the comment is controversial (0 or 1)
- `gilded` - Number of Reddit awards received

**For Submissions:**
- `subreddit` - Which subreddit the post was made in
- `author` - Username of poster
- `title` - Post title
- `selftext` - Post body text (for text posts)
- `url` - URL of the post
- `score` - Upvotes minus downvotes
- `created_utc` - Unix timestamp of when post was created
- `num_comments` - Number of comments on the post
- `over_18` - Whether post is marked NSFW

### 5. Detailed Sample Records
The script shows 2 sample records with full field details to help you understand the data format.

## Using This Information

After examining the schema, you can:

1. **Plan your analysis** - Decide which columns you need
2. **Customize the filter script** - Update `COMMENT_COLUMNS` and `SUBMISSION_COLUMNS` in [reddit_data_filter_example.py](spark-cluster/cluster-files/reddit_data_filter_example.py)
3. **Design your research questions** - Based on available fields
4. **Understand data relationships** - How comments link to submissions via `link_id` and `parent_id`


## Cleanup (Optional)

After examining the schema, you can delete the sample files to free up space:

```bash
rm /tmp/reddit_comments_sample.parquet
rm /tmp/reddit_submissions_sample.parquet
```

## Next Steps

1. Review the [reddit-s3-copy-instructions.md](reddit-s3-copy-instructions.md) for copying the full dataset
2. Customize [reddit_data_filter_example.py](spark-cluster/cluster-files/reddit_data_filter_example.py) based on your needs
3. Set up your Spark cluster following [spark-cluster/README.md](spark-cluster/README.md)
4. Begin your analysis!
