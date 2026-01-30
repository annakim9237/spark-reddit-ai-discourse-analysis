# Reddit Dataset S3 Copy Instructions

*Created: 2025-10-15*

## Overview
This document contains instructions and information for copying the Reddit parquet dataset from the source bucket to your own S3 bucket.

## Source Dataset Information

**Source Bucket:** `s3://dsan6000-datasets/reddit/parquet/`

### Folder Structure
```
s3://dsan6000-datasets/reddit/parquet/
├── comments/
│   ├── yyyy=2023/
│   │   ├── mm=06/  (June 2023)
│   │   ├── mm=07/  (July 2023)
│   │   ├── mm=08/  (August 2023)
│   │   ├── mm=09/  (September 2023)
│   │   ├── mm=10/  (October 2023)
│   │   ├── mm=11/  (November 2023)
│   │   └── mm=12/  (December 2023)
│   └── yyyy=2024/
│       ├── mm=01/  (January 2024)
│       ├── mm=02/  (February 2024)
│       ├── mm=03/  (March 2024)
│       ├── mm=04/  (April 2024)
│       ├── mm=05/  (May 2024)
│       ├── mm=06/  (June 2024)
│       └── mm=07/  (July 2024)
└── submissions/
    ├── yyyy=2023/
    │   ├── mm=06/  (June 2023)
    │   ├── mm=07/  (July 2023)
    │   ├── mm=08/  (August 2023)
    │   ├── mm=09/  (September 2023)
    │   ├── mm=10/  (October 2023)
    │   ├── mm=11/  (November 2023)
    │   └── mm=12/  (December 2023)
    └── yyyy=2024/
        ├── mm=01/  (January 2024)
        ├── mm=02/  (February 2024)
        ├── mm=03/  (March 2024)
        ├── mm=04/  (April 2024)
        ├── mm=05/  (May 2024)
        ├── mm=06/  (June 2024)
        └── mm=07/  (July 2024)
```

### Dataset Statistics

#### Comments Directory
- **Path:** `s3://dsan6000-datasets/reddit/parquet/comments/`
- **Total Files:** 3,680 parquet files
- **Total Size:** 369.6 GiB
- **File Size Range:** ~95-107 MiB per file
- **Partitioning:** Hive-style partitioning by year/month (yyyy=YYYY/mm=MM)

#### Submissions Directory
- **Path:** `s3://dsan6000-datasets/reddit/parquet/submissions/`
- **Total Files:** 575 parquet files
- **Total Size:** 76.3 GiB
- **File Size Range:** ~95-107 MiB per file
- **Partitioning:** Hive-style partitioning by year/month (yyyy=YYYY/mm=MM)

#### Combined Totals
- **Total Files:** 4,255 parquet files
- **Total Size:** ~445.9 GiB (369.6 + 76.3)
- **Time Period:** June 2023 - July 2024 (14 months)
  - **2023:** June - December (7 months)
  - **2024:** January - July (7 months)
- **Data Source:** Reddit comments and submissions archives

## Copy Instructions

### Prerequisites
1. AWS CLI configured with appropriate credentials
2. Sufficient IAM permissions for S3 operations
3. Note: Source bucket uses requester-pays pricing

### Create Destination Bucket First (Optional)

```bash
# Create the destination bucket if it doesn't exist
NET_ID="ms4821"
aws s3 mb s3://${NET_ID}-dsan6000-datasets --region us-east-1
```

### Basic Copy Command

Replace `your-net-id` with your actual net-id:

```bash
# Set your net-id
NET_ID="your-net-id"

# Copy the entire dataset
aws s3 sync \
  s3://dsan6000-datasets/reddit/parquet/ \
  s3://${NET_ID}-dsan6000-datasets/reddit/parquet/ \
  --request-payer requester \
  --region us-east-1
```

## Estimated Transfer Time

- **Same Region (US-EAST-1 to US-EAST-1):** 30 minutes to 2 hours
- **Cross Region:** 2-4 hours
- **Depends on:** Network conditions, parallelization settings, AWS infrastructure load

## Cost Estimate

### Same Region Copy (e.g., US-EAST-1 to US-EAST-1)
- **Data Transfer:** $0 (no charge within same region)
- **GET Requests:** ~$0.002 (4,255 requests × $0.0004 per 1,000)
- **PUT Requests:** ~$0.02 (4,255 requests × $0.005 per 1,000)
- **Requester-Pays Charges:** Minimal (you pay for GET requests as requester)
- **Total One-Time Cost:** ~$0.02 (essentially free)

### Storage Cost (Ongoing)
- **Standard S3 Storage:** 446 GiB × $0.023/GB/month = ~$10.26/month
- **Intelligent-Tiering:** Could reduce costs if data is infrequently accessed

### Cross-Region Copy (e.g., US-EAST-1 to US-WEST-2)
- **Data Transfer OUT:** 446 GiB × $0.02/GB = ~$9.12
- **Request Costs:** ~$0.02
- **Total Cross-Region Cost:** ~$9.14

## Monitoring Progress

The `aws s3 sync` command will display progress as it copies:

```
Completed 256.0 KiB/~446.0 GiB (12.5 MiB/s) with 1 file(s) remaining
```

## Verify Copy Completion

After the copy completes, verify the file count and size:

```bash
# Check comments
aws s3 ls s3://${NET_ID}-dsan6000-datasets/reddit/parquet/comments/ \
  --recursive --summarize --human-readable | tail -3

# Check submissions
aws s3 ls s3://${NET_ID}-dsan6000-datasets/reddit/parquet/submissions/ \
  --recursive --summarize --human-readable | tail -3
```

Expected output:
- Comments: 3,680 objects, 369.6 GiB
- Submissions: 575 objects, 76.3 GiB

## Data Format Information

- **Format:** Apache Parquet (columnar storage)
- **Compression:** Parquet files are already compressed
- **Schema:** Reddit API schema for comments and submissions
- **Partitioning:** Hive-style partitioning enables efficient filtering by year/month in Spark/Hive

## Next Steps After Copying Data

### 1. Set Up Your Spark Cluster

Follow instructions in [spark-cluster/README.md](spark-cluster/README.md) to:
- Launch master and worker nodes
- Configure S3 access
- Test cluster connectivity

### 2. Filter Data to Subreddits of Interest

The full dataset is ~446 GB. For most analyses, you'll want to filter to specific subreddits. Use the provided example script:

```bash
# On your Spark cluster master node
cd spark-cluster/cluster-files
python reddit_data_filter_example.py <your-net-id> spark://<master-ip>:7077
```

This script will:
- Read the full Reddit dataset from your S3 bucket
- Filter by subreddits defined in `EXAMPLE_SUBREDDITS`
- Select commonly useful columns
- Save filtered data to a new S3 prefix for your project

**Output locations:**
- `s3://<your-net-id>-dsan6000-datasets/project/reddit/parquet/comments/`
- `s3://<your-net-id>-dsan6000-datasets/project/reddit/parquet/submissions/`

### 3. Customize the Filtering

Edit [reddit_data_filter_example.py](spark-cluster/cluster-files/reddit_data_filter_example.py):

```python
# Change these subreddits to match your research questions
EXAMPLE_SUBREDDITS: List[str] = [
    "datascience",
    "MachineLearning",
    "artificial",
    # Add your subreddits here
]
```

**Useful columns included by default:**

**Comments:**
- `id`, `subreddit`, `author`, `body`, `score`
- `created_utc`, `parent_id`, `link_id`
- `controversiality`, `gilded`

**Submissions:**
- `id`, `subreddit`, `author`, `title`, `selftext`
- `score`, `created_utc`, `num_comments`, `url`, `over_18`

### 4. Start Your Analysis

Once you have filtered data:
- Begin EDA (Exploratory Data Analysis)
- Perform NLP tasks (sentiment analysis, topic modeling)
- Build ML models
- Create visualizations for your website

## Example Spark Read Commands

### Reading Full Dataset

```python
# Read all comments (full dataset)
df_comments = spark.read.parquet(
    f"s3a://{net_id}-dsan6000-datasets/reddit/parquet/comments/"
)

# Read all submissions (full dataset)
df_submissions = spark.read.parquet(
    f"s3a://{net_id}-dsan6000-datasets/reddit/parquet/submissions/"
)
```

### Reading Filtered Project Data

```python
# Read your filtered comments
df_comments_filtered = spark.read.parquet(
    f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/comments/"
)

# Read your filtered submissions
df_submissions_filtered = spark.read.parquet(
    f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/submissions/"
)
```

### Reading Specific Time Periods

```python
# Filter by specific month from full dataset
df_june_comments = spark.read.parquet(
    f"s3a://{net_id}-dsan6000-datasets/reddit/parquet/comments/yyyy=2023/mm=06/"
)

# Filter by year
df_2024_submissions = spark.read.parquet(
    f"s3a://{net_id}-dsan6000-datasets/reddit/parquet/submissions/yyyy=2024/"
)
```

### Filtering After Reading

```python
from pyspark.sql.functions import col, from_unixtime, to_date

# Read filtered data
df = spark.read.parquet(
    f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/comments/"
)

# Add date column for easier filtering
df = df.withColumn("date", to_date(from_unixtime(col("created_utc"))))

# Filter by date range
df_filtered = df.filter(
    (col("date") >= "2023-06-01") & (col("date") <= "2023-12-31")
)

# Filter by score threshold
df_popular = df.filter(col("score") >= 10)

# Filter by subreddit (if you need subset of your filtered data)
df_ml_only = df.filter(col("subreddit") == "MachineLearning")
```

## Data Schema

### Comments Schema (17 columns)

```
author                         String
author_flair_css_class         String
author_flair_text              String
body                           String      # Comment text content
controversiality               Int64       # 0 or 1
created_utc                    Int64       # Unix timestamp
distinguished                  String
edited                         Float64
gilded                         Int64       # Number of gildings
id                             String      # Unique comment ID
link_id                        String      # Parent submission ID
parent_id                      String      # Parent comment/submission ID
retrieved_on                   Int64
score                          Int64       # Upvotes - downvotes
stickied                       Boolean
subreddit                      String      # Subreddit name
subreddit_id                   String
```

### Submissions Schema (21 columns)

```
author                         String
author_flair_css_class         String
author_flair_text              String
created_utc                    Int64       # Unix timestamp
distinguished                  String
domain                         String
edited                         Float64
id                             String      # Unique submission ID
is_self                        Boolean     # Is text post
locked                         Boolean
num_comments                   Int64       # Comment count
over_18                        Boolean     # NSFW flag
quarantine                     Boolean
retrieved_on                   Int64
score                          Int64       # Upvotes - downvotes
selftext                       String      # Post body text
stickied                       Boolean
subreddit                      String      # Subreddit name
subreddit_id                   String
title                          String      # Post title
url                            String      # Post URL
```

## Notes

- The source bucket uses requester-pays, so ensure you include `--request-payer requester`
- AWS CLI automatically handles multipart uploads for large files
- The `sync` command only copies new/changed files (safe to re-run)
- Data covers 14 months of Reddit activity (June 2023 - July 2024)
- Both comments and submissions maintain Reddit's original data structure
- Use partition filtering (yyyy=YYYY/mm=MM) for efficient queries on time ranges
- Filter to specific subreddits early to reduce data size and processing time
