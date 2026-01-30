# Lab: Spark Cluster Setup on AWS EC2

Step-by-step guide for setting up a 3-node Apache Spark cluster on AWS EC2.

## Overview

This lab provides hands-on experience with setting up and managing a distributed Apache Spark cluster on Amazon EC2. You will learn how to create, configure, and use a Spark cluster through three progressive problems:

1. **Manual Setup** - Build cluster step-by-step to understand all components
2. **Automated Setup** - Use automation scripts for rapid deployment
3. **Data Analysis** - Run real-world data processing jobs on your cluster

## What You'll Learn

By completing this lab, you will gain practical experience with:
- Creating and configuring EC2 instances using AWS CLI
- Setting up network security groups for cluster communication
- Configuring SSH keys and passwordless access between nodes
- Installing and configuring Apache Spark in distributed mode
- Monitoring Spark jobs via Web UIs
- Running PySpark jobs on a distributed cluster
- Processing large-scale data stored in Amazon S3
- Proper cluster resource cleanup and cost management

## Prerequisites

- AWS Account with appropriate IAM permissions
- Access to an EC2 instance (Linux) to run the setup commands
- Basic familiarity with terminal/command line
- AWS CLI configured with credentials
- Your laptop's public IP address (get from https://ipchicken.com/)

---

## 🟢 IMPORTANT: Critical Skills for Your Project

**This lab is probably the most important one for your project.** The skills you learn here - setting up a cluster, running code both locally and on a distributed cluster, cleaning up resources, viewing metrics in the Spark UI - are **critical** for using Spark in your final project.

Make sure you are **very comfortable** with all these operations and can easily do them repeatedly. In fact, it might be a good idea to **repeat this lab once you have completed and submitted it** so that you become super conversant with:
- Setting up a cluster from scratch
- Running code in distributed mode
- Monitoring jobs through the Spark Web UIs
- Troubleshooting issues
- Cleaning up resources properly

The step-by-step guide in [SPARK_CLUSTER_SETUP.md](SPARK_CLUSTER_SETUP.md) gives you insight into what actually goes on when setting up a distributed cluster. All of these are **important foundational concepts** that apply to having multiple machines communicate and work together. While the implementation is AWS-specific, **these concepts are cloud-agnostic** and apply to all cloud providers (AWS, Azure, GCP, etc.).

Understanding these fundamentals will make you proficient in distributed computing, regardless of which cloud platform you use in your career.

---

## Three-Problem Learning Approach

### But First... A Review of Last Week

Before diving into the cluster setup, let's review last week's NYC TLC problem - but this time in a Jupyter notebook. Sometimes it's easier to work with and experiment with code in a notebook environment rather than running standalone Python scripts.

**Task:** Run the NYC TLC Problem 1 analysis in a Jupyter notebook with Spark running locally.

All code has been provided for you in [`cluster-files/nyc_tlc_problem1.ipynb`](cluster-files/nyc_tlc_problem1.ipynb). Just open the notebook and run all the cells to confirm that Spark works correctly on your EC2 instance.

---

## Problem 1: Manual Cluster Setup (Learning the Fundamentals) - 20 Points

**Objective:** Gain hands-on understanding of every component involved in creating a Spark cluster.

Follow the detailed instructions in [SPARK_CLUSTER_SETUP.md](SPARK_CLUSTER_SETUP.md) to:
- Manually create EC2 instances
- Configure security groups and network rules
- Set up SSH keys and passwordless authentication
- Install Java, Python, and Spark on each node
- Configure master and worker nodes
- Start the Spark cluster
- Verify cluster operation through Web UIs

**Deliverables:**
- Working 3-node Spark cluster (1 master + 2 workers)
- Screenshots of:
  - Spark Master Web UI showing connected workers (placeholder - to be provided)
  - Spark Application UI showing a running job (placeholder - to be provided)
- Download result files from master node to your EC2 instance:
  ```bash
  scp -i spark-cluster-key.pem ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/problem1_cluster.txt .
  ```
- Commit and push the following files:
  - `cluster-config.txt`
  - `cluster-ips.txt`
  - `problem1_cluster.txt` (cluster run output)

**Cleanup:** After completing Step 1, manually delete resources in this order:
1. Terminate EC2 instances
2. Delete key pair
3. Delete security group

**⚠️ WARNING: If you do not delete your cluster, you will exhaust the funds in your account. Always delete your cluster when done!**

## Problem 2: Automated Cluster Setup (Scripted Deployment) - 20 Points

**Objective:** Learn how to automate cluster deployment for rapid provisioning.

Follow the instructions in [AUTOMATION_README.md](AUTOMATION_README.md) to:
- Run the automated setup script `setup-spark-cluster.sh`
- Create a 4-node cluster (1 master + 3 workers) automatically
- Understand the automation workflow
- Submit a Spark job to the cluster
- Use the automated cleanup script

**Key Script:**
```bash
./setup-spark-cluster.sh <YOUR_LAPTOP_IP>
```

This script automates everything from Step 1, including:
- Security group creation
- Key pair generation
- EC2 instance provisioning
- Software installation on all nodes
- Spark configuration
- Cluster startup

**Deliverables:**
- Download result files from master node to your EC2 instance:
  ```bash
  # Use the SSH key from the setup (check cluster-config.txt for KEY_FILE name)
  source cluster-config.txt
  scp -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/*.txt .
  ```
- Commit and push the following files:
  - `cluster-config.txt`
  - `cluster-ips.txt`
  - `ssh_to_master_node.sh`
  - Output files from running the NYC TLC job (downloaded from master node)

**Cleanup:** Use the automated cleanup script:
```bash
./cleanup-spark-cluster.sh
```

Or manually delete resources as in Problem 1.

**⚠️ WARNING: If you do not delete your cluster, you will exhaust the funds in your account. Always delete your cluster when done! Use [cleanup-spark-cluster.sh](cleanup-spark-cluster.sh) for automated cleanup or manual deletion.**

## Problem 3: Real-World Data Analysis (Reddit Data Processing) - 20 Points

**Objective:** Apply your Spark cluster skills to analyze real-world social media data.

#### Setup

1. **Install Java (required for local PySpark development):**
```bash
# Install Java 17 (required for PySpark 4.x)
sudo apt update
sudo apt install -y openjdk-17-jdk-headless

# Verify installation
java -version

# Set JAVA_HOME (add to ~/.bashrc for persistence)
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin
```

2. **Download a sample file for local development:**
```bash
# Download one file to test locally first (saves to current directory)
aws s3 cp s3://dsan6000-datasets/reddit/parquet/comments/yyyy=2024/mm=01/comments_RC_2024-01.zst_97.parquet \
  reddit_sample.parquet \
  --request-payer requester
```

3. **Create S3 bucket for your data (after local testing works):**
```bash
aws s3 mb s3://your-netid-spark-reddit
```

4. **Copy full Reddit data from shared dataset (after local testing works):**
```bash
aws s3 cp s3://dsan6000-datasets/reddit/parquet/comments/yyyy=2024/mm=01/ \
  s3://your-netid-spark-reddit/reddit/comments/yyyy=2024/mm=01/ \
  --recursive \
  --request-payer requester \
  --metadata-directive REPLACE
```

#### Problem 3: Reddit Subreddit Analysis

**Dataset:** Reddit comments from January 2024 (Parquet format)
- Source: `s3://dsan6000-datasets/reddit/parquet/comments/yyyy=2024/mm=01/` (use `--request-payer requester` to access)
- Format: Parquet files with columns including `subreddit`, `body`, `author`, `created_utc`, etc.

**Task:** Analyze the Reddit comments data to answer the following questions:

1. **Dataset Statistics:** Count the total number of unique comments and unique users (authors) in the dataset
2. **Most Popular Subreddits:** Find the top 10 most active subreddits by comment count
3. **Temporal Analysis:** Identify peak commenting hours (UTC) across all subreddits

**Development Workflow:**

1. **Local Development (Recommended Approach):**
   - Download one sample file: `reddit_sample.parquet` (in current directory)
   - Create `reddit_analysis_local.py` that works with local file (save in repo root)
   - Use local Spark session (similar to the lab exercises):
     ```python
     spark = SparkSession.builder \
         .appName("Reddit_Local") \
         .getOrCreate()

     df = spark.read.parquet("reddit_sample.parquet")
     ```
   - Test your analysis logic on the sample file
   - Verify outputs are correct (see Deliverables section below for exact file names)

2. **Cluster Deployment:**
   - Once local version works, create `reddit_analysis_cluster.py`
   - Modify the code to use cluster master URL (see [nyc_tlc_problem1_cluster.py](cluster-files/nyc_tlc_problem1_cluster.py) for reference):
     - Change `.getOrCreate()` to `.master(master_url).getOrCreate()`
     - Change file path from local to S3A: `s3a://your-netid-spark-reddit/...`
     - Add S3A configuration (copy from NYC TLC example)
     - Accept master URL from command line argument
   - Copy full dataset to your S3 bucket
   - Copy your script to the cluster master node:
     ```bash
     # Load cluster configuration
     source cluster-config.txt

     # Copy your cluster script to master node
     scp -i $KEY_FILE reddit_analysis_cluster.py ubuntu@$MASTER_PUBLIC_IP:~/
     ```
   - SSH to master and run on your Spark cluster:
     ```bash
     ssh -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP
     source cluster-ips.txt
     uv run python reddit_analysis_cluster.py spark://$MASTER_PRIVATE_IP:7077
     ```

   **⏱️ IMPORTANT - Expected Runtime:**
   This job processes ~280 million Reddit comments and is expected to take **10-15 minutes** to complete when running correctly on the cluster. This is normal! While the job is running:
   - Monitor progress in the **Spark Master UI** at `http://$MASTER_PUBLIC_IP:8080` to see worker activity
   - Watch the **Spark Application UI** at `http://$MASTER_PUBLIC_IP:4040` to see the job DAG, stages, and tasks
   - You'll see three separate analyses run sequentially (dataset stats, top subreddits, peak hours)
   - Each analysis will read the data from S3 and perform distributed computations across your worker nodes

   Don't worry if it seems slow - you're processing hundreds of millions of records! Use the Web UIs to understand what Spark is doing behind the scenes.

   - Download results from cluster:
     ```bash
     # From your EC2 instance (not on the cluster)
     source cluster-config.txt

     # Download the cluster output CSV files (already have _cluster suffix)
     scp -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/dataset_stats_cluster.csv .
     scp -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/top_subreddits_cluster.csv .
     scp -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/peak_hours_cluster.csv .
     ```

**Deliverables:**
- `reddit_analysis_local.py` (local development version)
- `reddit_analysis_cluster.py` (cluster version)
- Local output CSV files:
  - `dataset_stats_local.csv` (Unique comments and users)
  - `top_subreddits_local.csv` (Top 10 most popular subreddits)
  - `peak_hours_local.csv` (Hourly comment distribution)
- Cluster output CSV files:
  - `dataset_stats_cluster.csv` (Unique comments and users)
  - `top_subreddits_cluster.csv` (Top 10 most popular subreddits)
  - `peak_hours_cluster.csv` (Hourly comment distribution)
- Screenshots of:
  - Spark Web UI showing the job execution
  - Application UI showing the DAG (Directed Acyclic Graph)

**Hints:**
- Use `spark.read.parquet()` to load data
- Count unique values using `df.select("column").distinct().count()` or `countDistinct("column")`
- Extract hour from `created_utc` using `hour(from_unixtime(col("created_utc")))`
- Use `groupBy()` and `agg()` for aggregations
- Use `orderBy(desc("count"))` for sorting
- Save results using `.toPandas().to_csv()` or `df.write.csv()`

**Expected Warning (Normal Behavior):**
When reading parquet files from S3, you may see a `FileNotFoundException` warning about streaming metadata:
```
WARN FileStreamSink: Assume no metadata directory. Error while looking for metadata directory in the path: s3a://...
java.io.FileNotFoundException: No such file or directory: s3a://your-netid-spark-reddit/...
```
This is **normal and harmless**. Spark checks if the path contains streaming metadata (used for Spark Structured Streaming), and when it doesn't find any, it proceeds to read the files as regular batch data. Your job will continue successfully despite this warning.

**⚠️ WARNING: If you do not delete your cluster, you will exhaust the funds in your account. Always delete your cluster when done! Use [cleanup-spark-cluster.sh](cleanup-spark-cluster.sh) for automated cleanup.**

## Cost Estimate

**Important:** Running this cluster will incur AWS costs.

- **Instance type:** t3.large
- **Number of instances:**
  - Problem 1: 3 instances (1 master + 2 workers) ≈ $0.20/hour
  - Problem 2: 4 instances (1 master + 3 workers) ≈ $0.25/hour
  - Problem 3: 4 instances + S3 storage ≈ $0.25/hour + minimal storage costs
- **Estimated total cost:** $6-8 for completing all three problems

**Always remember to terminate your instances and delete S3 data when done to avoid unnecessary charges.**

## Repository Structure

```
lab-spark-cluster/
├── README.md                      # This file - overview and 3-problem guide
├── SPARK_CLUSTER_SETUP.md         # Problem 1: Manual setup instructions
├── AUTOMATION_README.md           # Problem 2: Automated setup guide
├── setup-spark-cluster.sh         # Automated cluster creation script
├── cleanup-spark-cluster.sh       # Automated cluster cleanup script
├── reddit_analysis_local.py       # Problem 3: Local analysis script
├── reddit_analysis_cluster.py     # Problem 3: Cluster analysis script
├── cluster-files/                 # Configuration files for cluster nodes
│   └── nyc_tlc_problem1_cluster.py
├── pyproject.toml                 # Python dependencies
└── .gitignore                     # Git ignore file (includes *.pem, *.parquet)
```

## Getting Started

1. **Clone this repository** (if in a git environment) or navigate to the project directory:
```bash
cd /home/ubuntu/lab-spark-cluster
```

2. **Start with Step 1** - Follow [SPARK_CLUSTER_SETUP.md](SPARK_CLUSTER_SETUP.md) for manual setup

3. **Progress to Step 2** - Follow [AUTOMATION_README.md](AUTOMATION_README.md) for automated setup

4. **Complete Step 3** - Work on the Reddit data analysis problem

## Support

If you encounter issues:
1. Check the Troubleshooting sections in the respective guides
2. Verify all environment variables are set correctly
3. Check the Spark logs on master and worker nodes: `$SPARK_HOME/logs/`
4. Ensure security group rules are properly configured
5. Verify AWS credentials and S3 bucket permissions

## Important Security Notes

- **Never commit `.pem` files** - They are in `.gitignore`
- **Limit security group access** - Only allow your IP addresses
- **Use IAM roles** - Instances use `LabInstanceProfile` for S3 access
- **Delete resources** - Always clean up when done

## Related Resources

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [AWS EC2 User Guide](https://docs.aws.amazon.com/ec2/)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)

## License

MIT License
