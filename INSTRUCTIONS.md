# DSAN 6000 Project

## Overview

This is an end-semester group project for DSAN 6000 focused on **big data analysis at scale**. The core objective is to work with a large dataset containing **at least a few hundred million rows** and apply big data processing techniques to answer interesting, relevant business questions that can **only be answered by examining large amounts of data**.

### Project Requirements

**Dataset Scale:**
- Minimum: Several hundred million rows
- Recommended: Multi-GB to hundreds of GB
- The dataset must be large enough that traditional tools (pandas on a single machine) would struggle or fail

**Big Data Technologies:**
- **Primary:** Apache Spark on an EC2 cluster (distributed computing)
- **Optional:** AWS Athena for SQL-based queries, multiprocessing with Polars, or other distributed frameworks
- The key is demonstrating techniques that scale beyond single-machine processing

**Analysis Components:**

1. **Exploratory Data Analysis (EDA)** - Answer questions through statistical analysis, temporal patterns, and visualizations
2. **Natural Language Processing (NLP)** - Apply text mining techniques like sentiment analysis, topic modeling, and entity extraction (if your dataset has text)
3. **Machine Learning (ML)** - Train and evaluate predictive models to answer questions requiring classification, regression, or clustering
4. **Comprehensive Reporting** - Present findings in a coherent, well-structured website with clear narratives

**Project Approach:**

Start by defining a **high-level problem statement** relevant to your dataset, then break it down into **10 specific, answerable questions** across EDA, NLP, and ML. This approach ensures your analysis is focused and comprehensive.

### Example: NYC Taxi Dataset

**High-Level Problem:** *"How can we optimize NYC taxi operations to maximize revenue and improve customer satisfaction?"*

This broad problem can be broken down into specific questions at different complexity levels:

**EDA Questions (Statistical & Temporal Analysis):**
- How do taxi demand patterns vary by time of day, day of week, and season across different NYC boroughs?
- What is the relationship between trip distance and fare amount, and how do outliers affect pricing?
- Which pickup/dropoff location pairs generate the highest revenue, and how has this changed over time?
- How do tip percentages vary by payment method, and what does this reveal about passenger behavior?
- Can we identify temporal patterns in surge pricing or high-demand periods?

**NLP Questions (if dataset included text feedback):**
- What are the most common topics discussed in passenger/driver feedback?
- How does sentiment in feedback correlate with tip amounts or ratings?
- What service issues are most frequently mentioned during rush hours vs. off-peak times?
- Can we identify emerging complaints or trends in service quality over time?

**ML Questions (Predictive Modeling):**
- Can we predict high-tip trips based on trip characteristics (time, location, distance)?
- What factors best predict trip duration, and can we build a model more accurate than GPS estimates?
- Can we classify trips into customer segments (commuters, tourists, business travelers) based on patterns?
- Can we predict demand at specific locations to optimize taxi fleet deployment?

### Deliverables

Your team will produce:
- **Clean, well-documented Spark code** demonstrating big data techniques
- **Comprehensive analysis** spanning EDA, NLP (if applicable), and ML
- **Professional website** presenting findings, visualizations, and insights
- **Final report** with clear business recommendations based on your analysis

## Important: All Work Done on Spark Cluster

**ALL analysis and data processing for this project should be done on your Apache Spark cluster**. The Spark cluster setup instructions are provided in [spark-cluster/README.md](spark-cluster/README.md).

### Setting Up Your Spark Cluster

Before beginning your analysis, you must:

1. Set up your Apache Spark cluster following the instructions in [spark-cluster/README.md](spark-cluster/README.md)
2. Configure S3 access for reading and writing data
3. Test your cluster with the provided example scripts

All the infrastructure setup instructions, including master/worker configuration and automated deployment, are documented in the [spark-cluster/](spark-cluster/) directory.

## Dataset

### Recommended: Reddit Dataset

This repository provides starter code and instructions for working with the **Reddit dataset** (comments and submissions from June 2023 - July 2024). This is a large-scale dataset (~446 GB) ideal for demonstrating big data skills.

**Getting the Reddit Data:**
- **First**, examine the schema: See [SCHEMA_EXAMINATION.md](SCHEMA_EXAMINATION.md) to understand the data structure
- **Then**, copy the data: See [reddit-s3-copy-instructions.md](reddit-s3-copy-instructions.md) for complete instructions
- **Finally**, filter the data: Use [spark-cluster/cluster-files/reddit_data_filter_example.py](spark-cluster/cluster-files/reddit_data_filter_example.py)

### Using Your Own Dataset

If you prefer to use a different dataset, you must:
1. Get professor approval for your chosen dataset
2. Ensure it's large enough to demonstrate big data techniques
3. Figure out data acquisition and preprocessing on your own (the Reddit instructions won't apply)

**Where to find large datasets:**
- [AWS Open Data Registry](https://registry.opendata.aws/) - Datasets already on S3
- [Google Dataset Search](https://datasetsearch.research.google.com/)
- [Common Crawl](https://commoncrawl.org/) - Web crawl data
- [GitHub Archive](https://www.gharchive.org/) - GitHub activity data


**Examples of suitable large-scale datasets:**
- Wikipedia dumps
- News article archives
- Government data (census, weather, transportation)
- Scientific datasets (genomics, astronomy, climate)
- E-commerce transaction data
- IoT sensor data

## Project Workflow

### Milestone 0: Data Acquisition and Initial Filtering

**For Reddit dataset users:**

1. **Copy the full Reddit dataset to your S3 bucket** (see [reddit-s3-copy-instructions.md](reddit-s3-copy-instructions.md)):
   ```bash
   NET_ID="your-net-id"
   aws s3 sync \
     s3://dsan6000-datasets/reddit/parquet/ \
     s3://${NET_ID}-dsan6000-datasets/reddit/parquet/ \
     --request-payer requester
   ```

2. **Filter data to your subreddits of interest** using the provided example:
   ```bash
   # On your Spark cluster
   python reddit_data_filter_example.py <your-net-id> spark://<master-ip>:7077
   ```

   This will create filtered datasets at:
   - `s3://<your-net-id>-dsan6000-datasets/project/reddit/parquet/comments/`
   - `s3://<your-net-id>-dsan6000-datasets/project/reddit/parquet/submissions/`

3. **Customize the filtering**:
   - Edit `EXAMPLE_SUBREDDITS` in [reddit_data_filter_example.py](spark-cluster/cluster-files/reddit_data_filter_example.py)
   - Add date range filters if needed
   - Select additional columns relevant to your analysis

**Expected Output:**

Your filtered dataset **for your chosen subreddits of interest** should contain **at least a few hundred million rows** across comments and submissions. This demonstrates you are working with data at a scale that requires big data techniques.

**Required Documentation:**

1. **Dataset Statistics (save to `data/csv/` folder):**

   Create the following CSV files documenting your filtered dataset **for your chosen subreddits of interest**:

   - **`data/csv/dataset_summary.csv`** - Overall statistics
     - Columns: `data_type` (comments/submissions), `total_rows`, `size_gb`, `date_range_start`, `date_range_end`

   - **`data/csv/subreddit_statistics.csv`** - Per-subreddit breakdown
     - Columns: `subreddit`, `num_comments`, `num_submissions`, `total_rows`, `avg_score`, `date_range`

   - **`data/csv/temporal_distribution.csv`** - Data distribution over time
     - Columns: `year_month`, `num_comments`, `num_submissions`, `total_rows`

   These CSV files should be generated using Spark aggregations on your filtered data and will be included in your project report to demonstrate the scale and characteristics of your dataset.

2. **Business Questions (save as `BUSINESS_QUESTIONS.md` in project root):**

   Define **10 business questions** that you will answer using your dataset. For each question, provide:
   - The business question (clear and specific)
   - The technical approach (3-5 lines describing the methodology)
   - The analysis type (EDA, NLP, or ML)

   **See [BUSINESS_QUESTIONS_TEMPLATE.md](BUSINESS_QUESTIONS_TEMPLATE.md) for a complete example with 10 sample questions.**

   **Distribution guideline:**
   - 3-4 questions answered through EDA (statistical analysis, visualizations, temporal patterns)
   - 3-4 questions requiring NLP (sentiment analysis, topic modeling, text mining)
   - 2-3 questions requiring ML (classification, regression, clustering, prediction)

   This document serves as your project roadmap and will guide your analysis through all milestones.

### Milestone 1: Exploratory Data Analysis (EDA)

Use Spark to perform comprehensive EDA:
- Data quality assessment
- Descriptive statistics
- Temporal patterns
- Subreddit/topic distributions
- Initial visualizations

**Git Tag:** `milestone-1-eda`
**Detailed Instructions:** [EDA.md](EDA.md)

### Milestone 2: Natural Language Processing (NLP)

Apply NLP techniques to text data:
- Text preprocessing and cleaning
- Sentiment analysis
- Topic modeling
- Entity extraction
- Feature engineering for ML

**Git Tag:** `milestone-2-nlp`
**Detailed Instructions:** [NLP.md](NLP.md)

### Milestone 3: Machine Learning (ML)

Build and evaluate ML models:
- Classification/regression tasks
- Model training with Spark MLlib
- Hyperparameter tuning
- Model evaluation and comparison

**Git Tag:** `milestone-3-ml`
**Detailed Instructions:** [ML.md](ML.md)

### Peer Feedback

Give and receive constructive feedback from other teams.

**Instructions:** [project-feedback-guidelines.md](project-feedback-guidelines.md)

### Milestone 4: Final Delivery

Complete project delivery including:
- Polished analysis code
- Comprehensive website
- Final report
- All results and visualizations

**Git Tag:** `milestone-4-final`
**Detailed Instructions:** [WEBSITE.md](WEBSITE.md)

## Milestones

The project will be executed over several milestones, and each one has a specific set of requirements and instructions. Exact due dates for each milestone are available on the course Canvas website.

All of your work will be done within this team GitHub repository, and each milestone will be tagged with a specific [release tag](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) by the due date.

### Milestone Schedule and Deliverables

| Milestone | Date | Git Tag | Deliverables |
|-----------|------|---------|--------------|
| **Milestone 0: Data Acquisition & Planning** | Week 2 | `milestone-0-data` | • Filtered dataset (100M+ rows for chosen subreddits)<br>• `data/csv/dataset_summary.csv`<br>• `data/csv/subreddit_statistics.csv`<br>• `data/csv/temporal_distribution.csv`<br>• `BUSINESS_QUESTIONS.md` (10 questions with technical approaches) |
| **[Milestone 1: Exploratory Data Analysis](EDA.md)** | Week 3 | `milestone-1-eda` | • EDA Spark scripts in `code/eda/`<br>• Statistical analysis results in `data/csv/`<br>• Visualizations in `data/plots/`<br>• EDA findings document<br>• Answers to 3-4 EDA business questions |
| **[Milestone 2: Natural Language Processing](NLP.md)** | Week 5 | `milestone-2-nlp` | • NLP Spark scripts in `code/nlp/`<br>• Sentiment analysis results in `data/csv/`<br>• Topic modeling outputs in `data/csv/`<br>• NLP visualizations in `data/plots/`<br>• Answers to 3-4 NLP business questions |
| **[Milestone 3: Machine Learning](ML.md)** | Week 6 | `milestone-3-ml` | • ML Spark scripts in `code/ml/`<br>• Trained models in `code/ml/models/`<br>• Model evaluation metrics in `data/csv/`<br>• Feature importance analysis in `data/csv/`<br>• ML visualizations in `data/plots/`<br>• Answers to 2-3 ML business questions |
| **[Peer Feedback](project-feedback-guidelines.md)** | Week 7 | N/A | • Feedback received from other teams<br>• Feedback provided to other teams<br>• `PEER_FEEDBACK.md` documenting feedback |
| **[Milestone 4: Final Delivery](WEBSITE.md)** | Week 8 | `milestone-4-final` | • Complete website in `docs/`<br>• Final report (comprehensive analysis)<br>• All polished code<br>• All CSV results in `data/csv/`<br>• All visualizations in `data/plots/`<br>• README with project summary<br>• Presentation materials<br>• All 10 business questions answered |

**Detailed Instructions:**

1. [Milestone 1: Exploratory Data Analysis](EDA.md)
1. [Milestone 2: Natural Language Processing](NLP.md)
1. [Milestone 3: Machine Learning](ML.md)
1. [Milestone 4: Final Delivery and Website](WEBSITE.md)

## Repository structure

You will work within an **organized** repository and apply coding and development best practices. The repository has the following structure:

```.
├── LICENSE
├── README.md
├── code/
├── data/
├── docs/
└── website-source/
```
### Description

* The `code/` directory is where you will write all of your scripts. You will have a combination of PySpark and Python code, with one sub-directory per major task area (e.g., `code/eda/`, `code/nlp/`, `code/ml/`). You may add additional sub-directories as needed to modularize your development.
* The `data/` directory contains output plots in `data/plots/` and CSV files in `data/csv/`, or any other artifacts you plan to deliver.
* The `docs/` directory is where the final website will be built (GitHub Pages compatible). The website is generated from files in `website-source/`.
* The `website-source/` is where you will develop the website using Quarto (or your preferred method). It must render to `docs/`.

## Code

* Your code files must be well organized
* Do not work in a messy repository and then try to clean it up
* Use clear documentation and comments to explain what you are doing and the decisions you are making
* Do not write monolithic scripts - modularize your code (a script should do a single task)
* Use functions to promote code reuse
* Follow best practices for PySpark development

## Delivery mechanism

The output of the project will be delivered through a self-contained website in the `docs/` subdirectory, having `index.html` as the starting point. You will build the website incrementally over the milestones using the Quarto templates in `website-source/`.

[Read the website requirements.](WEBSITE.md)

## Evaluation

The project will be evaluated using the following high-level criteria:

* Level of analytical rigor at the graduate student level
* Level of technical approach
* Quality and clarity of your writing and overall presentation


### Grading rubric

- If a deliverable exceeds the requirements and expectations, that is considered A level work.
- If a deliverable just meets the requirements and expectations, that is considered A-/B+ level work.
- If a deliverable does not meet the requirements, that is considered B or lesser level work.

Deductions will be made for any of the following reasons:

- There is lack of analytical rigor:
    - Analytical decisions are not justified
    - Analysis is too simplistic
- Big data files included in the repository
- Instruction are not followed
- There are missing sections of the deliverable
- The overall presentation and/or writing is sloppy
- There are no comments in your code
- There are absolute filename links in your code
- The repository structure is sloppy
- Files are named incorrectly (wrong extensions, wrong case, etc.)
