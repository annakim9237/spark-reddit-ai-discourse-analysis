# AI Discourse at Scale: Topic, Sentiment, and Community Dynamics Across Reddit

*A data-driven exploration of AI discussions on Reddit using distributed computing.*

**You can explore more on the project website and in the full PDF report.**
- [Project Website](https://kim-anna.quarto.pub/dsan-6000-big-data-project/)

- [Project Report (PDF)](https://github.com/annakim9237/spark-reddit-ai-discourse-analysis/blob/main/GU-DSAN6000-FALL-TEAM05.pdf)



## Abstract
The rapid growth of AI in 2023–2024 sparked widespread discussion online, and Reddit became a central platform for early reactions, debates, and speculation. Its tech-leaning demographic and culture of candid, unfiltered commentary made it a useful environment for observing how people engaged with new generative AI systems as they emerged. Using a 445 GB archive of Reddit comments and submissions spanning June 2023 to July 2024, we filtered the data to focus on selected AI-related and general-interest subreddits in order to study these conversations more closely. With distributed processing in Spark, we generated temporal and subreddit-level summaries to track changes in monthly activity and identify which communities contributed most to the overall discourse.

By capturing the scale and structure of AI-related discourse on Reddit, this analysis provides an early snapshot of how digital communities processed, debated, and responded to one of the most transformative technological shifts. 

---

## Context

This project analyzes a large-scale Reddit archive containing comments and submissions stored in parquet format. The dataset spans June 2023 through July 2024, covering a pivotal 14-month period during which public interest in generative AI grew rapidly. Because the full archive contains billions of rows and hundreds of gigabytes of data, the analysis required distributed computing rather than single-machine processing.

## Dataset Characteristics after filtering(June 2023 – July 2024)

| Data Type     | Date Range Start | Date Range End | Total Rows     | 
|---------------|------------------|----------------|----------------|
| Comments      | 2023-06-01       | 2024-07-31     | 3,675,768,958  |
| Submissions   | 2023-06-01       | 2024-07-31     |   567,890,869  |

## Filtered Subreddits

To focus the analysis on AI-related conversations—and include a few general-interest communities for comparison—we filtered the archive to include only selected subreddits.

| Category          | Subreddits Included                                                    |
| ----------------- | ---------------------------------------------------------------------- |
| AI Consumer       | ChatGPT, StableDiffusion, OpenAI, ClaudeAI, PerplexityAI               |
| AI Technical      | LocalLLaMA, MachineLearning, datascience, computerscience, programming |
| Future/AI Culture | Futurology, singularity                                                |
| General Interest  | AskReddit, neoliberal                                                  |
| Total             | 15 subreddits                                                          |


| Subreddit       | # Comments | # Submissions | Total Rows | Avg Score | Date Range                  |
|-----------------|------------|---------------|------------|-----------|-----------------------------|
| AskReddit       | 55,851,868 |  2,686,082    | 58,537,950 | 10.12     | 2023-06-01 to 2024-07-31    |
| neoliberal      |  4,673,171 |     35,352    |  4,708,523 | 38.27     | 2023-06-01 to 2024-07-31    |
| ChatGPT         |  1,748,893 |    145,862    |  1,894,755 | 28.91     | 2023-06-01 to 2024-07-31    |
| singularity     |  1,166,552 |     37,112    |  1,203,664 | 24.85     | 2023-06-01 to 2024-07-31    |
| Futurology      |    898,463 |     15,884    |    914,347 | 91.45     | 2023-06-01 to 2024-07-31    |
| StableDiffusion |    763,492 |     74,526    |    838,018 | 12.37     | 2023-06-01 to 2024-07-31    |
| LocalLLaMA      |    425,095 |     31,020    |    456,115 | 11.93     | 2023-06-01 to 2024-07-31    |
| OpenAI          |    337,768 |     25,574    |    363,342 | 14.36     | 2023-06-01 to 2024-07-31    |
| programming     |    325,140 |     31,396    |    356,536 | 14.12     | 2023-06-01 to 2024-07-31    |
| datascience     |    193,664 |     24,717    |    218,381 |  5.23     | 2023-06-01 to 2024-07-31    |
| MachineLearning |    146,901 |     37,191    |    184,092 |  4.31     | 2023-06-01 to 2024-07-31    |
| ClaudeAI        |     59,790 |      5,097    |     64,887 |  6.87     | 2023-06-02 to 2024-07-31    |
| computerscience |     31,181 |      7,504    |     38,685 |  3.81     | 2023-06-01 to 2024-07-31    |
| GPT4            |      4,009 |      1,623    |      5,632 |  1.65     | 2023-06-01 to 2024-07-31    |
| PerplexityAI    |         33 |         28    |         61 |  1.19     | 2023-06-17 to 2024-07-31    |


---


## Repository Structure
```
FALL-2025-PROJECT-TEAM05/
│
├── code/                     # All processing + modeling scripts
├── data/                     # Processed local samples (not the full dataset)
├── docs/                     # Notes, schema summaries
├── local_data/               # Small local subsets for testing
├── spark-cluster/            # EC2 + Spark setup scripts
├── website-source/           # Interactive visualization site
│
├── README.md                 # (This file)
├── EDA.md                    # Extended EDA notes
├── NLP.md                    # NLP pipeline notes
├── ML.md                     # Machine learning notes
├── SCHEMA_EXAMINATION.md     # Schema deep dive
└── GU-DSAN6000-FALL-TEAM05.pdf   # Full project report
```
