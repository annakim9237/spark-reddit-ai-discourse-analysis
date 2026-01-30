# Milestone 2: Natural Language Processing

**Due:** Week 5
**Git Tag:** `v0.2-nlp`

## Overview

In this milestone, you will apply Natural Language Processing techniques to your Reddit text data. The goal is to extract insights from unstructured text, perform sentiment analysis, discover topics, and answer 3-4 of your NLP business questions.

## Learning Objectives

- Apply NLP techniques at scale using Apache Spark
- Perform sentiment analysis on large text corpora
- Apply topic modeling to discover themes
- Extract meaningful insights from text data
- Handle text preprocessing for big data

## Deliverables

### 1. Code (`code/nlp/` directory)

Create PySpark scripts for your NLP analysis:
- Text preprocessing (cleaning, tokenization, lemmatization)
- Sentiment analysis implementation
- Topic modeling (LDA or similar)
- Text feature extraction (TF-IDF, word embeddings)
- Any specialized NLP analyses for your domain

**Best practices:**
- Use Spark NLP library or similar distributed NLP tools
- Process text in batches for efficiency
- Cache intermediate results
- Handle Unicode and special characters properly

### 2. Results (`data/csv/` directory)

Save NLP analysis results:
- Sentiment scores by subreddit/post/comment
- Topic distributions and top keywords
- Word frequency analysis
- Trending terms over time
- Entity extraction results (if applicable)

**Naming convention:** `nlp_sentiment_by_subreddit.csv`, `nlp_topic_keywords.csv`, etc.

### 3. Visualizations (`data/plots/` directory)

Create visualizations specific to NLP:
- Sentiment distributions and trends over time
- Topic word clouds or bar charts
- Sentiment vs. engagement correlations
- Language patterns across subreddits
- Topic evolution over time

**Naming convention:** `nlp_[description].png` (e.g., `nlp_sentiment_trends.png`)

### 4. NLP Findings Document

Document your NLP analysis:
- Preprocessing decisions and rationale
- Sentiment analysis methodology and results
- Topic modeling approach and discovered topics
- Answers to 3-4 NLP business questions
- Insights for ML stage

### 5. Website Page (`website/nlp.html`)

Create your NLP page including:
- Overview of NLP techniques used
- Sentiment analysis results with visualizations
- Topic modeling findings
- Text mining insights
- Clear answers to your NLP business questions
- Connection to your high-level problem statement

## Example NLP Questions

Adapt these to your specific subreddits and problem:

1. **Sentiment analysis**: How does sentiment vary across subreddits and over time? Does sentiment correlate with engagement?
2. **Topic modeling**: What are the dominant topics discussed in your subreddits, and how do they evolve?
3. **Toxicity detection**: How does language toxicity vary across communities?
4. **Trend detection**: Can we identify emerging topics or breaking news from sudden changes in discussion patterns?

## NLP Techniques to Apply

### Text Preprocessing
- Lowercasing
- Removing URLs, mentions, special characters
- Tokenization
- Stop word removal
- Lemmatization/stemming
- Handling Reddit-specific formatting (markdown, quotes)

### Sentiment Analysis
Choose an approach:
- **VADER**: Good for social media text, handles emoticons and slang
- **TextBlob**: Simple polarity and subjectivity scores
- **Transformer models**: BERT-based models for more accurate sentiment
- **Custom lexicon**: Domain-specific sentiment dictionaries

### Topic Modeling
- **LDA (Latent Dirichlet Allocation)**: Traditional approach, works well at scale
- **NMF (Non-negative Matrix Factorization)**: Alternative to LDA
- **BERTopic**: Modern approach using embeddings (more computationally expensive)

### Text Feature Extraction
- **TF-IDF**: Term frequency-inverse document frequency
- **Word2Vec**: Word embeddings
- **N-grams**: Capture phrases and collocations

## Technical Tips

### Spark NLP Setup
```python
# Use Spark NLP or similar library
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA

# Text preprocessing pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
```

### Sentiment Analysis Example
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# For smaller scale, use pandas UDF
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf("double")
def sentiment_udf(texts: pd.Series) -> pd.Series:
    analyzer = SentimentIntensityAnalyzer()
    return texts.apply(lambda x: analyzer.polarity_scores(x)['compound'])

df = df.withColumn("sentiment", sentiment_udf(col("body")))
```

### Topic Modeling Example
```python
# LDA with Spark MLlib
lda = LDA(k=10, maxIter=20, featuresCol="features")
model = lda.fit(vectorized_df)

# Get topics
topics = model.describeTopics(maxTermsPerTopic=10)

# Transform documents
doc_topics = model.transform(vectorized_df)
```

### Text Cleaning Function
```python
import re
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(StringType())
def clean_text(text):
    if text is None:
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove special chars but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

df_cleaned = df.withColumn("clean_text", clean_text(col("body")))
```

## Common Pitfalls to Avoid

1. **Not handling null text**: Check for and handle missing/null text fields
2. **Ignoring data skew**: Some subreddits may dominate - consider sampling
3. **Over-preprocessing**: Don't remove too much; emoticons and caps can indicate sentiment
4. **Ignoring context**: Reddit-specific terms and slang need special handling
5. **Not validating topics**: Manually review topic keywords to ensure they make sense
6. **Memory issues**: Process text in batches, don't load everything into memory

## Resources

### Spark NLP
- [Spark NLP Documentation](https://nlp.johnsnowlabs.com/)
- [PySpark MLlib Text Processing](https://spark.apache.org/docs/latest/ml-features.html)

### Sentiment Analysis
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [TextBlob](https://textblob.readthedocs.io/)

### Topic Modeling
- [Gensim LDA](https://radimrehurek.com/gensim/models/ldamodel.html)
- [Spark MLlib LDA](https://spark.apache.org/docs/latest/ml-clustering.html#latent-dirichlet-allocation-lda)

## Integration with Other Milestones

### From EDA (Milestone 1)
- Use temporal patterns to analyze sentiment over time
- Focus on high-engagement posts/comments
- Filter to relevant time periods or subreddits

### To ML (Milestone 3)
- Use sentiment scores as features for prediction
- Use topic distributions as features
- Identify text patterns that predict engagement/virality
- Create text-based features for classification models

## Getting Help

- Check Spark NLP documentation for distributed text processing
- Review example NLP pipelines in course materials
- Ask on discussion board for technique recommendations
- Use office hours for debugging NLP issues

## Next Steps

After completing Milestone 2:
- Decide which NLP features to use in ML models
- Identify patterns that could improve predictions
- Consider how sentiment/topics relate to your ML questions
- Begin planning your ML approach
