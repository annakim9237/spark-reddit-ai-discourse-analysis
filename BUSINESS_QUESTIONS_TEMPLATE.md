# Business Questions and Technical Approaches

> ⚠️ **IMPORTANT: This is an EXAMPLE template!**
>
> These example questions provide a starting point, but your project should be unique:
> - **EDA questions** may have similar themes (temporal patterns, distributions, correlations), but your specific business problem, metrics, and technical approach should reflect your chosen subreddits and analysis goals
> - **NLP and ML questions** should be novel and tailored to your dataset - aim to find interesting problems rather than reproduce these examples verbatim
> - Your **high-level problem statement** and how you break it down should be original and relevant to your domain
>
> Use this template as a guide for structure, format, and level of detail expected.

---

**Project:** [Your Project Title]
**Team:** [Team Member Names]
**Dataset:** [Dataset Name and Scale]
**High-Level Problem Statement:** [Your overarching business problem]


**Updated questions EDA for our team (Erika)**:

- Which AI subreddit categories show the highest overall engagement (average comments and scores)?
-  How does post title length relate to user engagement across different AI subreddit categories?
- Do posts phrased as questions receive higher engagement than non-question posts?
- What title length ranges generate the highest average engagement within each AI subreddit category?
---

## Updated questions EDA for our team (EDA – Anna)

I'm putting more detail for later documenting

**When is the optimal time to post content on AI-related subreddits to maximize engagement and content quality?**

## Updated questions EDA for our team (EDA - Violet)

- How has the popularity and sentiment around different types of generative AI (e.g., image, text, music, code) evolved across Reddit subreddits over time, and what patterns emerge in user engagement and comment dynamics?

### Context / Purpose
On community platforms like Reddit, the posting time significantly influences a post’s visibility, score, and number of comments.

By analyzing user activity patterns (day of week and hour of day) within AI-related subreddits, this study aims to identify the time periods when content is most actively consumed and positively received.

These insights can directly inform:
- Content marketing strategies  
- Posting schedule optimization  
- Community management efficiency


### Analytical Approach
1. **Feature Extraction**  
   Extract `day_of_week` and `hour` variables from the `created_utc` timestamp.

2. **Aggregation**  
   Aggregate data by `subreddit × day_of_week × hour` to compute:
   - `num_posts` (posting volume)  
   - `avg(score)` (average engagement)

3. **Visualization**  
   Create heatmaps and bar charts to visualize:
   - Posting activity patterns  
   - Relationship between post volume and average score

4. **Interpretation**  
   Compare the time windows with:
   - Highest posting activity  
   - Highest average scores  
   → Determine the **optimal posting window** that balances engagement and content quality.


---
From here professor example

## Question 1: How do posting patterns vary across different subreddits and time periods?

**Analysis Type:** EDA

**Technical Approach:**
- Extract temporal features (hour, day of week, month) from `created_utc` timestamps using Spark
- Aggregate post counts by subreddit and time dimensions using groupBy operations
- Calculate engagement metrics (average score, comment counts) across temporal buckets
- Create visualizations showing peak activity times and seasonal trends per subreddit
- Expected outcome: Identify optimal posting times and understand community activity patterns for content strategy



---

## Question 2: What topics are most discussed in technology-focused subreddits?

**Analysis Type:** NLP

**Technical Approach:**
- Preprocess text data (tokenization, stop word removal, lemmatization) using Spark NLP
- Apply Latent Dirichlet Allocation (LDA) topic modeling to post titles and body text
- Extract top keywords and dominant topics using term frequency analysis
- Visualize topic distributions and track topic evolution over time
- Expected outcome: Understand trending discussions and emerging technologies to inform product development

---

## Question 3: Can we predict which posts will receive high engagement?

**Analysis Type:** ML

**Technical Approach:**
- Engineer features from post metadata (title length, time of day, author history, subreddit)
- Extract text features using TF-IDF or word embeddings from post titles
- Train binary classification model (Random Forest/Gradient Boosting) to predict high-score posts
- Evaluate model using precision, recall, and F1-score with cross-validation
- Expected outcome: Build a model to guide content creators on writing engaging posts

---

## Question 4: What is the relationship between post characteristics and comment activity?

**Analysis Type:** EDA

**Technical Approach:**
- Join submissions and comments data on post IDs using Spark SQL
- Calculate correlation between post features (title length, score, time) and comment metrics
- Perform statistical analysis to identify significant predictors of discussion volume
- Create scatter plots and correlation matrices to visualize relationships
- Expected outcome: Understand what drives community discussions and engagement depth

---

## Question 5: How does sentiment vary across different communities?

**Analysis Type:** NLP

**Technical Approach:**
- Apply sentiment analysis to comment text using pre-trained models (VADER or transformer-based)
- Aggregate sentiment scores by subreddit and calculate distribution metrics
- Compare sentiment patterns across communities using statistical tests
- Visualize sentiment distributions and identify communities with extreme sentiment patterns
- Expected outcome: Understand community culture and toxicity levels for moderation strategies

---

## Question 6: Can we identify bot accounts based on posting behavior?

**Analysis Type:** ML

**Technical Approach:**
- Extract user-level features (posting frequency, time patterns, text similarity, account age)
- Calculate behavioral metrics (response times, post diversity, engagement patterns)
- Train anomaly detection or classification model to identify suspicious accounts
- Validate results by examining flagged accounts and their posting patterns
- Expected outcome: Develop system to detect automated accounts for platform integrity

---

## Question 7: What factors influence upvote/downvote ratios?

**Analysis Type:** EDA

**Technical Approach:**
- Analyze score distributions across different post types and subreddits
- Calculate statistical measures of controversy and polarization
- Examine relationship between content characteristics and vote patterns using regression
- Create visualizations comparing vote distributions across categories
- Expected outcome: Understand what content resonates with different audiences

---

## Question 8: Can we detect emerging trends before they go viral?

**Analysis Type:** NLP + ML

**Technical Approach:**
- Track keyword frequency changes over rolling time windows using Spark streaming concepts
- Calculate velocity metrics for topic growth using time-series analysis
- Apply change point detection algorithms to identify sudden topic surges
- Build early warning system using threshold-based alerts on trend metrics
- Expected outcome: Create system to identify trending topics for early content strategy

---

## Question 9: What is the lifecycle of viral posts?

**Analysis Type:** EDA

**Technical Approach:**
- Track engagement metrics (score, comments) over time for high-performing posts
- Calculate growth curves and time-to-peak metrics using temporal analysis
- Compare viral post characteristics to typical posts using statistical tests
- Visualize engagement trajectories and identify common patterns
- Expected outcome: Understand virality mechanics for content optimization

---

## Question 10: Can we predict user churn or sustained engagement?

**Analysis Type:** ML

**Technical Approach:**
- Define engagement metrics and churn criteria from user posting history
- Engineer temporal features (posting frequency trends, engagement decline)
- Train survival analysis or classification model to predict user retention
- Use feature importance to identify key drivers of continued engagement
- Expected outcome: Develop retention strategies by understanding disengagement signals

---

## Summary

**EDA Questions:** 1, 4, 7, 9 (4 questions)
**NLP Questions:** 2, 5, 8 (3 questions)
**ML Questions:** 3, 6, 10 (3 questions)

These questions span multiple analysis techniques and provide comprehensive insights into community behavior, content strategy, and platform health. Each question requires big data processing techniques to analyze hundreds of millions of rows efficiently.
