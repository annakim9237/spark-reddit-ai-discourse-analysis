# Milestone 3: Machine Learning

**Due:** Week 6
**Git Tag:** `v0.3-ml`

## Overview

In this milestone, you will build and evaluate machine learning models to answer 2-3 of your ML business questions. This involves feature engineering, model training, evaluation, and interpretation using Apache Spark's MLlib.

## Learning Objectives

- Build ML pipelines with Spark MLlib
- Perform feature engineering at scale
- Train classification, regression, or clustering models
- Evaluate models using appropriate metrics
- Interpret model results and feature importance
- Handle class imbalance and other ML challenges

## Deliverables

### 1. Code (`code/ml/` directory)

Create well-structured PySpark ML scripts:
- Feature engineering pipeline
- Model training scripts
- Model evaluation and validation
- Hyperparameter tuning
- Prediction generation

Save trained models in: `code/ml/models/`

**Best practices:**
- Use Spark ML Pipelines for reproducibility
- Implement cross-validation
- Track experiments (model versions, parameters, results)
- Use appropriate train/validation/test splits

### 2. Results (`data/csv/` directory)

Save ML outputs:
- Model evaluation metrics (accuracy, precision, recall, F1, AUC, RMSE, etc.)
- Feature importance rankings
- Prediction results (sample or aggregated)
- Confusion matrices
- Hyperparameter tuning results

**Naming convention:** `ml_model_evaluation.csv`, `ml_feature_importance.csv`, etc.

### 3. Visualizations (`data/plots/` directory)

Create ML-specific visualizations:
- ROC curves and precision-recall curves
- Confusion matrices
- Feature importance bar charts
- Learning curves
- Prediction distribution plots
- Model comparison charts

**Naming convention:** `ml_[model]_[metric].png` (e.g., `ml_random_forest_roc_curve.png`)

### 4. ML Findings Document

Document your ML work:
- Problem formulation (classification/regression/clustering)
- Feature engineering approach
- Model selection rationale
- Training procedure and hyperparameters
- Evaluation results
- Model interpretation and insights
- Answers to 2-3 ML business questions

### 5. Website Page (`website/ml.html`)

Create your ML page including:
- Overview of ML problems tackled
- Feature engineering description
- Model descriptions and architectures
- Evaluation metrics with interpretations
- Feature importance analysis
- Clear answers to your ML business questions
- Business implications of model results

## Example ML Questions

Adapt these to your domain:

1. **Classification**: Can we predict which posts will receive high engagement based on content, timing, and author features?
2. **Regression**: Can we predict comment count or score for new posts?
3. **Clustering**: Can we identify user segments based on posting behavior and interests?
4. **Anomaly detection**: Can we detect bots or spam accounts from posting patterns?

## ML Tasks to Consider

### Classification Problems
- Binary: High/low engagement, viral/non-viral, toxic/non-toxic
- Multi-class: Topic categorization, sentiment classification
- Multi-label: Post tagging, content classification

### Regression Problems
- Score prediction
- Comment count prediction
- Time-to-peak prediction
- Engagement rate prediction

### Clustering Problems
- User segmentation
- Community detection
- Topic clustering
- Content similarity grouping

## Feature Engineering

### Temporal Features
```python
from pyspark.sql.functions import hour, dayofweek, month

df = df.withColumn("hour_of_day", hour("created_utc"))
df = df.withColumn("day_of_week", dayofweek("created_utc"))
df = df.withColumn("month", month("created_utc"))
```

### Text Features
```python
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="raw_features")
idf = IDF(inputCol="raw_features", outputCol="text_features")
```

### Aggregated Features
```python
# Author history features
author_stats = df.groupBy("author").agg(
    count("*").alias("total_posts"),
    avg("score").alias("avg_score"),
    stddev("score").alias("score_std")
)

df = df.join(author_stats, on="author", how="left")
```

### Derived Features
```python
from pyspark.sql.functions import length, col

df = df.withColumn("title_length", length(col("title")))
df = df.withColumn("has_url", col("url").isNotNull().cast("int"))
df = df.withColumn("is_weekend", col("day_of_week").isin([6, 7]).cast("int"))
```

## Model Training Examples

### Classification with Random Forest
```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Split data
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Train model
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=10,
    seed=42
)
model = rf.fit(train)

# Evaluate
predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator()
auc = evaluator.evaluate(predictions)
```

### Regression with Gradient Boosting
```python
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="score",
    maxIter=100,
    maxDepth=5
)
model = gbt.fit(train)

predictions = model.transform(test)
evaluator = RegressionEvaluator(labelCol="score", metricName="rmse")
rmse = evaluator.evaluate(predictions)
```

### Hyperparameter Tuning
```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create parameter grid
paramGrid = (ParamGridBuilder()
    .addGrid(rf.numTrees, [50, 100, 150])
    .addGrid(rf.maxDepth, [5, 10, 15])
    .build())

# Cross validation
cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=5
)

cv_model = cv.fit(train)
best_model = cv_model.bestModel
```

## Model Evaluation

### Classification Metrics
```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Accuracy
accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = accuracy_eval.evaluate(predictions)

# F1 Score
f1_eval = MulticlassClassificationEvaluator(metricName="f1")
f1 = f1_eval.evaluate(predictions)

# Confusion Matrix
predictions.groupBy("label", "prediction").count().show()
```

### Feature Importance
```python
# For tree-based models
importances = model.featureImportances
feature_names = ["feature1", "feature2", ...]  # Your feature names

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances.toArray()
}).sort_values('importance', ascending=False)

importance_df.to_csv('data/csv/ml_feature_importance.csv', index=False)
```

## ML Pipeline Example
```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler

# Create pipeline
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "feature3"],
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features"
)

classifier = RandomForestClassifier(
    featuresCol="features",
    labelCol="label"
)

pipeline = Pipeline(stages=[assembler, scaler, classifier])

# Fit pipeline
model = pipeline.fit(train)

# Save pipeline
model.write().overwrite().save("code/ml/models/rf_pipeline")
```

## Common Pitfalls to Avoid

1. **Data leakage**: Don't use future information to predict the past
2. **Not handling class imbalance**: Use stratified sampling or class weights
3. **Overfitting**: Use cross-validation and regularization
4. **Ignoring feature scaling**: Normalize/standardize features when needed
5. **Not interpreting models**: Feature importance is crucial for business insights
6. **Poor train/test splits**: Ensure temporal ordering if time-based
7. **Not validating on holdout**: Always keep a test set untouched until final evaluation

## Resources

### Spark MLlib
- [MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Classification and Regression](https://spark.apache.org/docs/latest/ml-classification-regression.html)
- [ML Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html)
- [ML Tuning](https://spark.apache.org/docs/latest/ml-tuning.html)

### Best Practices
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Rules of Machine Learning (Google)](https://developers.google.com/machine-learning/guides/rules-of-ml)

## Integration with Previous Milestones

### From EDA (Milestone 1)
- Use statistical insights to engineer features
- Understand data distributions for preprocessing
- Identify important variables

### From NLP (Milestone 2)
- Use sentiment scores as features
- Use topic distributions as features
- Include text-derived features (toxicity, readability)

## Getting Help

- Review Spark MLlib documentation for API details
- Check course discussion board for common ML issues
- Use office hours for model selection advice
- Review example ML notebooks in course materials

## Next Steps

After completing Milestone 3:
- Begin integrating results into final website
- Prepare comprehensive analysis narrative
- Identify key findings across all milestones
- Start working on conclusion and recommendations
