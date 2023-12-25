# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import when

# Import pandas
import pandas as pd

# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder.appName("Insurance_Analytics").getOrCreate()

# COMMAND ----------

# Specify the path to your CSV file
csv_file_path = 'data.csv'

# Read the CSV file into a PySpark DataFrame
df_spark = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Perform data preprocessing and exploratory data analysis (EDA) here if needed

# COMMAND ----------

# Classification - Fraud Detection
threshold = 0.5  # You need to choose an appropriate threshold
df_spark = df_spark.withColumn('label', when(df_spark['Credit_Score'] == 'fraudulent', 1).otherwise(0))

feature_cols = ['Age', 'Annual_Income', 'Num_Credit_Card', 'Credit_History_Age']

assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

lr = LogisticRegression(featuresCol='features', labelCol='label', family='binomial')

classification_pipeline = Pipeline(stages=[assembler, lr])

# Split data into train and test sets
train_data, test_data = df_spark.randomSplit([0.8, 0.2], seed=42)

# Fit and evaluate the classification model
classification_model = classification_pipeline.fit(train_data)
classification_predictions = classification_model.transform(test_data)

classification_evaluator = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy')
classification_accuracy = classification_evaluator.evaluate(classification_predictions)

print(f"Classification Accuracy: {classification_accuracy}")

# COMMAND ----------

# Regression - Premium Prediction
regression_feature_cols = ['Age', 'Monthly_Inhand_Salary', 'Num_Credit_Card', 'Credit_History_Age']

regression_assembler = VectorAssembler(inputCols=regression_feature_cols, outputCol='features')

rf_regressor = RandomForestRegressor(featuresCol='features', labelCol='Annual_Income')

regression_pipeline = Pipeline(stages=[regression_assembler, rf_regressor])

# Split data into train and test sets
regression_train_data, regression_test_data = df_spark.randomSplit([0.8, 0.2], seed=42)

# Fit and evaluate the regression model
regression_model = regression_pipeline.fit(regression_train_data)
regression_predictions = regression_model.transform(regression_test_data)

regression_evaluator = RegressionEvaluator(labelCol='Annual_Income', metricName='rmse')
regression_rmse = regression_evaluator.evaluate(regression_predictions)

print(f"Regression RMSE: {regression_rmse}")

# COMMAND ----------

# Clustering - Customer Segmentation
clustering_feature_cols = ['Age', 'Annual_Income', 'Num_Credit_Card', 'Credit_History_Age']

clustering_assembler = VectorAssembler(inputCols=clustering_feature_cols, outputCol='features')

kmeans = KMeans(featuresCol='features', k=3, seed=42)  # You need to choose an appropriate k value

clustering_pipeline = Pipeline(stages=[clustering_assembler, kmeans])

# Fit and transform data using clustering model
clustering_model = clustering_pipeline.fit(df_spark)
clustered_data = clustering_model.transform(df_spark)

# Explore and analyze clustered_data to derive insights

# COMMAND ----------

# Additional steps as needed for reporting, visualizations, and documentation

# Save models, visualizations, and insights as needed for final deliverables
