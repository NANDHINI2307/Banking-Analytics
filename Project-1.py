# Databricks notebook source
# MAGIC %md #Documentation:
# MAGIC
# MAGIC 1. **Spark Session Creation:**
# MAGIC    - `SparkSession.builder.appName("Data_Visualization").getOrCreate()`: Initializes a Spark session with the name "Data_Visualization". This is the entry point to interact with Spark functionalities.
# MAGIC
# MAGIC 2. **Data Conversion:**
# MAGIC    - `df_spark.toPandas()`: Converts the Spark DataFrame 'df_spark' to a Pandas DataFrame ('df_pandas') for local data manipulation and visualization.
# MAGIC
# MAGIC 3. **Matplotlib Scatter Plot:**
# MAGIC    - `plt.scatter(df_pandas['Age'], df_pandas['Annual_Income'], c=df_pandas['label'], cmap='viridis')`: Creates a scatter plot using Matplotlib with 'Age' on the x-axis, 'Annual_Income' on the y-axis, and colors points based on the 'label' column. The color map 'viridis' is used for coloring.
# MAGIC    - `plt.xlabel('Age')`: Sets the x-axis label.
# MAGIC    - `plt.ylabel('Annual Income')`: Sets the y-axis label.
# MAGIC    - `plt.title('Scatter Plot using Matplotlib')`: Sets the plot title.
# MAGIC    - `plt.show()`: Displays the Matplotlib plot.
# MAGIC
# MAGIC 4. **Seaborn Scatter Plot:**
# MAGIC    - `sns.scatterplot(x='Age', y='Annual_Income', hue='label', data=df_pandas, palette='viridis')`: Creates a scatter plot using Seaborn with the same specifications as the Matplotlib scatter plot.
# MAGIC    - `plt.xlabel('Age')`, `plt.ylabel('Annual Income')`, `plt.title('Scatter Plot using Seaborn')`, `plt.show()`: Similar to Matplotlib, these lines set axis labels, title, and display the Seaborn scatter plot.
# MAGIC
# MAGIC 5. **Seaborn Pair Plot:**
# MAGIC    - `sns.pairplot(df_pandas[['Age', 'Annual_Income', 'Num_Credit_Card', 'Credit_Score', 'label']], hue='label', palette='viridis')`: Creates a pair plot using Seaborn for selected columns, with different colors for each 'label'.
# MAGIC    - `plt.suptitle('Pair Plot using Seaborn', y=1.02)`: Sets the title for the pair plot.
# MAGIC    - `plt.show()`: Displays the Seaborn pair plot.
# MAGIC
# MAGIC Note: Adjust column names and data based on your actual dataset.

# COMMAND ----------

# MAGIC %md # Import Dependencies
# MAGIC # Import necessary libraries and modules for data processing and machine learning.

# COMMAND ----------

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import when
from pyspark.ml.regression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import datasets

# COMMAND ----------

# MAGIC %md #Read CSV File
# MAGIC # Read CSV file into a PySpark DataFrame for further analysis.

# COMMAND ----------

# Specify the path to your CSV file
csv_file_path = 'data.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Converting Pandas DataFrame to PySpark DataFrame
df_spark = spark.createDataFrame(df)

# COMMAND ----------

# MAGIC %md #Customer_Segmentation
# MAGIC # Perform customer segmentation using KMeans clustering algorithm.
# MAGIC # Visualize the optimal number of clusters using the Elbow Method.

# COMMAND ----------

# MAGIC %md
# MAGIC - Customer segmentation is a crucial aspect of understanding and managing customer behavior. In this analysis, we leverage the KMeans clustering algorithm to segment customers based on specific features. The goal is to group customers with similar characteristics, enabling personalized targeting and tailored strategies.
# MAGIC
# MAGIC - It provides valuable insights into distinct customer groups. This information can be utilized for targeted marketing campaigns, personalized services, and improved customer satisfaction.

# COMMAND ----------

# Create a Spark session
spark = SparkSession.builder.appName("Customer_Segmentation").getOrCreate()

# Assuming 'df' is your PySpark DataFrame with customer data
# 'features' column is created by combining relevant features into a vector
assembler = VectorAssembler(
    inputCols=['Age', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts'],
    outputCol='features'
)

# Transform the DataFrame by adding the 'features' column
data = assembler.transform(df_spark)

# Assuming 'k' is the number of clusters you want to create
k = 10
kmeans = KMeans(k=k, seed=1)
model = kmeans.fit(data)

# Add a 'prediction' column to the DataFrame indicating the assigned cluster for each customer
result = model.transform(data)

# Display the customer_id and the assigned cluster label
result.select('Customer_ID', 'prediction').show()

# COMMAND ----------

distortions = []
K_values = range(2, 10)  # Start from k=2

for k in K_values:
    kmeans = KMeans(k=k, seed=1)
    model = kmeans.fit(data)
    distortions.append(model.summary.trainingCost)  # Corrected line

# Plotting the elbow curve
plt.plot(K_values, distortions, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Distortion (Sum of Squared Distances)')
plt.title('Elbow Method for Optimal K')
plt.show()

# COMMAND ----------

# MAGIC %md #Credit Risk Assessment (Classification)
# MAGIC # Build a Logistic Regression model to assess credit risk based on customer features.

# COMMAND ----------

# MAGIC %md
# MAGIC - Credit risk assessment is a critical task for financial institutions. In this analysis, we employ Logistic Regression to assess credit risk based on customer features. The goal is to predict whether a customer is likely to have good or bad credit based on relevant attributes.
# MAGIC
# MAGIC - Logistic Regression proves to be a valuable tool for credit risk assessment. It providing financial institutions with actionable insights for decision-making.

# COMMAND ----------

# Create a Spark session
spark = SparkSession.builder.appName("Logistic_Regression").getOrCreate()

# Assuming 'df_spark' is your PySpark DataFrame
# Assuming 'Credit_Score' is a string column that you want to convert to a numerical label
# Adjust the threshold based on your business logic
threshold = 700
df_spark = df_spark.withColumn('label', when(df_spark['Credit_Score'] >= threshold, 1).otherwise(0))

# Features for logistic regression
feature_cols = ['Age', 'Annual_Income', 'Num_Credit_Card']

# VectorAssembler to combine features into a single 'features' column
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
data = assembler.transform(df_spark)

# Logistic Regression model
lr = LogisticRegression()
model = lr.fit(data)
result = model.transform(data)

# Display relevant columns: Customer_ID, features, label, and prediction
result.select('Customer_ID', 'features', 'label', 'prediction').show()

# COMMAND ----------

# MAGIC %md #Performance Prediction (Regression)
# MAGIC # Build a Linear Regression model to predict key performance indicators for a banking institution.

# COMMAND ----------

# MAGIC %md
# MAGIC - Linear Regression is a powerful tool for predicting numerical values based on input features. In this scenario, we use Linear Regression to predict a performance metric (target variable) based on relevant features. The objective is to understand and model the relationship between the chosen features and the target variable.
# MAGIC
# MAGIC - This predictive model aids in understanding trends and making informed decisions based on historical and current data.

# COMMAND ----------

# Assuming 'features' is a vector column containing relevant features
assembler = VectorAssembler(inputCols=['Num_of_Loan', 'Outstanding_Debt', 'Credit_History_Age'], outputCol='features')
data = assembler.transform(df_spark)

# Assuming 'label' is the target variable for performance prediction
lr = LinearRegression()
model = lr.fit(data)
result = model.transform(data)

# Display date and predicted performance
result.select('Month', 'prediction').show()

# COMMAND ----------

# MAGIC %md #Data_Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC - Data visualization is a crucial step in exploratory data analysis (EDA) to gain insights and understand patterns within the dataset. In this analysis, we utilize Matplotlib and Seaborn libraries for creating various visualizations based on customer data stored in a PySpark DataFrame.
# MAGIC
# MAGIC - It aids in uncovering patterns, relationships, and trends within the dataset and providing a clearer understanding of the data and facilitating informed decision-making.

# COMMAND ----------

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("Data_Visualization").getOrCreate()

# Convert to Pandas DataFrame for easier visualization
df_pandas = df_spark.toPandas()

# Matplotlib scatter plot
plt.scatter(df_pandas['Age'], df_pandas['Annual_Income'], c=df_pandas['label'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.title('Scatter Plot using Matplotlib')
plt.show()

# Seaborn scatter plot
sns.scatterplot(x='Age', y='Annual_Income', hue='label', data=df_pandas, palette='viridis')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.title('Scatter Plot using Seaborn')
plt.show()

# Seaborn pair plot
sns.pairplot(df_pandas[['Age', 'Annual_Income', 'Num_Credit_Card', 'Credit_Score', 'label']], hue='label', palette='viridis')
plt.suptitle('Pair Plot using Seaborn', y=1.02)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # K-Nearest Neighbors (k-NN) Classification

# COMMAND ----------

# MAGIC %md
# MAGIC - K-Nearest Neighbors (k-NN) is a simple and effective classification algorithm used for predicting the class labels of data points based on the majority class of their k-nearest neighbors. In this example, we use the Iris dataset to demonstrate the k-NN classification approach.
# MAGIC
# MAGIC - K-Nearest Neighbors is a versatile and intuitive algorithm for classification tasks. It's essential to split the data into training and testing sets to assess the model's performance accurately. The accuracy score and classification report provide insights into the model's effectiveness and its ability to classify instances correctly.

# COMMAND ----------

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (class labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a k-Nearest Neighbors (k-NN) classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))


# COMMAND ----------

# MAGIC %md #Explanation of the code:
# MAGIC
# MAGIC 1. **Loading the Dataset:**
# MAGIC    - We use the Iris dataset provided by Scikit-learn, which is a well-known dataset for classification.
# MAGIC
# MAGIC 2. **Train-Test Split:**
# MAGIC    - The dataset is split into training and testing sets using the `train_test_split` function.
# MAGIC
# MAGIC 3. **Creating and Training the Model:**
# MAGIC    - We create a k-Nearest Neighbors classifier (`KNeighborsClassifier`) with a specified number of neighbors (in this case, 3).
# MAGIC    - The classifier is trained on the training data using the `fit` method.
# MAGIC
# MAGIC 4. **Making Predictions:**
# MAGIC    - We use the trained model to make predictions on the test data.
# MAGIC
# MAGIC 5. **Evaluating the Model:**
# MAGIC    - We calculate the accuracy of the model using the `accuracy_score` function.
# MAGIC    - We display a classification report that includes precision, recall, and F1-score for each class.
# MAGIC
# MAGIC This example demonstrates a basic classification task using Scikit-learn. Depending on your specific project and data, you may choose a different algorithm or adjust parameters accordingly.
