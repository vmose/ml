# %%
# Import necessary libraries
# numpy: For numerical computations
# pandas: For data manipulation and analysis
# seaborn: For data visualization
# matplotlib: For plotting
# os: For interacting with the operating system
# Kaggle API: For downloading datasets from Kaggle
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# %%
# Initialize the Kaggle API and authenticate using existing credentials
api = KaggleApi()
api.authenticate()

# Define dataset information
# "dataset" specifies the Kaggle dataset slug
# "file_name" is the name of the dataset file to download
dataset = "abbasit/kyphosis-dataset"
file_name = "kyphosis.csv"

# Download the dataset to a specified location
# "destination" specifies the folder to save the dataset locally
destination = "data/"  # Ensure this folder exists or create it
os.makedirs(destination, exist_ok=True)
api.dataset_download_file(dataset, file_name, path=destination)

# Extract the CSV file if it is downloaded as a zip archive
import zipfile

file_path = os.path.join(destination, file_name + ".zip")
if os.path.exists(file_path):
    # Unzip the file and remove the zip archive
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination)
    os.remove(file_path)

# Load the extracted CSV file into a pandas DataFrame
csv_path = os.path.join(destination, file_name)
df = pd.read_csv(csv_path)

# Display the first few rows of the dataset to verify successful loading
print(df.head())

# %%
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
# Drop the target column ('Kyphosis') from the features
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Kyphosis', axis=1),  # Features
    df['Kyphosis'],               # Target variable
    test_size=0.3,                # 30% of data for testing
    random_state=101              # Set seed for reproducibility
)

# %%
# Train a Decision Tree Classifier on the training data
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

# Fit the model to the training data
dtc.fit(X_train, y_train)

# %%
# Make predictions on the test data
pred = dtc.predict(X_test)

# %%
# Evaluate the Decision Tree Classifier's performance
from sklearn.metrics import classification_report, confusion_matrix

# Print a classification report and confusion matrix
print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))

# %%
# Train a Random Forest Classifier on the training data
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest Classifier with 200 estimators
rfc = RandomForestClassifier(n_estimators=200)

# Fit the model to the training data
rfc.fit(X_train, y_train)

# %%
# Make predictions on the test data using the Random Forest Classifier
pred_r = rfc.predict(X_test)

# %%
# Evaluate the Random Forest Classifier's performance
# Print a classification report and confusion matrix
print(classification_report(y_test, pred_r))
print('\n')
print(confusion_matrix(y_test, pred_r))

# %%
# Analyze the class distribution of the target variable in the dataset
# This helps understand the balance of the dataset
df['Kyphosis'].value_counts()

# %%