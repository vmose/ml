# %%
# A notebook on Support Vector Machines (SVMs)
# Importing necessary libraries for data manipulation, visualization, and modeling
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# %%
# Loading the breast cancer dataset from sklearn's built-in datasets
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# %%
# Inspecting the keys available in the breast cancer dataset
cancer.keys()

# %%
# Printing a detailed description of the breast cancer dataset
print(cancer['DESCR'])

# %%
# Creating a DataFrame for the feature data and displaying the first few rows
# Also printing the shape of the dataset to understand its dimensions
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df.head())
print('\n')
print(cancer['data'].shape)

# %%
# Splitting the data into training and testing sets using an 70-30 split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, cancer['target'], test_size=0.3, random_state=101)

# %%
# Initializing and training a Support Vector Classifier (SVC) model
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

# Making predictions using the trained model
predictions = model.predict(X_test)

# %%
# Evaluating the performance of the initial SVC model
# Confusion Matrix provides insight into prediction performance for each class
# Classification Report includes precision, recall, f1-score, and support metrics
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

# %%
# Using GridSearchCV to find the optimal hyperparameters for the SVC model
# Parameter grid includes a range of values for C (regularization parameter) and gamma (kernel coefficient)
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train)

# %%
# Displaying the best parameters found by GridSearchCV
grid.best_params_

# %%
# Displaying the best estimator determined by GridSearchCV
grid.best_estimator_

# %%
# Making predictions using the optimized SVC model
pred_g = grid.predict(X_test)

# Evaluating the optimized model's performance using Confusion Matrix and Classification Report
print(confusion_matrix(y_test, pred_g))
print('\n')
print(classification_report(y_test, pred_g))

# %%
