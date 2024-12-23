# %%
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os

from kaggle.api.kaggle_api_extended import KaggleApi


# %%
# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Define dataset information
dataset = "abbasit/kyphosis-dataset"
file_name = "kyphosis.csv"

# Download the dataset to a specified location
destination = "data/"  # Choose your desired folder
os.makedirs(destination, exist_ok=True)
api.dataset_download_file(dataset, file_name, path=destination)

# Extract the CSV if downloaded as a zip
import zipfile

file_path = os.path.join(destination, file_name + ".zip")
if os.path.exists(file_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination)
    os.remove(file_path)

# Load the CSV into a pandas DataFrame
csv_path = os.path.join(destination, file_name)
df = pd.read_csv(csv_path)

# Use the DataFrame
print(df.head())

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(df.drop('Kyphosis',axis=1), df['Kyphosis'], test_size=0.3, random_state=101)

# %%
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()

# %%
dtc.fit(X_train,y_train)

# %%
pred = dtc.predict(X_test)
# %%
from sklearn.metrics import classification_report, confusion_matrix

# %%
print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))

# %%
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=200)

# %%
rfc.fit(X_train,y_train)

# %%
pred_r=rfc.predict(X_test)

# %%
print(classification_report(y_test,pred_r))
print('\n')
print(confusion_matrix(y_test,pred_r))

# %%
df['Kyphosis'].value_counts()

# %%
