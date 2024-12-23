# %%
#using a linear regression algorithm to predict the price of a home
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os

from kaggle.api.kaggle_api_extended import KaggleApi

# %%
api = KaggleApi()
api.authenticate()

# Define dataset information
dataset = "aariyan101/usa-housingcsv"
file_name = "USA_Housing.csv"

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
data = pd.read_csv(csv_path)

# Use the DataFrame
print(data.head())

# %%
data.info()

# %%
data.describe()

# %%

sb.pairplot(data)

# %%
sb.histplot(data['Price'])

# %%
data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price']].corr()

# %%
sb.heatmap(data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price']].corr())

# %%
X=data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y=data['Price']

# %%
#sklearn.cross_validation is no longer a recognized module
from sklearn.model_selection import train_test_split  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# %%
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

# %%
print(lm.intercept_)
print('\n')
print(lm.coef_)

# %%
df=pd.DataFrame(lm.coef_,X.columns, columns=['Coefficients'])
df

# %%
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()
california.keys()

# %%
print(california['DESCR'])
# %%
predictions=lm.predict(X_test)
predictions

# %%
y_test

# %%
sb.scatterplot(y_test,predictions)

# %%
sb.histplot((y_test-predictions))

# %%
from sklearn import metrics

# %%
metrics.mean_absolute_error(y_test,predictions)

# %%
MSE=metrics.mean_squared_error(y_test,predictions)
MSE

# %%
np.sqrt(MSE)

# %%
from sklearn.datasets import load_iris

# %%
iris= load_iris()
iris.keys()

# %%
print(iris['DESCR'])

# %%
