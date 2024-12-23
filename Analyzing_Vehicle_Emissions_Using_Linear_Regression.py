# %%
# #using a linear regression algorithm to predict the CO2 output of a vehicle method 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn import linear_model

# %%
api = KaggleApi()
api.authenticate()

# Define dataset information
dataset = "ramlalnaik/fuelconsumptionco2"
file_name = "FuelConsumptionCo2.csv"

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
data1 = pd.read_csv(csv_path)
data1.head()

# %%
#Simple regression
data1=data1[["ENGINESIZE","CO2EMISSIONS"]]
plt.scatter(data1["ENGINESIZE"],data1["CO2EMISSIONS"],color='g')
plt.xlabel("CO2EMISSIONS")
plt.ylabel("ENGINESIZE")
plt.title("CO2EMISSIONS vs ENGINESIZE")
plt.show()


# %%
#generating training and testing data. 
#using 80% data for training

train = data1[:(int((len(data1)*0.8)))]
test = data1[(int((len(data1)*0.8))):]

# %%
# Modeling:
# Using sklearn package to model data :
regr = linear_model.LinearRegression()
train_x = np.array(train[['ENGINESIZE']])
train_y = np.array(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)

# The coefficients:
print ('coefficients :' ,regr.coef_) #Slope
print ('Intercept : ',regr.intercept_) #Intercept

# %%
# Plotting the regression line:
plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'], color='g')
plt.plot(train_x, regr.coef_*train_x + regr.intercept_, '-r')
plt.xlabel('Engine size')
plt.ylabel('Emission')
plt.title('Regression line')
plt.show()

# %%
# Predicting values:
# Function for predicting future values :
def get_regression_predictions(input_features,intercept,slope):
    predicted_values = input_features*slope + intercept
    return predicted_values

# %%
# Predicting emission for future car:
my_engine_size = 3.5
estimatd_emission = get_regression_predictions(my_engine_size,regr.intercept_[0],regr.coef_[0][0])
print ('Estimated Emission :',estimatd_emission)
      
# %%
# Checking various accuracy:
from sklearn.metrics import r2_score
test_x = np.array(test[['ENGINESIZE']])
test_y = np.array(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Mean sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

# %%
#Multiple regression
# Consider features we want to work on:
data1 = pd.read_csv(csv_path)
X = data1[[ 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY', 
 'FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']]
Y = data1['CO2EMISSIONS']
# Generating training and testing data from our data:
# We are using 80% data for training.
train = data1[:(int((len(data1)*0.8)))]
test = data1[(int((len(data1)*0.8))):]

# %%
#Modeling:
#Using sklearn package to model data:

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.array(train[[ 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']])

train_y = np.array(train['CO2EMISSIONS'])

test_x = np.array(test[[ 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']])

test_y = np.array(test['CO2EMISSIONS'])

regr.fit(train_x,train_y)

# %%
# print the coefficient values:
coeff_data = pd.DataFrame(regr.coef_ , X.columns , columns=['Coefficients'])
coeff_data

# %%
#Now let’s do prediction of data:
Y_pred = regr.predict(test_x)

# %%
# Check accuracy:
from sklearn.metrics import r2_score
R = r2_score(test_y , Y_pred)
print ("R² :",R)
# %%
