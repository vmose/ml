# %%
#a notebook on Support Vector Machines
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# %%
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()

# %%
cancer.keys()
# %%
print(cancer['DESCR'])

# %%
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df.head())
print('\n')
print(cancer['data'].shape)

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, cancer['target'], test_size=0.3, random_state=101)

# %%
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
predictions= model.predict(X_test)

# %%
from sklearn.metrics import classification_report,confusion_matrix  
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

# %%
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.001]}
grid=GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train,y_train)

# %%
grid.best_params_

# %%
grid.best_estimator_

# %%
pred_g=grid.predict(X_test)
print(confusion_matrix(y_test,pred_g))
print('\n')
print(classification_report(y_test,pred_g))

# %%
