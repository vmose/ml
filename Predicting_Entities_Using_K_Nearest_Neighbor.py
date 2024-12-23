# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# %%
df=pd.read_csv('Classified Data.csv',index_col=0)
df.head()

# %%
#the point of a scaler is to standardize all the numerical metrics to fit the model
#this is necessary since we do not know what the different fields refer to
from sklearn.preprocessing import StandardScaler 
SS=StandardScaler()
SS.fit(df.drop('TARGET CLASS',axis=1))

# %%
Scal=SS.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(Scal,columns=df.columns[:-1])
df_feat
#this new dataframe now becomes our training data

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_feat, df['TARGET CLASS'], test_size=0.3, random_state=101)

from sklearn.neighbors import KNeighborsClassifier 
KNN=KNeighborsClassifier(n_neighbors=1)

# %%
KNN.fit(X_train, y_train)

# %%
prediction = KNN.predict(X_test)
prediction

# %%
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))

# %%
error_rate=[]

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# %%
#the point of this is to see if we can squeeze some more performance from the model
#the lower the error rate the better
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# %%
#Compare analysis using new K-Value
knn = KNeighborsClassifier(n_neighbors=36)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=36')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

# %%
