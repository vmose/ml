# %%
#using Principal Component Analysis to compress data with multiple attributes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# %%
from sklearn.datasets import load_breast_cancer
cancer= load_breast_cancer()

# %%
df=pd.DataFrame(cancer['data'],columns= cancer
['feature_names'])
df.head()

# %%
#these are the two categorizations of the types of breast tumors
print(cancer['target_names'])

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# %%
scaler.fit(df)

# %%
scaled_data = scaler.transform(df)

# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

# %%
pca.fit(scaled_data)
# %%
x_pca = pca.transform(scaled_data)

# %%
#reducing the multiple (30) components down to 2 for simpler categorization
print(scaled_data.shape)
print(x_pca.shape)

# %%
#plotting the transformed dataset
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],)

# %%
#this is not easy to interpret at first but adding a color filter helps show the variance
#now we can see clearly which tumors are benign or malignant
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# %%
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
df_comp

# %%
#this is map of correlation of the feature components
plt.figure(figsize=(8,6))
sb.heatmap(df_comp,cmap='plasma')

# %%
