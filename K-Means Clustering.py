# %%
# Import necessary libraries
# numpy: for numerical operations
# pandas: for handling and analyzing data
# matplotlib: for data visualization
# seaborn: for enhanced visualizations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# %%
# Import `make_blobs` from sklearn.datasets
# `make_blobs` is used to generate synthetic datasets with clusters
from sklearn.datasets import make_blobs

# %%
# Generate a synthetic dataset with 4 centers using `make_blobs`
# n_samples: Number of data points (200)
# n_features: Number of features/dimensions (2)
# centers: Number of clusters (4)
# cluster_std: Standard deviation of clusters (1.8)
# random_state: Ensures reproducibility (101)
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)

# %%
# Display the shape of the feature dataset (200 samples, 2 features)
data[0].shape

# %%
# Visualize the dataset
# The data points are color-coded according to their cluster labels (data[1])
plt.scatter(data[0][:,0], data[0][:,1], c=data[1])
plt.title("Synthetic Data with Cluster Labels")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# %%
# Display the true labels of the generated clusters (data[1])
data[1]

# %%
# Import KMeans algorithm from sklearn.cluster
# KMeans is used for unsupervised clustering
from sklearn.cluster import KMeans

# Initialize the KMeans algorithm
# n_clusters: Number of clusters to form (4)
kmeans = KMeans(n_clusters=4)

# %%
# Train the KMeans model using the feature data (data[0])
kmeans.fit(data[0])

# %%
# Display the coordinates of the cluster centers identified by KMeans
kmeans.cluster_centers_

# %%
# Display the cluster labels predicted by KMeans for each data point
kmeans.labels_

# %%
# Compare the KMeans cluster assignments with the true cluster labels
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))

# Plot KMeans cluster assignments
ax1.set_title('KMeans Clustering')
ax1.scatter(data[0][:,0], data[0][:,1], c=kmeans.labels_, cmap='rainbow')

# Plot the original true cluster labels
ax2.set_title("Original Cluster Labels")
ax2.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')

plt.show()

# %%
