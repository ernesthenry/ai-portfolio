from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# BUSINESS PROBLEM: Customer Segmentation
# Group customers so Marketing can send relevant emails (e.g. "Budget Savers" vs "Big Spenders").

# 1. GENERATE DATA (Income vs Spending Score)
X = pd.DataFrame({
    'income': np.concatenate([np.random.normal(20, 5, 100), np.random.normal(80, 10, 100)]),
    'spending': np.concatenate([np.random.normal(20, 5, 100), np.random.normal(90, 10, 100)])
})

# 2. K-MEANS CLUSTERING
# Finds 'k' centoids that minimize the distance between points.
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

X['cluster'] = clusters
print(X.groupby('cluster').mean())
# Result should show Low Income/Low Spend cluster vs High/High.
