from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np

# 1. ANOMALY DETECTION (Isolation Forest)
# Business Use: Fraud Detection, Server Failure Prediction.
# Logic: "Anomalies are easier to isolate (require fewer splits) than normal points."
def run_anomaly_detection():
    # Generate data: 100 normal points, 10 outliers
    X_normal = np.random.normal(0, 1, (100, 2))
    X_outliers = np.random.uniform(5, 10, (10, 2))
    X = np.vstack([X_normal, X_outliers])
    
    clf = IsolationForest(contamination=0.1, random_state=42)
    preds = clf.fit_predict(X) 
    # -1 = Outlier, 1 = Normal
    
    detected = np.sum(preds == -1)
    print(f"[Isolation Forest] Detected {detected} anomalies (Should be ~10)")

# 2. DIMENSIONALITY REDUCTION (PCA)
# Business Use: Image compression, visualizing high-dim data.
def run_pca():
    # 50 features (dimensions)
    X = np.random.rand(100, 50) 
    
    # Squash to 2 dims
    pca = PCA(n_components=2)
    X_new = pca.fit_transform(X)
    
    variance = pca.explained_variance_ratio_
    print(f"[PCA] Reduced 50 dims -> 2 dims. Retained Variance: {sum(variance):.2f}")

# 3. DENSITY CLUSTERING (DBSCAN)
# Business Use: Geographic data (clustering GPS points).
# Logic: K-Means fails on moon-shapes. DBSCAN finds dense regions.
def run_dbscan():
    X = np.random.rand(100, 2)
    # eps=radius, min_samples=density threshold
    db = DBSCAN(eps=0.1, min_samples=5).fit(X)
    labels = db.labels_ # -1 means "Noise" (not in any cluster)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[DBSCAN] Found {n_clusters} dense clusters.")

if __name__ == "__main__":
    print("--- UNSUPERVISED & ANOMALY ---")
    run_anomaly_detection()
    run_pca()
    run_dbscan()
