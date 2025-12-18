from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# BUSINESS PROBLEM: GEOSPATIAL MARKETING & TAXONOMY
# ---------------------------------------------------------
# Scenario 1 (DBSCAN): Identifying high-density areas for new Coffee Shops.
# Scenario 2 (Hierarchical): Understanding User Behavior relationships.
# Scenario 3 (PCA): Visualizing high-dimensional User Data.
# ---------------------------------------------------------

def run_clustering_suite():
    np.random.seed(42)
    
    print("--- 1. DBSCAN (Store Location Planning) ---")
    # Feature: [Latitude, Longitude]
    # We generate dense blobs (Neighborhoods) and random scattered noise (Rural)
    neighborhood_1 = np.random.normal(0, 0.5, (50, 2))
    neighborhood_2 = np.random.normal(5, 0.5, (50, 2))
    noise = np.random.uniform(-5, 10, (10, 2))
    
    X = np.vstack([neighborhood_1, neighborhood_2, noise])
    
    # Logic: Points must be within 'eps' distance to be neighbors. 
    # High density regions form clusters.
    db = DBSCAN(eps=0.8, min_samples=5).fit(X)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"Found {n_clusters} potential distinct neighborhoods for store opening.")
    print(f"Noise points (outliers/rural): {list(db.labels_).count(-1)}")
    
    print("\n--- 2. HIERARCHICAL CLUSTERING (User Taxonomy) ---")
    # Hierarchical builds a tree. Good for seeing 'Sub-groups'.
    # e.g., 'Power Users' -> 'Gamers' vs 'Developers'
    hc = AgglomerativeClustering(n_clusters=2).fit(neighborhood_1) # Just using one blob
    print(f"Hierarchical Labels: {hc.labels_[:10]} (Tree-based grouping)")
    
    print("\n--- 3. PCA (Information Compression) ---")
    # Scenario: We have 50 features describing a user, we want 2D plot.
    X_high_dim = np.random.rand(100, 50)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_high_dim)
    
    print(f"Compressed {X_high_dim.shape[1]} features down to {X_reduced.shape[1]}.")
    print(f"Retained Variance: {np.sum(pca.explained_variance_ratio_):.2f}")

if __name__ == "__main__":
    run_clustering_suite()
