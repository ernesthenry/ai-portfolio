from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# BUSINESS PROBLEM: PRODUCT RECOMMENDATION (Similar Items)
# ---------------------------------------------------------
# Scenario: E-commerce site. "Users who viewed this also viewed..."
# Goal: Find products with similar features.
# ---------------------------------------------------------

def run_recommender():
    # Mock Features: [Price, Battery_Life, Screen_Size, Is_Waterproof]
    # Products: Phones/Gadgets
    products = [
        "Phone A (Budget)", "Phone B (Flagship)", "Tablet X", "Watch Y", "Phone C (Mid)"
    ]
    
    # Normalized features (0-1 approx)
    features = np.array([
        [0.2, 0.5, 0.4, 0], # Phone A: Cheap, Med Battery, Small screen
        [0.9, 0.8, 0.6, 1], # Phone B: Exp, Good Battery, Med screen, Waterproof
        [0.6, 0.9, 0.9, 0], # Tablet X: Med Price, Great Battery, Big Screen
        [0.3, 0.2, 0.1, 1], # Watch Y: Cheap, Low Battery, Tiny Screen, Waterproof
        [0.5, 0.6, 0.5, 0]  # Phone C: Mid everything
    ])
    
    # ALGORITHM: K-NEAREST NEIGHBORS (KNN)
    # Why? It calculates geometric distance between feature vectors.
    knn = NearestNeighbors(n_neighbors=2, metric='euclidean')
    knn.fit(features)
    
    # User views "Phone C" (Index 4)
    query_index = 4
    distances, indices = knn.kneighbors([features[query_index]])
    
    print(f"--- KNN Product Recommender ---")
    print(f"User Viewing: {products[query_index]}")
    print("Recommendations:")
    
    for i in range(1, len(indices[0])): # Skip self
        idx = indices[0][i]
        dist = distances[0][i]
        print(f" - {products[idx]} (Distance: {dist:.4f})")
        
    print("\nInsight: KNN found the closest product in feature space.")

if __name__ == "__main__":
    run_recommender()
