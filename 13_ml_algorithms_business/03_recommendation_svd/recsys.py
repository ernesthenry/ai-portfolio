import numpy as np
from scipy.sparse.linalg import svds

# BUSINESS PROBLEM: Recommendation System
# "Users who bought X also bought Y."

# 1. USER-ITEM MATRIX
# Rows = Users, Cols = Items (Movies)
# Values = Ratings (0 means not seen)
ratings_matrix = np.array([
    [5, 4, 0, 0], # User 1 likes Action
    [4, 5, 0, 0], # User 2 likes Action
    [0, 0, 5, 4], # User 3 likes Romance
    [0, 0, 4, 5], # User 4 likes Romance
]).astype(float)

# 2. SVD (Singular Value Decomposition)
# Decomposes matrix into U, Sigma, Vt
# This is "Matrix Factorization" - finding hidden features (latent factors)
U, sigma, Vt = svds(ratings_matrix, k=2)
sigma = np.diag(sigma)

predicted_ratings = np.dot(np.dot(U, sigma), Vt)

print("Original:\n", ratings_matrix)
print("Predicted (Reconstructed):\n", np.round(predicted_ratings, 1))
# The "0" values would be filled with predictions in a real sparse matrix scenario
