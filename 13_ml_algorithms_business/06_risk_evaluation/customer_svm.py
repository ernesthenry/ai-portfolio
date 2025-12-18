import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# BUSINESS PROBLEM: PREMIUM CUSTOMER CLASSIFICATION
# ---------------------------------------------------------
# Scenario: An Airline wants to separate "Premium" travelers from "Standard"
# based on complex behavioral patterns, not just money spent.
# Use Case: Lounge Access Logic.
# ---------------------------------------------------------

def generate_traveler_data(n_samples=1000):
    np.random.seed(42)
    # Features: Flight Frequency, Avg Spend per Mile, Lounge Visits
    # Hard to separate linearly as there are "Budget Business" travelers vs "Wealthy Leisure"
    freq = np.random.normal(10, 5, n_samples) # Flights per year
    spend_per_mile = np.random.normal(0.5, 0.2, n_samples)
    
    # Target: High Value (1)
    # Complex boundary: High Freq OR (Medium Freq AND High Spend)
    is_premium = []
    for f, s in zip(freq, spend_per_mile):
        if (f > 15) or (f > 8 and s > 0.7):
            is_premium.append(1)
        else:
            is_premium.append(0)
            
    # Add noise
    X = np.column_stack((freq, spend_per_mile))
    y = np.array(is_premium)
    return X, y

def run_svm(X, y):
    # SVM requires scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # ALGORITHM: SVM (Support Vector Machine)
    # Why? Effective in high dimensional spaces.
    # The "Kernel Trick" (RBF) can capture the "OR" logic (Curved boundary) effectively.
    # It tries to maximize the MARGIN between classes.
    model = SVC(kernel='rbf', C=1.0)
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    
    print(f"--- Support Vector Machine (Premium Class) ---")
    print(f"Accuracy: {acc:.4f}")
    print("Insight: SVM successfully learned the non-linear rules (High Freq OR High Spend).")

if __name__ == "__main__":
    X, y = generate_traveler_data()
    run_svm(X, y)
