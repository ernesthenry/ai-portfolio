import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# BUSINESS PROBLEM: LEAD SCORING
# ---------------------------------------------------------
# Scenario: Sales teams have too many leads. They waste time calling
# people who won't buy.
# Goal: Predict the probability (0 to 1) that a lead assumes a paid subscription.
# ---------------------------------------------------------

def generate_lead_data(n_samples=1000):
    np.random.seed(42)
    # Features
    web_visits = np.random.poisson(5, n_samples) # How many times they visited site
    time_on_site_mins = np.random.normal(10, 5, n_samples)
    whitepaper_download = np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) # Binary
    
    # Logic: More time + Download = Higher Chance
    score = (0.3 * web_visits) + (0.2 * time_on_site_mins) + (2.0 * whitepaper_download) - 4
    prob = 1 / (1 + np.exp(-score)) # Sigmoid
    
    converted = [1 if np.random.rand() < p else 0 for p in prob]
    
    data = pd.DataFrame({
        'Web_Visits': web_visits,
        'Time_On_Site': time_on_site_mins,
        'Whitepaper_Download': whitepaper_download,
        'Converted': converted
    })
    return data

def run_logistic_regression(data):
    X = data[['Web_Visits', 'Time_On_Site', 'Whitepaper_Download']]
    y = data['Converted']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ALGORITHM: LOGISTIC REGRESSION
    # Why? We need a PROBABILITY output (0-100%), not just a label.
    # It gives us the "Odds" of conversion.
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    print(f"--- Logistic Regression (Lead Scoring) ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nFeature Importance (Log Odds):")
    cols = ['Web_Visits', 'Time_On_Site', 'Whitepaper_Download']
    for col, coef in zip(cols, model.coef_[0]):
        print(f"{col}: {coef:.4f} (Pos = Increases Likelihood)")
        
    # Simulation
    sample_lead = [[8, 15, 1]] # High usage, downloaded paper
    prob = model.predict_proba(sample_lead)[0][1]
    print(f"\nSimulation: Lead with 8 visits, 15 mins, Downloaded Paper")
    print(f"Conversion Probability: {prob:.2%}")

if __name__ == "__main__":
    df = generate_lead_data()
    run_logistic_regression(df)
