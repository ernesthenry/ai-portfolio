import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# BUSINESS PROBLEM: LOAN DEFAULT RISK
# ---------------------------------------------------------
# Scenario: A Bank needs to approve/deny loans automatically.
# Constraint: Use Random Forest for accuracy, but Decision Tree for explainability 
# if a customer asks "Why was I rejected?".
# ---------------------------------------------------------

def generate_loan_data(n_samples=2000):
    np.random.seed(42)
    income = np.random.normal(50000, 15000, n_samples)
    debt = np.random.normal(20000, 10000, n_samples)
    credit_score = np.random.normal(650, 100, n_samples)
    
    # Logic: High Debt + Low Income + Low Score = Default
    risk_score = (debt / (income + 1)) * 5 - (credit_score / 200)
    default = [1 if r > 0.5 + np.random.normal(0, 0.2) else 0 for r in risk_score]
    
    data = pd.DataFrame({
        'Income': income,
        'Debt': debt,
        'Credit_Score': credit_score,
        'Defaulted': default
    })
    return data

def run_risk_models(data):
    X = data[['Income', 'Debt', 'Credit_Score']]
    y = data['Defaulted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. RANDOM FOREST (The Engine)
    # Why? Ensemble of trees reduces variance and overfitting. High Accuracy.
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    
    print(f"--- Random Forest (Production Model) ---")
    print(f"Accuracy: {rf_acc:.4f}")
    
    # 2. DECISION TREE (The Explainer)
    # Why? A single tree is easy to visualize as IF-ELSE rules.
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    
    print(f"\n--- Decision Tree (Explainability Layer) ---")
    print("Rules extracted for Loan Officers:")
    print(export_text(dt, feature_names=['Income', 'Debt', 'Credit_Score']))

if __name__ == "__main__":
    df = generate_loan_data()
    run_risk_models(df)
