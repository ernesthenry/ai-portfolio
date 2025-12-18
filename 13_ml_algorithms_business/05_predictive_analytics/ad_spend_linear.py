import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------
# BUSINESS PROBLEM: AD SPEND OPTIMIZATION
# ---------------------------------------------------------
# Scenario: A Marketing Team wants to know the ROI of TV, Radio, 
# and Social Media ads on Sales.
# Goal: Build a model to predict Sales based on Ad Spend.
# ---------------------------------------------------------

def generate_marketing_data(n_samples=1000):
    np.random.seed(42)
    # Features: Spend in thousands
    tv_spend = np.random.normal(200, 50, n_samples)
    radio_spend = np.random.normal(50, 20, n_samples)
    social_spend = np.random.normal(100, 30, n_samples)
    
    # Target: Sales (Linear relationship + noise)
    # Logic: TV has high impact, Radio medium, Social low but steady?
    sales = 5.0 + (0.05 * tv_spend) + (0.1 * radio_spend) + (0.02 * social_spend) + np.random.normal(0, 2, n_samples)
    
    data = pd.DataFrame({
        'TV_Spend': tv_spend, 
        'Radio_Spend': radio_spend, 
        'Social_Spend': social_spend,
        'Sales_Revenue': sales
    })
    return data

def run_linear_regression(data):
    X = data[['TV_Spend', 'Radio_Spend', 'Social_Spend']]
    y = data['Sales_Revenue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ALGORITHM: LINEAR REGRESSION
    # Why? We assume a direct addictive relationship between spend and sales.
    # It provides "Coefficients" which tell us exactly the ROI per dollar.
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"--- Linear Regression Results ---")
    print(f"Mean Absolute Error: ${mae*1000:.2f} (Average prediction error)")
    print(f"R2 Score: {r2:.4f} (Variance explained)")
    print(f"\nModel Insights (ROI):")
    print(f"Base Sales (Intercept): ${model.intercept_*1000:.2f}")
    print(f"TV ROI: ${model.coef_[0]*1000:.2f} per $1k spend")
    print(f"Radio ROI: ${model.coef_[1]*1000:.2f} per $1k spend")
    print(f"Social ROI: ${model.coef_[2]*1000:.2f} per $1k spend")

if __name__ == "__main__":
    df = generate_marketing_data()
    print("Data Preview:")
    print(df.head())
    print("\nRunning Analysis...")
    run_linear_regression(df)
