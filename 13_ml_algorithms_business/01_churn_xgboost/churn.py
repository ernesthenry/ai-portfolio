import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# BUSINESS PROBLEM: Customer Churn
# Identify customers who are about to cancel so we can send them a discount coupon.

# 1. GENERATE DATA
# Features: Usage Minutes, Contract Duration, Monthly Bill
df = pd.DataFrame({
    'usage_minutes': np.random.normal(100, 50, 1000),
    'contract_months': np.random.choice([1, 12, 24], 1000),
    'monthly_bill': np.random.normal(50, 10, 1000),
})
# Logic: Short contract + Low usage = Churn
df['churn'] = (df['contract_months'] == 1).astype(int) & (df['usage_minutes'] < 80).astype(int)

# 2. TRAIN XGBOOST
# XGBoost is the "King of Tabular Data". It uses Gradient Boosted Decision Trees.
X_train, X_test, y_train, y_test = train_test_split(df.drop('churn', axis=1), df['churn'])

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 3. EXPLAIN
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test))}")
print("Feature Importance:")
for name, score in zip(df.columns, model.feature_importances_):
    print(f"- {name}: {score:.4f}")
