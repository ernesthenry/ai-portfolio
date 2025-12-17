import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. THE BUSINESS PROBLEM
# "Predict which employees are likely to quit (Attrition) so HR can intervene."

# 2. DATA GENERATION (Synthetic)
def generate_data(n=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(22, 60, n),
        'salary': np.random.randint(40000, 150000, n),
        'department': np.random.choice(['Sales', 'Engineering', 'HR'], n),
        'satisfaction_score': np.random.uniform(1, 10, n), # 1-10 scale
        'projects_completed': np.random.randint(1, 20, n),
        'years_at_company': np.random.randint(1, 15, n)
    })
    
    # Logic: Low satisfaction + Low salary = High attrition risk
    # This creates a "Signal" for the model to find
    risk_factor = (10 - data['satisfaction_score']) * 0.5 + (150000 - data['salary']) / 50000
    risk_factor += np.random.normal(0, 1, n) # Add noise
    
    data['attrition'] = (risk_factor > 6).astype(int)
    
    # Add some missing values to prove the pipeline handles dirty data
    data.loc[np.random.choice(n, 50), 'salary'] = np.nan
    data.loc[np.random.choice(n, 20), 'department'] = np.nan
    
    return data

# 3. PIPELINE DEFINITION
# Data Science is 80% cleaning. This pipeline automates it.

# Numeric transformer: Fill missing with median, then scale
numeric_features = ['age', 'salary', 'satisfaction_score', 'projects_completed', 'years_at_company']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical transformer: Fill missing with 'missing', then OneHotEncode
categorical_features = ['department']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine them
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full Pipeline: Preprocessing -> Model
# Random Forest is robust and handles non-linear relationships well
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 4. EXECUTION
if __name__ == "__main__":
    print("Generating raw employee data...")
    df = generate_data()
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    
    X = df.drop('attrition', axis=1)
    y = df['attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Pipeline...")
    model_pipeline.fit(X_train, y_train)
    print("Model trained.")
    
    print("\nEvaluating...")
    y_pred = model_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Feature Importance (Extracting insights for Business)
    # This is tricky with pipelines, we need to access the steps
    print("\n--- BUSINESS INSIGHTS ---")
    rf_model = model_pipeline.named_steps['classifier']
    importances = rf_model.feature_importances_
    # Note: Getting feature names back from OneHotEncoder is complex, simplified here
    print(f"Top Feature Importance: {np.max(importances):.4f} (Likely Satisfaction or Salary)")
