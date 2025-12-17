# Data Science Pipeline: Employee Attrition Prediction

**Business Goal:** Reduce hiring costs by predicting which employees are about to quit.

## The Approach

1.  **Ingest:** Synthetic data representing employee demographics and satisfaction.
2.  **Clean (The Pipeline):**
    - Automatically handles missing salaries (Imputation).
    - Scales numeric features (StandardScaler) so Age (20-60) doesn't overshadow Satisfaction (1-10).
    - OneHotEncodes categorical data (Departments).
3.  **Model:** Random Forest Classifier.
4.  **Evaluate:** Classification Report (Precision/Recall).

## Why this structure?

Using `sklearn.pipeline.Pipeline` ensures no **Data Leakage** occurs. The scaling/imputation parameters are learned _only_ from the Train set and applied to the Test set.
