# 🧠 Machine Learning for Business: The Complete Cheatsheet

This repository implements **all standard Machine Learning algorithms** mapped directly to real-world business problems. Each script is a self-contained "Case Study".

## 📋 Algorithm Index

### 🔹 Supervised Learning (Prediction)

| Algorithm                    | Type           | Business Use Case                                                              | Project Path                                       |
| :--------------------------- | :------------- | :----------------------------------------------------------------------------- | :------------------------------------------------- |
| **Linear Regression**        | Regression     | **Ad Spend Optimization**: Predicting sales revenue based on marketing budget. | `05_predictive_analytics/ad_spend_linear.py`       |
| **Logistic Regression**      | Classification | **Lead Scoring**: Calculating the probability a lead converts to a sale.       | `05_predictive_analytics/lead_scoring_logistic.py` |
| **Decision Tree**            | Classification | **Loan Explanation**: White-box rules for why a loan was rejected.             | `06_risk_evaluation/loan_risk_forest.py`           |
| **Random Forest**            | Ensemble       | **Credit Risk Engine**: High-accuracy default prediction.                      | `06_risk_evaluation/loan_risk_forest.py`           |
| **XGBoost (Gradient Boost)** | Ensemble       | **Churn Prediction**: Identifying users likely to cancel.                      | `01_churn_xgboost`                                 |
| **SVM**                      | Classification | **High-Value Customer ID**: Complex boundary separation for premium status.    | `06_risk_evaluation/customer_svm.py`               |
| **KNN**                      | Instance-Based | **Product Recommendation**: "Find similar products" engine.                    | `07_text_similarity/product_recommender_knn.py`    |
| **Naive Bayes**              | Probabilistic  | **Support Ticket Routing**: Classifying email as Billing/Tech/General.         | `07_text_similarity/support_ticket_bayes.py`       |
| **CNN**                      | Deep Learning  | **Quality Control**: Detecting defects in manufacturing visual data.           | `08_computer_vision_ops/quality_control_cnn.py`    |
| **RNN**                      | Deep Learning  | **Server Load Forecasting**: Predicting infrastructure demand.                 | `09_sequence_prediction/server_load_rnn.py`        |
| **Transformer**              | Deep Learning  | **Language Understanding**: (See root project `01_transformer_from_scratch`)   | `../01_transformer_from_scratch`                   |

### 🔸 Unsupervised Learning (Structure Discovery)

| Algorithm                   | Type           | Business Use Case                                                   | Project Path                                    |
| :-------------------------- | :------------- | :------------------------------------------------------------------ | :---------------------------------------------- |
| **K-Means**                 | Clustering     | **Customer Segmentation**: Grouping users by purchasing behavior.   | `02_segmentation_kmeans`                        |
| **Hierarchical Clustering** | Clustering     | **Behavior Taxonomy**: Building a user type tree.                   | `10_advanced_clustering/advanced_clustering.py` |
| **DBSCAN**                  | Clustering     | **Store Location Planning**: Finding geographic density hotspots.   | `10_advanced_clustering/advanced_clustering.py` |
| **PCA**                     | Dim. Reduction | **Data Compression**: Visualizing high-dimensional user data.       | `10_advanced_clustering/advanced_clustering.py` |
| **Autoencoders**            | Deep Learning  | **Anomaly Detection**: Predictive maintenance for factory machines. | `08_computer_vision_ops/anomaly_autoencoder.py` |
| **SVD**                     | Matrix Factor  | **Movie Recommendation**: Latent factor analysis.                   | `03_recommendation_svd`                         |

## 🚀 How to Run

1. Navigate to this directory:
   ```bash
   cd 13_ml_algorithms_business
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   _(Note: Ensure torch, scikit-learn, pandas, numpy are installed)_

3. Run any specific business case:
   ```bash
   python 05_predictive_analytics/ad_spend_linear.py
   python 06_risk_evaluation/loan_risk_forest.py
   python 08_computer_vision_ops/anomaly_autoencoder.py
   ```
