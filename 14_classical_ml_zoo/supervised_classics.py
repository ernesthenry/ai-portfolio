from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# GENERATE DATA
# We make specific datasets where certain models shine vs fail
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_complex, y_complex = make_moons(n_samples=500, noise=0.3, random_state=42) # Non-linear data

def run_svm():
    # SVM: Great for high-dimensional margins. "Kernel Trick" allows separating non-linear data.
    model = SVC(kernel='rbf') 
    model.fit(X_complex, y_complex)
    acc = accuracy_score(y_complex, model.predict(X_complex))
    print(f"[SVM] RBF Kernel Accuracy on Moons: {acc:.4f} (Can capture curves)")

def run_knn():
    # KNN: "Birds of a feather flock together". Simple, lazy learning.
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    print(f"[KNN] Accuracy: {acc:.4f} (Good baseline)")

def run_naive_bayes():
    # Naive Bayes: "Assume feature independence". Fast, great for text/real-time.
    model = GaussianNB()
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    print(f"[Naive Bayes] Accuracy: {acc:.4f} (Super fast)")

def run_decision_tree():
    # Tree: "If-Else rules". Highly interpretability.
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)
    print(f"[Decision Tree] Feature Importance: {model.feature_importances_[:3]}... (Explainable)")

if __name__ == "__main__":
    print("--- SUPERVISED CLASSICS ---")
    run_svm()
    run_knn()
    run_naive_bayes()
    run_decision_tree()
