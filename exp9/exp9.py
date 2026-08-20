"""
Experiment 9: Comparative Analysis of Statistical & Predictive Models on Diabetes Datasets

AIM:
To compare univariate statistics and regression predictive models (Linear Regression vs. Logistic Regression)
between the UCI Diabetes Dataset and the Pima Indians Diabetes Dataset.

SOFTWARE REQUIREMENTS:
- Python: Version 3.13.x
- Jupyter Notebook: Version 7.x
- Packages: pandas, numpy, scikit-learn, matplotlib, seaborn
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

def run_experiment_9():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uci_path = os.path.join(base_dir, "uci_diabetes.csv")
    pima_path = os.path.join(base_dir, "pima_diabetes.csv")

    uci_diabetes = pd.read_csv(uci_path)
    pima_columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    pima_diabetes = pd.read_csv(pima_path, header=None, names=pima_columns) if os.path.exists(pima_path) else pd.read_csv(pima_path)

    numerical_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

    print("--- 1. UNIVARIATE STATISTICAL COMPARISON ---")
    uci_stats = uci_diabetes[numerical_columns].describe().T
    pima_stats = pima_diabetes[numerical_columns].describe().T
    print("\nUCI Diabetes Dataset Statistics:")
    print(uci_stats[["count", "mean", "std", "min", "50%", "max"]])
    print("\nPima Indians Diabetes Dataset Statistics:")
    print(pima_stats[["count", "mean", "std", "min", "50%", "max"]])

    print("\n--- 2. REGRESSION MODEL PERFORMANCE COMPARISON ---")
    def get_linear_r2(df):
        X = df[["Glucose"]]
        y = df["BMI"]
        model = LinearRegression().fit(X, y)
        return r2_score(y, model.predict(X))

    def get_logistic_accuracy(df):
        features = ["Glucose", "BloodPressure", "BMI", "Age"]
        X = df[features]
        y = df["Outcome"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))

    uci_r2 = get_linear_r2(uci_diabetes)
    pima_r2 = get_linear_r2(pima_diabetes)
    uci_acc = get_logistic_accuracy(uci_diabetes)
    pima_acc = get_logistic_accuracy(pima_diabetes)

    print(f"Linear Regression (Glucose vs BMI) R2 - UCI: {uci_r2:.4f} | Pima: {pima_r2:.4f}")
    print(f"Logistic Regression (Outcome) Accuracy - UCI: {uci_acc:.4f} | Pima: {pima_acc:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    models = ['UCI Diabetes', 'Pima Diabetes']
    r2_scores = [uci_r2, pima_r2]
    acc_scores = [uci_acc, pima_acc]

    ax[0].bar(models, r2_scores, color=['#2b5c8f', '#e06666'], edgecolor='k', alpha=0.85)
    ax[0].set_title('Linear Regression R2 Score Comparison')
    ax[0].set_ylabel('R2 Score')
    ax[0].grid(True, linestyle='--', alpha=0.5)

    ax[1].bar(models, acc_scores, color=['#38761d', '#f6b26b'], edgecolor='k', alpha=0.85)
    ax[1].set_title('Logistic Regression Accuracy Comparison')
    ax[1].set_ylabel('Accuracy Score')
    ax[1].set_ylim(0, 1)
    ax[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(base_dir, "model_comparison.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print("Saved comparison plot to 'model_comparison.png'")
    plt.close()

if __name__ == "__main__":
    run_experiment_9()
