"""
Experiment 8: Multiple Linear Regression Analysis on Diabetes Datasets

AIM:
To perform Multiple Linear Regression on the UCI Diabetes Dataset and Pima Indians Diabetes Dataset
to predict a continuous target variable (BMI) using multiple independent predictor variables (Glucose,
BloodPressure, and Age), and evaluate model performance using R2 score and regression visualization.

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def run_experiment_8():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uci_path = os.path.join(base_dir, "uci_diabetes.csv")
    pima_path = os.path.join(base_dir, "pima_diabetes.csv")

    uci_diabetes = pd.read_csv(uci_path)
    pima_columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    pima_diabetes = pd.read_csv(pima_path, header=None, names=pima_columns) if os.path.exists(pima_path) else pd.read_csv(pima_path)

    print("UCI Diabetes Dataset Sample:")
    print(uci_diabetes.head())
    print("\nPima Indians Diabetes Dataset Sample:")
    print(pima_diabetes.head())

    features = ["Glucose", "BloodPressure", "Age"]
    target = "BMI"
    print(f"\nIndependent Predictors: {features}")
    print(f"Target Variable: {target}")

    def multiple_regression_analysis(df, dataset_name, save_filename=None):
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"\n{dataset_name} - Multiple Regression Evaluation:")
        print(f"R2 Score: {r2:.4f}")
        print(f"Intercept: {model.intercept_:.4f}")
        for feat, coef in zip(features, model.coef_):
            print(f"Coefficient ({feat}): {coef:.4f}")

        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, y_pred, color='royalblue', alpha=0.7, edgecolors='k', label='Predicted vs Actual')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='crimson', linestyle='--', linewidth=2, label='Perfect Fit Line')
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'Multiple Regression ({dataset_name}): Actual vs Predicted {target}\n(R2 = {r2:.4f})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        if save_filename:
            save_path = os.path.join(base_dir, save_filename)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved plot to '{save_filename}'")
        plt.close()

    multiple_regression_analysis(uci_diabetes, "UCI Diabetes Dataset", "uci_multiple_regression.png")
    multiple_regression_analysis(pima_diabetes, "Pima Indians Diabetes Dataset", "pima_multiple_regression.png")

if __name__ == "__main__":
    run_experiment_8()
