"""
Experiment 14: Comprehensive Linear Regression Modeling and Error Diagnostics

AIM:
To build and evaluate linear regression models with train-test splitting, computing comprehensive
performance metrics (R2, MSE, RMSE, MAE) and analyzing residual distributions on diabetes datasets.

SOFTWARE REQUIREMENTS:
- Python: Version 3.13.x
- Jupyter Notebook: Version 7.x
- Packages: pandas, numpy, scikit-learn, matplotlib, seaborn
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def run_experiment_14():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uci_path = os.path.join(base_dir, "uci_diabetes.csv")
    pima_path = os.path.join(base_dir, "pima_diabetes.csv")

    uci_diabetes = pd.read_csv(uci_path)
    pima_columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    pima_diabetes = pd.read_csv(pima_path, header=None, names=pima_columns) if os.path.exists(pima_path) else pd.read_csv(pima_path)

    def linear_diagnostics(df, x_col, y_col, dataset_name):
        X = df[[x_col]]
        y = df[y_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"\n{dataset_name} - Linear Regression Diagnostics ({x_col} -> {y_col}):")
        print(f"R2 Score: {r2:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        return model, X_test, y_test, y_pred, residuals

    m, X_te, y_te, y_pred, residuals = linear_diagnostics(pima_diabetes, "Glucose", "BMI", "Pima Indians Diabetes")

    # Plot 1: Model Fit
    plt.figure(figsize=(7, 5))
    plt.scatter(X_te, y_te, color='dodgerblue', alpha=0.7, edgecolors='k', label='Actual Test Data')
    plt.plot(X_te, y_pred, color='red', linewidth=2, label='Fitted Line')
    plt.xlabel('Glucose')
    plt.ylabel('BMI')
    plt.title('Linear Regression Model Fit: Glucose vs BMI')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    save_path1 = os.path.join(base_dir, "linear_model_fit.png")
    plt.savefig(save_path1, bbox_inches='tight', dpi=300)
    print("Saved fit plot to 'linear_model_fit.png'")
    plt.close()

    # Plot 2: Residuals Diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_pred, residuals, color='purple', alpha=0.7, edgecolors='k')
    axes[0].axhline(0, color='crimson', linestyle='--', lw=2)
    axes[0].set_title('Residuals vs Fitted Values')
    axes[0].set_xlabel('Fitted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    sns.histplot(residuals, kde=True, ax=axes[1], color='teal', edgecolor='k')
    axes[1].set_title('Residuals Error Distribution')
    axes[1].set_xlabel('Residual Error')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path2 = os.path.join(base_dir, "residuals_plot.png")
    plt.savefig(save_path2, bbox_inches='tight', dpi=300)
    print("Saved residuals plot to 'residuals_plot.png'")
    plt.close()

if __name__ == "__main__":
    run_experiment_14()
