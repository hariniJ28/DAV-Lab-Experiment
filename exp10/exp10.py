"""
Experiment 10: Probability Distributions and Normal Curve Fitting on Diabetes Data

AIM:
To plot empirical feature distributions and fit theoretical Normal (Gaussian) Probability Density Function (PDF)
curves over continuous features of the UCI Diabetes Dataset using SciPy and Matplotlib.

SOFTWARE REQUIREMENTS:
- Python: Version 3.13.x
- Jupyter Notebook: Version 7.x
- Packages: pandas, numpy, scipy, matplotlib, seaborn
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def run_experiment_10():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uci_path = os.path.join(base_dir, "uci_diabetes.csv")
    uci_diabetes = pd.read_csv(uci_path)

    print("UCI Diabetes Dataset loaded successfully!")
    print(uci_diabetes.head())

    glucose = uci_diabetes["Glucose"].dropna()
    mu, std = glucose.mean(), glucose.std()
    print(f"\nGlucose Statistics -> Mean (mu): {mu:.2f}, Std Dev (sigma): {std:.2f}")

    plt.figure(figsize=(8, 5))
    sns.histplot(glucose, kde=True, stat="density", color="skyblue", edgecolor="black", label="Empirical Histogram & KDE")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r-', linewidth=2.5, label=f'Fitted Normal PDF (mu={mu:.1f}, sigma={std:.1f})')
    plt.title('UCI Diabetes: Glucose Distribution with Fitted Normal Curve')
    plt.xlabel('Glucose Level')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    save_path1 = os.path.join(base_dir, "normal_curve_glucose.png")
    plt.savefig(save_path1, bbox_inches='tight', dpi=300)
    print("Saved single normal curve to 'normal_curve_glucose.png'")
    plt.close()

    features = ["Glucose", "BloodPressure", "BMI", "Age"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, col in enumerate(features):
        data = uci_diabetes[col].dropna()
        m, s = data.mean(), data.std()
        sns.histplot(data, kde=True, stat="density", ax=axes[i], color="lightgreen", edgecolor="black")
        x_vals = np.linspace(data.min(), data.max(), 100)
        axes[i].plot(x_vals, norm.pdf(x_vals, m, s), 'crimson', lw=2, label=f'Normal (mu={m:.1f}, sigma={s:.1f})')
        axes[i].set_title(f'Normal Curve: {col}')
        axes[i].set_xlabel(col)
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path2 = os.path.join(base_dir, "normal_curves_features.png")
    plt.savefig(save_path2, bbox_inches='tight', dpi=300)
    print("Saved multi-feature normal curves to 'normal_curves_features.png'")
    plt.close()

if __name__ == "__main__":
    run_experiment_10()
