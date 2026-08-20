"""
Experiment 11: Hypothesis Testing — One-Sample Z-Test on Diabetes Dataset

AIM:
To perform a One-Sample Z-Test on the Glucose attribute of the UCI Diabetes dataset to test
whether the sample mean significantly differs from a hypothesized population mean (mu0 = 100)
using statsmodels and visualize the Z-distribution with critical rejection boundaries.

SOFTWARE REQUIREMENTS:
- Python: Version 3.13.x
- Jupyter Notebook: Version 7.x
- Packages: pandas, numpy, statsmodels, scipy, matplotlib, seaborn
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.weightstats import ztest

def run_experiment_11():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uci_path = os.path.join(base_dir, "uci_diabetes.csv")
    uci_diabetes = pd.read_csv(uci_path)

    print("UCI Diabetes Dataset loaded successfully!")
    print(uci_diabetes.head())

    glucose = uci_diabetes["Glucose"].dropna()
    hypothesized_mean = 100
    sample_mean = glucose.mean()
    sample_std = glucose.std()
    n = len(glucose)

    print(f"\nSample Size (N): {n}")
    print(f"Sample Mean Glucose: {sample_mean:.4f}")
    print(f"Sample Std Dev: {sample_std:.4f}")
    print(f"Hypothesized Population Mean (mu0): {hypothesized_mean}")

    # Hypothesis Testing
    # H0: mu = 100 (Sample mean is equal to 100)
    # H1: mu != 100 (Sample mean differs significantly from 100)
    z_stat, p_value = ztest(glucose, value=hypothesized_mean)
    alpha = 0.05

    print("\n--- Z-TEST RESULTS ---")
    print(f"Z-Statistic: {z_stat:.4f}")
    print(f"P-Value: {p_value:.6f}")
    print(f"Significance Level (alpha): {alpha}")

    if p_value < alpha:
        print("Conclusion: Reject the Null Hypothesis (H0). The sample mean Glucose level differs significantly from 100.")
    else:
        print("Conclusion: Fail to Reject the Null Hypothesis (H0). There is no significant difference from 100.")

    # Visualization
    x = np.linspace(-7, 7, 500)
    y = norm.pdf(x, 0, 1)
    z_crit = norm.ppf(1 - alpha/2)

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, label='Standard Normal Distribution N(0, 1)', color='navy', lw=2)
    plt.fill_between(x, y, where=(x >= z_crit) | (x <= -z_crit), color='red', alpha=0.35, label=f'Rejection Region (alpha={alpha})')
    plt.axvline(z_stat, color='darkgreen', linestyle='--', lw=2.5, label=f'Observed Z-Stat = {z_stat:.2f}')
    plt.axvline(z_crit, color='red', linestyle=':', lw=1.5, label=f'+Critical Z = {z_crit:.2f}')
    plt.axvline(-z_crit, color='red', linestyle=':', lw=1.5, label=f'-Critical Z = -{z_crit:.2f}')
    plt.title('One-Sample Z-Test: Standard Normal Distribution & Rejection Boundaries')
    plt.xlabel('Z-Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    save_path = os.path.join(base_dir, "z_test_plot.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print("Saved Z-Test plot to 'z_test_plot.png'")
    plt.close()

if __name__ == "__main__":
    run_experiment_11()
