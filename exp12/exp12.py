"""
Experiment 12: Hypothesis Testing — Independent Two-Sample T-Test

AIM:
To conduct Independent Two-Sample T-Tests using scipy.stats.ttest_ind to compare clinical feature
means (Glucose, BMI, Age) between Diabetic (Outcome=1) and Non-Diabetic (Outcome=0) groups on Diabetes datasets.

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
from scipy.stats import ttest_ind

def run_experiment_12():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uci_path = os.path.join(base_dir, "uci_diabetes.csv")
    pima_path = os.path.join(base_dir, "pima_diabetes.csv")

    uci_diabetes = pd.read_csv(uci_path)
    pima_columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    pima_diabetes = pd.read_csv(pima_path, header=None, names=pima_columns) if os.path.exists(pima_path) else pd.read_csv(pima_path)

    print("Datasets loaded successfully!")

    def perform_ttest(df, feature, group_col, dataset_name):
        group0 = df[df[group_col] == 0][feature].dropna()
        group1 = df[df[group_col] == 1][feature].dropna()
        t_stat, p_val = ttest_ind(group0, group1, equal_var=False)
        print(f"\n{dataset_name} - T-Test for {feature} (Outcome 0 vs 1):")
        print(f"Non-Diabetic Mean (N={len(group0)}): {group0.mean():.2f}")
        print(f"Diabetic Mean (N={len(group1)}): {group1.mean():.2f}")
        print(f"T-Statistic: {t_stat:.4f} | P-Value: {p_val:.6f}")
        if p_val < 0.05:
            print("Statistically Significant Difference (p < 0.05)")
        else:
            print("No Statistically Significant Difference (p >= 0.05)")
        return t_stat, p_val

    perform_ttest(uci_diabetes, "Glucose", "Outcome", "UCI Diabetes")
    perform_ttest(uci_diabetes, "BMI", "Outcome", "UCI Diabetes")
    perform_ttest(pima_diabetes, "Glucose", "Outcome", "Pima Indians Diabetes")
    perform_ttest(pima_diabetes, "BMI", "Outcome", "Pima Indians Diabetes")

    # Plot comparisons
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=uci_diabetes, x='Outcome', y='Glucose', ax=axes[0], palette=['#4a90e2', '#e94e77'])
    axes[0].set_title('UCI Diabetes: Glucose by Outcome (T-Test)')
    axes[0].set_xticklabels(['Non-Diabetic (0)', 'Diabetic (1)'])
    axes[0].grid(True, linestyle='--', alpha=0.5)

    sns.boxplot(data=pima_diabetes, x='Outcome', y='Glucose', ax=axes[1], palette=['#4a90e2', '#e94e77'])
    axes[1].set_title('Pima Indians: Glucose by Outcome (T-Test)')
    axes[1].set_xticklabels(['Non-Diabetic (0)', 'Diabetic (1)'])
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(base_dir, "t_test_comparison.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print("Saved T-Test plot to 't_test_comparison.png'")
    plt.close()

if __name__ == "__main__":
    run_experiment_12()
