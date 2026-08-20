"""
Experiment 13: Analysis of Variance (One-Way ANOVA) on Diabetes Datasets

AIM:
To perform One-Way ANOVA using scipy.stats.f_oneway to determine whether there are statistically
significant differences in continuous health metrics (Glucose and BMI) across different Age group categories.

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
from scipy.stats import f_oneway

def run_experiment_13():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uci_path = os.path.join(base_dir, "uci_diabetes.csv")
    pima_path = os.path.join(base_dir, "pima_diabetes.csv")

    uci_diabetes = pd.read_csv(uci_path)
    pima_columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    pima_diabetes = pd.read_csv(pima_path, header=None, names=pima_columns) if os.path.exists(pima_path) else pd.read_csv(pima_path)

    # Bin Age into 3 Groups: Young (20-30), Middle (31-50), Senior (51+)
    bins = [20, 30, 50, 100]
    labels = ["Young (20-30)", "Middle-Aged (31-50)", "Senior (51+)"]

    uci_diabetes["AgeGroup"] = pd.cut(uci_diabetes["Age"], bins=bins, labels=labels)
    pima_diabetes["AgeGroup"] = pd.cut(pima_diabetes["Age"], bins=bins, labels=labels)

    def perform_anova(df, metric, group_col, dataset_name):
        groups = [df[df[group_col] == label][metric].dropna() for label in labels]
        f_stat, p_val = f_oneway(*groups)
        print(f"\n{dataset_name} - One-Way ANOVA for {metric} across Age Groups:")
        for label, grp in zip(labels, groups):
            print(f"  {label} (N={len(grp)}): Mean = {grp.mean():.2f}")
        print(f"F-Statistic: {f_stat:.4f} | P-Value: {p_val:.6f}")
        if p_val < 0.05:
            print("Conclusion: Statistically significant difference between age group means (p < 0.05).")
        else:
            print("Conclusion: No statistically significant difference between age group means (p >= 0.05).")
        return f_stat, p_val

    perform_anova(uci_diabetes, "Glucose", "AgeGroup", "UCI Diabetes")
    perform_anova(pima_diabetes, "Glucose", "AgeGroup", "Pima Indians Diabetes")

    # Plot ANOVA Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=uci_diabetes, x="AgeGroup", y="Glucose", ax=axes[0], palette="Set2")
    axes[0].set_title("UCI Diabetes: Glucose across Age Groups (ANOVA)")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    sns.boxplot(data=pima_diabetes, x="AgeGroup", y="Glucose", ax=axes[1], palette="Set2")
    axes[1].set_title("Pima Indians: Glucose across Age Groups (ANOVA)")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(base_dir, "anova_analysis.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print("Saved ANOVA plot to 'anova_analysis.png'")
    plt.close()

if __name__ == "__main__":
    run_experiment_13()
