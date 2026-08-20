"""
Experiment 16: Time Series Decomposition and Forecasting using ARIMA on Sequential Health Data

AIM:
To perform Time Series Analysis on sequential health records, decomposing time-series trends and seasonal
components using statsmodels and fitting an ARIMA forecasting model to predict future glucose readings.

SOFTWARE REQUIREMENTS:
- Python: Version 3.13.x
- Jupyter Notebook: Version 7.x
- Packages: pandas, numpy, statsmodels, matplotlib, seaborn
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

def run_experiment_16():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pima_path = os.path.join(base_dir, "pima_diabetes.csv")

    pima_columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    pima_diabetes = pd.read_csv(pima_path, header=None, names=pima_columns) if os.path.exists(pima_path) else pd.read_csv(pima_path)

    # Construct sequential daily date index
    glucose_ts = pima_diabetes["Glucose"].iloc[:200].reset_index(drop=True)
    dates = pd.date_range(start="2024-01-01", periods=len(glucose_ts), freq="D")
    ts_data = pd.Series(glucose_ts.values, index=dates, name="Glucose")

    print("Sequential Time Series Data Sample:")
    print(ts_data.head())

    # 1. Seasonal Decomposition
    decomp = seasonal_decompose(ts_data, model='additive', period=7)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    decomp.observed.plot(ax=axes[0], color='blue', title='Observed Glucose Series')
    decomp.trend.plot(ax=axes[1], color='red', title='Trend Component')
    decomp.seasonal.plot(ax=axes[2], color='green', title='Seasonal (Weekly) Component')
    decomp.resid.plot(ax=axes[3], color='purple', title='Residual Error Component')
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path1 = os.path.join(base_dir, "time_series_decomposition.png")
    plt.savefig(save_path1, bbox_inches='tight', dpi=300)
    print("Saved decomposition plot to 'time_series_decomposition.png'")
    plt.close()

    # 2. ARIMA Model Fitting & Forecasting
    train_ts = ts_data.iloc[:-20]
    test_ts = ts_data.iloc[-20:]

    arima_model = ARIMA(train_ts, order=(1, 1, 1)).fit()
    forecast = arima_model.forecast(steps=20)
    forecast.index = test_ts.index

    print("\nARIMA (1,1,1) Model Fitted Successfully!")
    print(f"Akaike Information Criterion (AIC): {arima_model.aic:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_ts.index[-50:], train_ts.iloc[-50:], label='Historical Glucose Readings', color='navy')
    plt.plot(test_ts.index, test_ts, label='Actual Test Values', color='black', marker='o')
    plt.plot(forecast.index, forecast, label='ARIMA (1,1,1) Forecast', color='crimson', linestyle='--', marker='x', lw=2)
    plt.title('ARIMA Time Series Forecasting on Sequential Glucose Measurements')
    plt.xlabel('Date')
    plt.ylabel('Glucose Level')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    save_path2 = os.path.join(base_dir, "arima_forecast.png")
    plt.savefig(save_path2, bbox_inches='tight', dpi=300)
    print("Saved ARIMA forecast plot to 'arima_forecast.png'")
    plt.close()

if __name__ == "__main__":
    run_experiment_16()
