# Time Series Exercise 9: The Box–Jenkins Model (ARIMA)

## Name: Aravindan T

## Roll No: 2305001003

## Department: CSE

## AIM

To understand and implement the Box-Jenkins approach to time-series forecasting using an ARIMA model in Python.

## PROGRAM

### Python Code

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Sales data
sales = [
    100, 105, 103, 110, 115, 112, 120, 125,
    123, 130, 135, 132, 140, 145, 143, 150,
    155, 152, 160, 165, 163, 170, 175, 172
]

data = pd.Series(sales, name="Sales")

print("STEP 1: Original Sales Data")
print(data.to_string(index=False))

# Plot original sales data
plt.figure(figsize=(10, 4))
plt.plot(data, marker="o")
plt.title("Monthly Sales Data")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# ADF test on original data
adf_result = adfuller(data)

print("\nSTEP 2: ADF Test on Original Data")
print("ADF Statistic:", round(adf_result[0], 4))
print("p-value:", round(adf_result[1], 4))

if adf_result[1] < 0.05:
    print("Conclusion: Evidence of stationarity.")
else:
    print("Conclusion: Insufficient evidence of stationarity.")

# First differencing
diff_data = data.diff().dropna()

print("\nSTEP 3: First-Differenced Data")
print(diff_data.to_string(index=False))

# Plot differenced data
plt.figure(figsize=(10, 4))
plt.plot(diff_data, marker="o")
plt.title("First-Differenced Sales")
plt.xlabel("Month")
plt.ylabel("Change in Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# ADF test after differencing
adf_diff = adfuller(diff_data)

print("\nSTEP 4: ADF Test After Differencing")
print("ADF Statistic:", round(adf_diff[0], 4))
print("p-value:", round(adf_diff[1], 4))

if adf_diff[1] < 0.05:
    print("Conclusion: Evidence of stationarity.")
else:
    print("Conclusion: Stationarity is not confirmed.")

# ACF and PACF plots
print("\nSTEP 5: ACF and PACF Plots")

plot_acf(diff_data, lags=10)
plt.tight_layout()
plt.show()

plot_pacf(diff_data, lags=10, method="ywm")
plt.tight_layout()
plt.show()

# Fit ARIMA(1,1,1) model
model = ARIMA(data, order=(1, 1, 1))
fitted_model = model.fit()

print("\nSTEP 6: ARIMA Model Summary")
print(fitted_model.summary())

# Model selection criteria
print("\nSTEP 7: Model Selection Criteria")
print("AIC:", round(fitted_model.aic, 4))
print("BIC:", round(fitted_model.bic, 4))

# Residual analysis
residuals = fitted_model.resid

plt.figure(figsize=(10, 4))
plt.plot(residuals, marker="o")
plt.axhline(0, color="red", linestyle="--")
plt.title("ARIMA Residuals")
plt.xlabel("Month")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual ACF
print("\nSTEP 8: Residual ACF")
plot_acf(residuals, lags=10)
plt.tight_layout()
plt.show()

# Ljung-Box test
lb_result = acorr_ljungbox(
    residuals,
    lags=[5],
    return_df=True
)

print("\nLjung-Box Test Result")
print(lb_result)

if lb_result["lb_pvalue"].iloc[0] > 0.05:
    print("No significant residual autocorrelation detected at lag 5.")
else:
    print("Residual autocorrelation may remain; review the model.")

# Forecast next six months
forecast = fitted_model.get_forecast(steps=6)

forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

print("\nSTEP 9: Forecast for the Next Six Months")

for month, value in enumerate(
    forecast_mean,
    start=len(data) + 1
):
    print(f"Month {month}: {value:.2f}")

print("\nForecast Confidence Intervals")
print(forecast_ci.round(2))

# Plot forecast with confidence intervals
future_months = range(len(data), len(data) + 6)

plt.figure(figsize=(10, 5))

plt.plot(
    range(len(data)),
    data,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    future_months,
    forecast_mean.to_numpy(),
    marker="o",
    linestyle="--",
    label="Forecast"
)

plt.fill_between(
    future_months,
    forecast_ci.iloc[:, 0].to_numpy(),
    forecast_ci.iloc[:, 1].to_numpy(),
    alpha=0.2,
    label="95% Confidence Interval"
)

plt.title("Box-Jenkins ARIMA: Six-Month Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Final result
print("\nRESULT")
print("Box-Jenkins ARIMA modelling completed successfully.")
```

## OUTPUT
<img width="769" height="331" alt="image" src="https://github.com/user-attachments/assets/e6126919-64cb-44cf-96d9-4955e0a8d6ca" />
<img width="790" height="646" alt="image" src="https://github.com/user-attachments/assets/d883c08f-33ec-47bb-8054-7ac7af3f366c" />
<img width="790" height="1001" alt="image" src="https://github.com/user-attachments/assets/fc73d1c2-8558-49c5-8e14-f2c9bbed3d73" />
<img width="940" height="371" alt="image" src="https://github.com/user-attachments/assets/3c94d688-3b4e-4599-8994-84817800f71f" />
<img width="940" height="700" alt="image" src="https://github.com/user-attachments/assets/f46df326-0217-4691-844f-8e592c80ede4" />
<img width="940" height="700" alt="image" src="https://github.com/user-attachments/assets/8f3d6604-b349-42d0-bd4c-b4ac3602a5d2" />
<img width="909" height="420" alt="image" src="https://github.com/user-attachments/assets/02e2fad0-687e-4771-aa3c-92157baedc0e" />

## RESULT

The Box–Jenkins approach was successfully implemented using an ARIMA model in Python to analyze time-series data and forecast future values.
