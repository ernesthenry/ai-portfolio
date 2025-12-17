import pandas as pd
import numpy as np
# Note: Prophet is heavy, so we simulate simple Forecasting logic here or use statsmodels
# We'll use a simple statsmodels Holt-Winters for standard python compatibility
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# BUSINESS PROBLEM: Demand Forecasting
# "How much inventory do we need next month?"

# 1. GENERATE TIME SERIES (Trend + Seasonality)
dates = pd.date_range(start='2023-01-01', periods=24, freq='M')
values = [100 + i*5 + (20 if i%12==11 else 0) for i in range(24)] # Growing trend + Spike in Dec
data = pd.Series(values, index=dates)

# 2. FORECASTING MODEL (Holt-Winters)
# Handles Trend (add) and Seasonality (add, 12 months)
model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12).fit()

# 3. PREDICT FUTURE
forecast = model.forecast(6) # Next 6 months

print("Historical Data (Last 5):\n", data.tail())
print("\nForecast (Next 6 Months):\n", forecast)
# Should see the trend continue
