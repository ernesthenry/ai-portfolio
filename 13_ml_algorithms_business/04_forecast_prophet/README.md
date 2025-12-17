# Business Problem: Demand Forecasting (Time Series)

**Algorithm:** Exponential Smoothing / Prophet.

**The Question:** "How much stock should we buy?"
**The Value:** Inventory Optimization. Too much stock = storage cost. Too little stock = lost sales.

**Why Holt-Winters/Prophet?**
Standard regression fails here because "Time" matters. These models explicitly account for **Trend** (Sales are going up) and **Seasonality** (Sales spike every Christmas).
