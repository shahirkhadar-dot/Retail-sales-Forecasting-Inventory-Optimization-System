# Retail-sales-Forecasting-Inventory-Optimization-System
This project uses machine learning and data analytics to forecast retail product demand and optimize inventory management. It generates synthetic retail data, performs feature engineering, trains a demand forecasting model, and calculates inventory recommendations to support smarter business decisions. 
"""
Retail Sales Forecasting & Inventory Optimization System
Pydroid 3 Mobile Execution Script
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("=" * 50)
print("1. GENERATING VIRTUAL RETAIL DATASET...")
print("=" * 50)

# Set random seed for reproducibility
np.random.seed(42)

# Create a date range for 1 year of daily data
dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")
stores = ["Store_A", "Store_B"]
items = ["Item_101", "Item_102"]

data_rows = []
for store in stores:
  for item in items:
    base_demand = np.random.randint(20, 50)
    for d in dates:
      # Simulate weekly seasonality and random noise
      dow_effect = 5 if d.dayofweek >= 5 else 0
      noise = np.random.normal(0, 5)
      qty_sold = max(
          0, int(base_demand + dow_effect + noise)
      )  # Ensure non-negative sales

      data_rows.append({
          "store_id": store,
          "item_id": item,
          "date": d,
          "qty_sold": qty_sold,
          "stockout_flag": 0,
      })

df = pd.DataFrame(data_rows)
print(f"Dataset generated successfully with {len(df)} records.\n")

print("=" * 50)
print("2. FEATURE ENGINEERING (Lags & Rolling Windows)...")
print("=" * 50)


def make_features(group_df):
  group_df = group_df.sort_values("date")
  # Lag features
  for lag in [1, 7]:
    group_df[f"lag_{lag}"] = group_df["qty_sold"].shift(lag)
  # Rolling window features
  for window in [7]:
    group_df[f"rollmean_{window}"] = (
        group_df["qty_sold"].shift(1).rolling(window).mean()
    )
  group_df["dow"] = group_df["date"].dt.dayofweek
  return group_df


df_features = (
    df.groupby(["store_id", "item_id"], group_keys=False)
    .apply(make_features)
    .dropna()
)
print(f"Features created. Clean dataset shape: {df_features.shape}\n")

print("=" * 50)
print("3. TRAINING MACHINE LEARNING FORECASTER...")
print("=" * 50)

features = [c for c in df_features.columns if c not in ["qty_sold", "date"]]
X = df_features[features]
y = df_features["qty_sold"]

# Train/Test split based on time
split_idx = int(len(df_features) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestRegressor(
    n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Model Training Complete! Test MAE: {mae:.2f}\n")

print("=" * 50)
print("4. INVENTORY OPTIMIZATION LOGIC (SS, ROP, EOQ)")
print("=" * 50)


def calculate_inventory_policy(
    forecast_mean,
    residual_std,
    on_hand,
    lead_time_days,
    service_level=0.95,
    annual_demand=10000,
    ordering_cost=500,
    unit_cost=100,
    holding_rate=0.2,
):
  z = norm.ppf(service_level)
  mu_L = forecast_mean * lead_time_days
  sigma_L = residual_std * np.sqrt(lead_time_days)

  safety_stock = z * sigma_L
  reorder_point = mu_L + safety_stock

  holding_cost_per_unit = unit_cost * holding_rate
  eoq = np.sqrt(
      (2 * annual_demand * ordering_cost) / holding_cost_per_unit
      if holding_cost_per_unit > 0
      else mu_L
  )

  order_quantity = max(0, max(eoq, reorder_point - on_hand))

  return {
      "Safety_Stock": round(safety_stock, 2),
      "Reorder_Point": round(reorder_point, 2),
      "EOQ": round(eoq, 2),
      "Recommended_Order_Qty": round(order_quantity, 2),
  }


# Calculate residual standard deviation as a proxy for demand uncertainty
resid_std = float(np.std(y_test - predictions))
avg_daily_forecast = float(np.mean(predictions))

# Execute optimization policy for a sample scenario
policy_result = calculate_inventory_policy(
    forecast_mean=avg_daily_forecast,
    residual_std=resid_std,
    on_hand=80,  # Current stock on hand
    lead_time_days=5,  # 5 days supplier lead time
)

print("Inventory Optimization Recommendations:")
print(policy_result)

print("=" * 50)
print("EXECUTION SUCCESSFUL")
print("=" * 50)
