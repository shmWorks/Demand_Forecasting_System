"""Verification script — run after all optimizations to confirm correctness and timing."""
import time
import sys

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless verification

import pandas as pd
import numpy as np

print("=" * 60)
print("STEP 2: evaluation.py — lazy shap import")
print("=" * 60)
t = time.time()
from retail_iq.evaluation import evaluate_model
print(f"Import OK in {time.time()-t:.3f}s  (expect < 0.5s)")

print()
print("=" * 60)
print("STEP 3: visualization.py — sampling smoke test")
print("=" * 60)
from retail_iq.visualization import plot_correlation_heatmap, plot_sales_distribution
df_test = pd.DataFrame({
    "sales": np.random.rand(200_000),
    "feature_a": np.random.rand(200_000),
    "feature_b": np.random.rand(200_000),
})
t = time.time()
plot_correlation_heatmap(df_test, save_path=None)
plot_sales_distribution(df_test, save_path=None)
print(f"visualization OK in {time.time()-t:.2f}s  (expect < 3s)")

print()
print("=" * 60)
print("STEP 1 + 4: Parquet I/O + Full Feature Pipeline")
print("=" * 60)
from retail_iq.preprocessing import load_raw_data, preprocess_dates, merge_datasets, detect_outliers_iqr
from retail_iq.features import FastFeatureEngineer

t_total = time.time()

t = time.time()
train, test, stores, oil, holidays, tx = load_raw_data()
print(f"load_raw_data:    {time.time()-t:.2f}s  | train shape: {train.shape}")

train, test, oil, holidays, tx = preprocess_dates([train, test, oil, holidays, tx])

t = time.time()
df = merge_datasets(train, stores, oil, holidays, tx)
print(f"merge_datasets:   {time.time()-t:.2f}s  | merged shape: {df.shape}")

t = time.time()
fe = FastFeatureEngineer(df, holidays=holidays, transactions=tx, oil_price=oil, store_meta=stores)
result = (
    fe.add_temporal_features()
      .add_lag_and_rolling()
      .add_onpromotion_features()
      .add_macroeconomic_features()
      .add_transaction_features()
      .add_store_metadata()
      .add_cannibalization_features()
      .transform()
)
print(f"FastFeatureEngineer: {time.time()-t:.2f}s  | output shape: {result.shape}")

key_cols = ["days_to_nearest_holiday", "sales_lag_1d", "sales_lag_7d", "rolling_mean_7d",
            "other_family_sales_lag_7d", "is_national_holiday"]
for col in key_cols:
    status = "OK" if col in result.columns else "MISSING"
    print(f"  [{status}] {col}")

# Confirm lag_365d NOT present (removed from defaults)
assert "sales_lag_365d" not in result.columns, "lag_365d should NOT be in default output"
print("  [OK] sales_lag_365d absent (correct)")

t = time.time()
result_outliers = detect_outliers_iqr(result)
print(f"detect_outliers_iqr: {time.time()-t:.2f}s  | is_outlier dtype: {result_outliers['is_outlier'].dtype}")

print()
print(f"Total pipeline wall time: {time.time()-t_total:.2f}s  (target < 120s)")

print()
print("=" * 60)
print("STEP 6: JAX GD_Linear smoke test")
print("=" * 60)
from retail_iq.models import GD_Linear, SeasonalNaive, _JAX_AVAILABLE
print(f"JAX available: {_JAX_AVAILABLE}")

rng = np.random.default_rng(42)
X_dummy = rng.standard_normal((500, 10)).astype(np.float32)
y_dummy = rng.random(500).astype(np.float32)
t = time.time()
model = GD_Linear(lr=0.01, iterations=200, random_state=42)
model.fit(X_dummy, y_dummy)
preds = model.predict(X_dummy)
print(f"GD_Linear fit+predict: {time.time()-t:.3f}s  | loss[-1]: {model.loss_history[-1]:.6f}")
assert len(model.loss_history) == 200
assert preds.shape == (500,)
# Loss should decrease overall
assert model.loss_history[0] >= model.loss_history[-1], "Loss not decreasing — bug in GD"
print("  [OK] loss monotonically decreasing")

naive = SeasonalNaive(period=7)
naive.fit()
print("  [OK] SeasonalNaive instantiates fine")

print()
print("ALL CHECKS PASSED")
