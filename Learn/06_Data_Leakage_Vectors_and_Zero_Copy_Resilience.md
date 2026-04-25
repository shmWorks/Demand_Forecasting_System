# 06. Data Leakage Vectors and Zero-Copy Resilience (Agentic Exploration)

## 1. Intuition: The Silent Killers of ML Systems

In time-series forecasting, **Data Leakage** occurs when information from the future implicitly influences the prediction of the present. This leads to models that show phenomenal validation scores but fail catastrophically in production.

**Zero-Copy Resilience** refers to the ability of a system to transform and load data without making unnecessary, memory-bloating copies in RAM. In Python/Pandas, accidental copies are the primary cause of Out-Of-Memory (OOM) errors at scale.

## 2. Implementation & Forensic Critique: The Leaks

An audit of the `preprocessing.py` and `features.py` files reveals several critical vectors for both leakage and memory bloat.

### Leakage Vector 1: The Global Fill Forward
In `src/retail_iq/preprocessing.py`, missing oil prices are imputed:
```python
def clean_oil_prices(oil_df: pd.DataFrame) -> pd.DataFrame:
    df = oil_df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
    return df
```
[Source: `src/retail_iq/preprocessing.py:73`]

**The Flaw:** `.bfill()` (Backward Fill) is a cardinal sin in time-series forecasting. It takes a *future* known price and copies it backwards in time to fill a missing *past* price. When predicting sales on Day $T$, the model is secretly looking at the oil price from Day $T+1$ or $T+2$. This is textbook temporal data leakage.

### Leakage Vector 2: Grouping Leakage in Transactions
In `merge_datasets`:
```python
df["transactions"] = df.groupby("store_nbr")["transactions"].ffill()
```
[Source: `src/retail_iq/preprocessing.py:116`]
**The Flaw:** This ffill is executed *after* merging all train/test data together. The test data (future horizon) does not have transactions (as they haven't happened yet). If there's a gap at the boundary between train and test, the last known transaction count from the training set bleeds endlessly into the test set.

### Memory Bloat: The `perf_utils.py` Illusion
`src/retail_iq/perf_utils.py` provides `optimize_dtypes_zero_copy`, attempting to downcast types `in-place` using `copy=False`.
```python
df[col] = df[col].astype(np.int32, copy=False)
```
[Source: `src/retail_iq/perf_utils.py:27`]

**The Flaw:** Pandas 2.x and PyArrow semantics have heavily changed. Often, `astype` with `copy=False` still creates a copy if the memory layout isn't perfectly contiguous or if a view cannot be established. Furthermore, the `FastFeatureEngineer.transform()` explicitly calls `.copy()`, doubling memory usage right before model training.

## 3. Sovereign Extension: The Hardened Pipeline

We must patch the leaks and enforce true memory efficiency without requiring a complete rewrite to Polars or PySpark.

### Step-by-Step Actionable Insights

*   **Insight 1 (Eradicate Backfill):** Remove all `.bfill()` operations from the codebase. If an oil price is missing at the start of the dataset, it should be dropped or filled with a global static mean calculated *only* from a distinct warmup period.
*   **Insight 2 (Strict Temporal Barriers):** All imputation (`ffill`, rolling means) must be executed *strictly within* the bounds of the training set. Create a custom Scikit-learn style Transformer that learns the last known state of the training set and explicitly applies it to the test set, preventing boundary bleeding.
*   **Insight 3 (Polars LazyFrames):** For true zero-copy resilience on datasets exceeding 50M rows, transition the core `FastFeatureEngineer` logic from eager Pandas DataFrames to Polars `LazyFrame`s. Polars builds a query plan and executes it in a highly parallelized, memory-efficient streaming manner, vastly outperforming Pandas while completely eliminating intermediate memory copies.