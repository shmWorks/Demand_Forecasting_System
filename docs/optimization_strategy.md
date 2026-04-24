# 🚀 1000x Speedup Strategy: Transitioning to JAX + Polars

The intuition is correct: **JAX + Polars** is the "holy grail" stack for 2026 data science pipelines. While NumPy and Pandas are industry standards, they are fundamentally limited by single-threaded execution and eager evaluation. Transitioning to a **Rust-based (Polars)** and **XLA-compiled (JAX)** architecture can realistically achieve 100x-1000x speedups in specific hot-paths.

## 📊 The "1000x" Breakdown
| Component | Current (Pandas/NumPy) | Target (Polars/JAX) | Speedup | Why? |
| :--- | :--- | :--- | :--- | :--- |
| **I/O** | `pd.read_csv` | `pl.read_parquet` | ~10x-20x | Arrow-native, multi-threaded I/O. |
| **Preprocessing** | Eager/Single-threaded | `LazyFrame` / Parallel | ~15x-50x | Query optimization & Rust core. |
| **Features** | `apply()` / `groupby` | `Expressions` / `over()` | ~50x-200x | SIMD & Avoidance of Python overhead. |
| **Training** | NumPy Loops / CPU | **JAX JIT / GPU** | **~1000x** | XLA compilation & Massive parallelism. |

---

## 🛠 Step 1: Polars for the Data Layer
Replace the slow, eager Pandas logic with Polars **Lazy API**.

### ❌ Before (Pandas)
```python
# src/retail_iq/preprocessing.py
df = pd.read_csv('train.csv')
df['sales_lag_7d'] = df.groupby(['store_nbr', 'family'])['sales'].shift(7)
# This is slow because it materializes each step and runs on a single core.
```

### ✅ After (Polars)
```python
import polars as pl

def load_and_preprocess():
    return (
        pl.scan_parquet('train.parquet')  # 1. Lazy Scan
        .with_columns([
            pl.col('sales').shift(7).over(['store_nbr', 'family']).alias('sales_lag_7d') # 2. Parallel GroupBy
        ])
        .collect() # 3. Execute optimized query plan
    )
```

---

## 🏗 Step 2: Optimizing Feature Engineering
The current `apply()` for holiday distances is the biggest bottleneck in your `features.py`.

### ❌ Before (O(N*M))
```python
self.df['days_to_nearest_holiday'] = self.df['date'].apply(
    lambda d: min([abs((d - h).days) for h in holiday_dates])
)
```

### ✅ After (O(N log N))
In Polars, we can use `join_asof` to find the nearest date in a sorted holiday list, which is near-instant.
```python
holidays = pl.from_pandas(holiday_dates).sort('date')
df = df.join_asof(holidays, on='date', strategy='nearest')
```

---

## 🧠 Step 3: JAX for High-Performance Modeling
Your `GD_Linear` in `models.py` is a prime candidate for JAX.

### ❌ Before (NumPy)
```python
# Single-threaded CPU
grad = (2/m) * (X.T @ errors) + (self.l2/m) * self.theta
self.theta -= self.lr * grad
```

### ✅ After (JAX + JIT)
```python
import jax
import jax.numpy as jnp

@jax.jit  # Compiles to optimized XLA kernels (GPU/TPU)
def update_step(theta, X, y, lr, l2):
    preds = X @ theta
    errors = preds - y
    # Auto-diff handles gradients for complex models
    grad = (2/m) * (X.T @ errors) + (l2/m) * theta
    return theta - lr * grad
```

---

## 🏁 How to reach 1000x?
1. **GPU Acceleration**: JAX *must* run on a CUDA-enabled GPU. CPU-only JAX is only ~2-5x faster than NumPy.
2. **Parquet Transition**: CSV is a legacy format. Use `.parquet` with Snappy/Zstd compression for 10x faster loads.
3. **Lazy Evaluation**: Always use `LazyFrame` in Polars to allow "Predicate Pushdown" (filtering data *before* it's even read into memory).
4. **Vectorization**: Remove ALL `.apply()` and Python loops. If it can't be done in Polars expressions, use JAX `vmap` to parallelize it.

---
> [!IMPORTANT]
> **Migration Risk**: Polars uses **1-based** indexing and is **Immutable** by default. JAX is **Functional** (no side effects). You will need to rewrite the core logic, not just swap imports.
