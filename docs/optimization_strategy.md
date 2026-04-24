# 🚀 1000x Speedup Strategy: Transitioning to JAX + Polars

Intuition correct. JAX + Polars = holy grail for 2026 data pipelines. NumPy/Pandas limited: single-threaded, eager evaluation. Rust (Polars) + XLA (JAX) = 100x-1000x speedups in hot-paths.

## 📊 The "1000x" Breakdown

| Component | Current (Pandas/NumPy) | Target (Polars/JAX) | Speedup | Why? |
| :--- | :--- | :--- | :--- | :--- |
| **I/O** | `pd.read_csv` | `pl.read_parquet` | ~10x-20x | Arrow-native, multi-threaded I/O. |
| **Preprocessing** | Eager/Single-threaded | `LazyFrame` / Parallel | ~15x-50x | Query optimization & Rust core. |
| **Features** | `apply()` / `groupby` | `Expressions` / `over()` | ~50x-200x | SIMD & Avoidance of Python overhead. |
| **Training** | NumPy Loops / CPU | **JAX JIT / GPU** | **~1000x** | XLA compilation & Massive parallelism. |

---

## 🛠 Step 1: Polars for the Data Layer
Replace eager Pandas with Polars **Lazy API**.

### 👎 Before (Pandas)

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

## 🔥 Step 2: Optimizing Feature Engineering
Current `apply()` for holiday distances = biggest bottleneck in `features.py`.

### 👎 Before (O(N*M))

```python
self.df['days_to_nearest_holiday'] = self.df['date'].apply(
    lambda d: min([abs((d - h).days) for h in holiday_dates])
)
```

### ✅ After (O(N log N))

Use `join_asof` for nearest holiday lookup — near-instant.

```python
holidays = pl.from_pandas(holiday_dates).sort('date')
df = df.join_asof(holidays, on='date', strategy='nearest')
```

---

## 🧠 Step 3: JAX for High-Performance Modeling
`GD_Linear` in `models.py` = prime JAX candidate.

### 👎 Before (NumPy)

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

## 💪 How to reach 1000x?

1. **GPU Acceleration**: JAX needs CUDA GPU. CPU JAX = only 2-5x faster than NumPy.
2. **Parquet Transition**: CSV = legacy. Use `.parquet` with Snappy/Zstd = 10x faster loads.
3. **Lazy Evaluation**: Use `LazyFrame`. Enables Predicate Pushdown — filter before data enters memory.
4. **Vectorization**: Remove ALL `.apply()` and Python loops. If Polars expressions can't handle it, use JAX `vmap`.

---

> [!IMPORTANT]
> **Migration Risk**: Polars uses 1-based indexing, immutable by default. JAX = functional (no side effects). Rewrite core logic — not just swap imports.