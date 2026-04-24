"""Performance regression harness for Phase 3 optimizations.

Run manually:
    python tests/perf_regression_harness.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from retail_iq.features import FastFeatureEngineer
from retail_iq.perf_utils import (
    load_feature_cache_parquet,
    optimize_dtypes_zero_copy,
    save_feature_cache_parquet,
)


def _old_notebook_object_encoding(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object" or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].astype(str).astype("category").cat.codes
    return out


def benchmark_dtype_pipeline(
    *,
    min_mem_gain_pct: float = 10.0,
    max_time_overhead_pct: float = 25.0,
) -> dict[str, float]:
    rng = np.random.default_rng(42)
    n = 250_000
    df = pd.DataFrame(
        {
            "store_nbr": rng.integers(1, 55, size=n, dtype=np.int64),
            "family": rng.choice(["A", "B", "C", "D", "E", "F"], size=n),
            "city": rng.choice(["Quito", "Guayaquil", "Cuenca"], size=n),
            "sales": rng.lognormal(mean=2.1, sigma=0.7, size=n).astype(np.float64),
            "onpromotion": rng.integers(0, 2, size=n, dtype=np.int64),
        }
    )

    t0 = time.perf_counter()
    old_df = _old_notebook_object_encoding(df)
    old_t = time.perf_counter() - t0
    old_mem = float(old_df.memory_usage(deep=True).sum())

    t0 = time.perf_counter()
    new_df = optimize_dtypes_zero_copy(df.copy(), exclude_cols=())
    new_t = time.perf_counter() - t0
    new_mem = float(new_df.memory_usage(deep=True).sum())

    time_gain = (old_t - new_t) / old_t * 100.0
    mem_gain = (old_mem - new_mem) / old_mem * 100.0

    if time_gain < -max_time_overhead_pct:
        raise AssertionError(
            f"dtype pipeline overhead too high: {time_gain:.2f}% < -{max_time_overhead_pct:.2f}%"
        )
    if mem_gain < min_mem_gain_pct:
        raise AssertionError(
            f"dtype pipeline memory gain below target: {mem_gain:.2f}% < {min_mem_gain_pct:.2f}%"
        )

    return {
        "old_time_s": old_t,
        "new_time_s": new_t,
        "time_gain_pct": time_gain,
        "old_mem_mb": old_mem / (1024 * 1024),
        "new_mem_mb": new_mem / (1024 * 1024),
        "mem_gain_pct": mem_gain,
    }


def _build_synthetic_features(rows_per_group: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    stores = np.arange(1, 26)
    families = [f"f{i}" for i in range(10)]
    dates = pd.date_range("2017-01-01", periods=rows_per_group, freq="D")
    idx = pd.MultiIndex.from_product(
        [stores, families, dates], names=["store_nbr", "family", "date"]
    )
    df = idx.to_frame(index=False)
    df["sales"] = rng.lognormal(mean=2.0, sigma=0.8, size=len(df)).astype(np.float32)
    df["onpromotion"] = rng.integers(0, 2, size=len(df), dtype=np.int8)

    feat = (
        FastFeatureEngineer(df)
        .add_lag_and_rolling(lags=[1, 7, 14], windows=[7, 14])
        .add_onpromotion_features()
        .transform()
        .fillna(0)
    )
    return optimize_dtypes_zero_copy(feat, exclude_cols=["date"])


def benchmark_cache_load(min_gain_pct: float = 10.0) -> dict[str, float]:
    cache_path = Path("data/processed/perf_harness_feature_cache.parquet")

    t0 = time.perf_counter()
    built_df = _build_synthetic_features()
    build_t = time.perf_counter() - t0
    save_feature_cache_parquet(built_df, cache_path)

    t0 = time.perf_counter()
    loaded_df = load_feature_cache_parquet(cache_path, use_mmap=True)
    load_t = time.perf_counter() - t0

    if loaded_df.shape != built_df.shape:
        raise AssertionError("cache load shape mismatch")

    gain = (build_t - load_t) / build_t * 100.0
    if gain < min_gain_pct:
        raise AssertionError(f"cache load gain below target: {gain:.2f}% < {min_gain_pct:.2f}%")

    return {
        "build_time_s": build_t,
        "load_time_s": load_t,
        "time_gain_pct": gain,
        "cache_size_mb": cache_path.stat().st_size / (1024 * 1024),
    }


if __name__ == "__main__":
    dtype_stats = benchmark_dtype_pipeline(min_mem_gain_pct=10.0, max_time_overhead_pct=25.0)
    cache_stats = benchmark_cache_load(min_gain_pct=10.0)
    print("dtype_pipeline:", dtype_stats)
    print("cache_load:", cache_stats)
    print("PERF_HARNESS_PASS")
