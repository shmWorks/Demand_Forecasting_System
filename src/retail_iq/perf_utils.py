"""Performance helpers for cache I/O and low-copy dataframe preparation."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def optimize_dtypes_zero_copy(
    df: pd.DataFrame,
    *,
    exclude_cols: Iterable[str] = (),
) -> pd.DataFrame:
    """Encode object/string columns and downcast numeric columns in-place."""
    excluded = set(exclude_cols)

    obj_cols = [
        c
        for c in df.columns
        if c not in excluded and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))
    ]
    for col in obj_cols:
        codes, _ = pd.factorize(df[col], sort=False, use_na_sentinel=True)
        df[col] = codes.astype(np.int32, copy=False)

    int_cols = [c for c in df.select_dtypes(include=["int64"]).columns if c not in excluded]
    for col in int_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if np.iinfo(np.int16).min <= col_min and col_max <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16, copy=False)
        elif np.iinfo(np.int32).min <= col_min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32, copy=False)

    float_cols = [c for c in df.select_dtypes(include=["float64"]).columns if c not in excluded]
    for col in float_cols:
        df[col] = df[col].astype(np.float32, copy=False)

    return df


def save_feature_cache_parquet(df: pd.DataFrame, cache_path: Path | str) -> Path:
    """Save feature DataFrame to Parquet (safe, non-pickle serialization)."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    return path


def load_feature_cache_parquet(
    cache_path: Path | str,
    *,
    columns: Sequence[str] | None = None,
    use_mmap: bool = True,
) -> pd.DataFrame:
    """Load Parquet cache; mmap path avoids extra buffered copy when possible."""
    path = Path(cache_path)
    table = pq.read_table(path, columns=list(columns) if columns else None, memory_map=use_mmap)
    return table.to_pandas()


def load_or_build_feature_cache(
    cache_path: Path | str,
    build_fn,
    *,
    use_mmap: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Load cached features when present; else build once then cache."""
    path = Path(cache_path)
    if path.exists():
        return load_feature_cache_parquet(path, use_mmap=use_mmap), True

    df = build_fn()
    save_feature_cache_parquet(df, path)
    return df, False


def benchmark_cache_load(
    cache_path: Path | str,
    *,
    repeats: int = 3,
    use_mmap: bool = True,
) -> dict[str, float]:
    """Benchmark load time for cached parquet feature frame."""
    path = Path(cache_path)
    if repeats <= 0:
        raise ValueError("repeats must be > 0")

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = load_feature_cache_parquet(path, use_mmap=use_mmap)
        times.append(time.perf_counter() - t0)

    return {
        "load_median_s": float(np.median(times)),
        "load_min_s": float(np.min(times)),
        "load_max_s": float(np.max(times)),
        "repeats": float(repeats),
        "file_size_mb": float(path.stat().st_size / (1024 * 1024)),
    }
