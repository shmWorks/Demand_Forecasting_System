"""Data loading, cleaning, and merging pipeline for Retail-IQ.

All I/O uses config.py path constants — never hardcode paths.
Parquet is preferred over CSV when available (10-20x faster, multi-threaded).
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_raw_data() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Load all raw datasets.  Prefers Parquet when available; falls back to CSV.

    Returns:
        Tuple of (train, test, stores, oil, holidays, transactions) as DataFrames.

    Note:
        Run ``scripts/convert_to_parquet.py`` once to enable the fast Parquet path.
        Parquet I/O is 10-20x faster than CSV due to columnar binary format and
        multi-threaded decompression via Polars / PyArrow.
    """
    from retail_iq.config import RAW_DATA_DIR, PARQUET_DATA_DIR

    def _load(name: str, date_col: str | None = "date") -> pd.DataFrame:
        """Load one file, preferring Parquet over CSV."""
        parquet_path = PARQUET_DATA_DIR / f"{name}.parquet"
        csv_path = RAW_DATA_DIR / f"{name}.csv"

        if parquet_path.exists():
            try:
                import polars as pl
                df = pl.read_parquet(parquet_path).to_pandas()
                if date_col and date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col])
                logger.debug("Loaded %s from Parquet (%d rows)", name, len(df))
                return df
            except Exception as exc:  # noqa: BLE001
                logger.warning("Parquet load failed for %s (%s); falling back to CSV", name, exc)

        parse_dates = [date_col] if date_col else False
        df = pd.read_csv(csv_path, parse_dates=parse_dates)
        logger.debug("Loaded %s from CSV (%d rows)", name, len(df))
        return df

    train        = _load("train")
    test         = _load("test")
    stores       = _load("stores", date_col=None)
    oil          = _load("oil")
    holidays     = _load("holidays_events")
    transactions = _load("transactions")

    return train, test, stores, oil, holidays, transactions


def preprocess_dates(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Ensure 'date' column is datetime64 for every DataFrame that has one.

    Args:
        dfs: List of DataFrames (any order, any may lack 'date' column).

    Returns:
        Same list with date columns cast to datetime64[ns].
    """
    result = []
    for df in dfs:
        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df = df.copy()
            raw = df["date"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            invalid_count = int(df["date"].isna().sum() - raw.isna().sum())
            if invalid_count > 0:
                logger.warning("preprocess_dates: coerced %d invalid date values to NaT", invalid_count)
        result.append(df)
    return result


# def clean_oil_prices(oil_df: pd.DataFrame) -> pd.DataFrame:
#     """Sort by date and forward-fill then backward-fill missing oil prices.

#     Args:
#         oil_df: Raw oil DataFrame with 'date' and 'dcoilwtico' columns.

#     Returns:
#         Cleaned copy with no missing oil prices.
#     """
#     df = oil_df.copy()
#     df = df.sort_values("date").reset_index(drop=True)
#     df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
#     return df

def clean_oil_prices(oil_df: pd.DataFrame) -> pd.DataFrame:
    """Clean oil prices without introducing future data leakage."""

    df = oil_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Step 1: forward fill (safe - uses past only)
    df["dcoilwtico"] = df["dcoilwtico"].ffill()

    # Step 2: fill remaining NaNs (usually at start) with global mean
    global_mean = df["dcoilwtico"].mean()
    df["dcoilwtico"] = df["dcoilwtico"].fillna(global_mean)

    return df


def merge_datasets(
    train: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all source tables into one modelling-ready DataFrame.

    Operations (in order):
        1. Left-join stores metadata on store_nbr.
        2. Left-join cleaned oil prices on date.
        3. Left-join transactions on (store_nbr, date); forward-fill per store.
        4. Flag national (non-transferred) holidays as is_national_holiday.
        5. Sort by [store_nbr, family, date] for downstream shift/rolling ops.

    Args:
        train:        Raw training DataFrame (3M+ rows).
        stores:       Store metadata (54 stores).
        oil:          Daily oil prices with gaps.
        holidays:     Holiday events with locale and transferred flag.
        transactions: Daily transaction counts per store.

    Returns:
        Merged DataFrame sorted by [store_nbr, family, date].
    """
    df = train.copy()

    # 1. Store metadata
    df = df.merge(stores, on="store_nbr", how="left")

    # 2. Oil prices (clean before merge)
    oil_clean = clean_oil_prices(oil)
    df = df.merge(oil_clean, on="date", how="left")

    # 3. Transactions — ffill per store to cover missing days
    df = df.merge(transactions, on=["store_nbr", "date"], how="left")
    df["transactions"] = df.groupby("store_nbr")["transactions"].ffill()

    # 4. National holiday flag (exclude transferred holidays per SPEC constraint 6)
    active_holidays = holidays[holidays["transferred"] == False].copy()  # noqa: E712
    national_dates = set(active_holidays.loc[active_holidays["locale"] == "National", "date"])
    df["is_national_holiday"] = df["date"].isin(national_dates).astype(np.int8)

    # 5. Sort — downstream feature engineering relies on this order
    sort_cols = ["store_nbr", "family", "date"] if "family" in df.columns else ["store_nbr", "date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def detect_outliers_iqr(
    df: pd.DataFrame,
    group_cols: List[str] | None = None,
    iqr_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Flag sales outliers per group using a vectorized IQR method.

    Uses groupby().transform() — O(N) — not apply(), which is O(N × groups).
    Outlier condition: sales > Q3 + iqr_multiplier * IQR.

    Args:
        df:             DataFrame containing 'sales' and group columns.
        group_cols:     Columns to group by. Defaults to ['store_nbr', 'family'].
        iqr_multiplier: Multiplier for IQR fence. Default 3.0 (conservative).

    Returns:
        df copy with boolean 'is_outlier' column appended.
    """
    df = df.copy()
    if "sales" not in df.columns:
        df["is_outlier"] = False
        return df

    group_cols = group_cols or ["store_nbr", "family"]
    grouped = df.groupby(group_cols)["sales"]

    q1 = grouped.transform("quantile", 0.25)
    q3 = grouped.transform("quantile", 0.75)
    iqr = q3 - q1

    df["is_outlier"] = df["sales"] > (q3 + iqr_multiplier * iqr)
    return df


def strict_temporal_holdout_split(
    df: pd.DataFrame,
    date_col: str = "date",
    holdout_days: int = 15,
    end_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by strict calendar holdout window with no temporal leakage.

    Test window is inclusive [end_date - (holdout_days - 1), end_date].
    Train contains rows strictly before test_start.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    if holdout_days <= 0:
        raise ValueError("holdout_days must be > 0")

    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    if data[date_col].isna().all():
        raise ValueError("All date values are invalid/NaT; cannot split.")

    split_end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(data[date_col].max())
    split_start = split_end - pd.Timedelta(days=holdout_days - 1)

    train_df = data[data[date_col] < split_start]
    test_df = data[(data[date_col] >= split_start) & (data[date_col] <= split_end)]

    if train_df.empty:
        raise ValueError("Temporal holdout produced empty train split.")
    if test_df.empty:
        raise ValueError("Temporal holdout produced empty test split.")
    if train_df[date_col].max() >= test_df[date_col].min():
        raise ValueError("Temporal leakage detected: train max date overlaps test window.")

    return train_df, test_df
