"""Feature engineering pipeline for Retail-IQ."""

"""✅ Sort invariant enforcement (critical)
✅ Fixed store_type dtype bug (operator precedence)
✅ No overengineering
✅ Same structure, just safer"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_SORT_COLS = ["store_nbr", "family", "date"]


class FastFeatureEngineer:

    def __init__(
        self,
        df: pd.DataFrame,
        transactions: pd.DataFrame | None = None,
        oil_price: pd.DataFrame | None = None,
        holidays: pd.DataFrame | None = None,
        store_meta: pd.DataFrame | None = None,
    ) -> None:
        # Sort ONCE here (core invariant)
        self.df = df.sort_values(_SORT_COLS).reset_index(drop=True)
        self.transactions = transactions
        self.oil_price = oil_price
        self.holidays = holidays
        self.store_meta = store_meta

    # -----------------------------
    # 🔒 Invariant check
    # -----------------------------
    def _assert_sorted(self) -> None:
        idx = self.df.set_index(_SORT_COLS).index
        if not idx.is_monotonic_increasing:
            raise ValueError(
                "DataFrame is not sorted by ['store_nbr', 'family', 'date']. "
                "Lag/rolling features would be incorrect."
            )

    # -----------------------------
    # Features
    # -----------------------------

    def add_temporal_features(self) -> "FastFeatureEngineer":
        d = self.df["date"]
        self.df["day_of_week"] = d.dt.dayofweek
        self.df["day_of_month"] = d.dt.day
        self.df["week_of_year"] = d.dt.isocalendar().week.astype(int)
        self.df["month"] = d.dt.month
        self.df["quarter"] = d.dt.quarter
        self.df["year"] = d.dt.year
        self.df["is_weekend"] = (self.df["day_of_week"] >= 5).astype(np.int8)

        if self.holidays is not None:
            active = self.holidays[self.holidays["transferred"] == False]
            holiday_dates = active["date"].dropna().unique()

            if len(holiday_dates) > 0:
                holiday_ns = np.sort(pd.to_datetime(holiday_dates).asi8)
                row_ns = self.df["date"].values.astype(np.int64)
                ns_per_day = np.int64(86_400 * 10**9)

                idx = np.searchsorted(holiday_ns, row_ns)
                left = np.abs(row_ns - holiday_ns[np.clip(idx - 1, 0, len(holiday_ns) - 1)])
                right = np.abs(row_ns - holiday_ns[np.clip(idx, 0, len(holiday_ns) - 1)])
                self.df["days_to_nearest_holiday"] = np.minimum(left, right) // ns_per_day
            else:
                self.df["days_to_nearest_holiday"] = 0

        return self

    def add_lag_and_rolling(
        self,
        lags: list[int] | None = None,
        windows: list[int] | None = None,
    ) -> "FastFeatureEngineer":

        self._assert_sorted()  # 🔥 critical

        if "sales" not in self.df.columns:
            return self

        lags = lags or [1, 7, 14, 28]
        windows = windows or [7, 14, 28]

        grouped_sales = self.df.groupby(_SORT_COLS[:2], sort=False)["sales"]
        group_keys = [self.df["store_nbr"], self.df["family"]]

        for lag in lags:
            self.df[f"sales_lag_{lag}d"] = grouped_sales.shift(lag)

        sales_shift_1d = grouped_sales.shift(1)
        for window in windows:
            roll = sales_shift_1d.groupby(group_keys, sort=False).rolling(window)
            self.df[f"rolling_mean_{window}d"] = roll.mean().reset_index(level=[0, 1], drop=True)
            self.df[f"rolling_std_{window}d"] = roll.std().reset_index(level=[0, 1], drop=True)

        return self

    def add_onpromotion_features(self) -> "FastFeatureEngineer":

        self._assert_sorted()  # 🔥 critical

        if "onpromotion" not in self.df.columns:
            return self

        grouped = self.df.groupby(_SORT_COLS[:2], sort=False)["onpromotion"]
        group_keys = [self.df["store_nbr"], self.df["family"]]

        onpromo_shift_1d = grouped.shift(1)
        self.df["onpromotion_lag_1d"] = onpromo_shift_1d
        self.df["onpromotion_rolling_7d"] = (
            onpromo_shift_1d
            .groupby(group_keys, sort=False)
            .rolling(7)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        return self

    def add_macroeconomic_features(self) -> "FastFeatureEngineer":

        if "dcoilwtico" not in self.df.columns:
            if self.oil_price is not None:
                self.df = self.df.merge(self.oil_price, on="date", how="left")

        if "dcoilwtico" in self.df.columns:
            oil_by_date = (
                self.df[["date", "dcoilwtico"]]
                .drop_duplicates(subset=["date"])
                .sort_values("date")
                .set_index("date")["dcoilwtico"]
            )

            oil_lag_7d = oil_by_date.shift(7)
            oil_roll_28d = oil_by_date.shift(1).rolling(28).mean()

            self.df["dcoilwtico_lag_7d"] = self.df["date"].map(oil_lag_7d)
            self.df["dcoilwtico_rolling_28d"] = self.df["date"].map(oil_roll_28d)

        return self

    def add_transaction_features(self) -> "FastFeatureEngineer":

        self._assert_sorted()  # 🔥 critical

        if "transactions" not in self.df.columns:
            if self.transactions is not None:
                self.df = self.df.merge(self.transactions, on=["store_nbr", "date"], how="left")

        if "transactions" in self.df.columns:
            self.df["transactions_lag_7d"] = (
                self.df.groupby("store_nbr")["transactions"].shift(7)
            )

        return self

    def add_store_metadata(self) -> "FastFeatureEngineer":

        if self.store_meta is not None and "store_type" not in self.df.columns:
            self.df = self.df.merge(self.store_meta, on="store_nbr", how="left")

        if "type" in self.df.columns:
            self.df = self.df.rename(columns={"type": "store_type"})

        # ✅ fixed condition (important bug fix)
        if (
            "store_type" in self.df.columns and
            (
                self.df["store_type"].dtype == object or
                self.df["store_type"].dtype.name in ("string", "str")
            )
        ):
            type_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            self.df["store_type"] = self.df["store_type"].map(type_map)

        return self

    def add_cannibalization_features(self) -> "FastFeatureEngineer":

        self._assert_sorted()  # 🔥 critical

        if "sales" not in self.df.columns or "onpromotion" not in self.df.columns:
            return self

        store_total = (
            self.df.groupby(["store_nbr", "date"])["sales"]
            .sum()
            .reset_index(name="store_total_sales")
        )

        self.df = self.df.merge(store_total, on=["store_nbr", "date"], how="left")

        self.df["other_family_sales"] = self.df["store_total_sales"] - self.df["sales"]

        self.df["other_family_sales_lag_7d"] = (
            self.df.groupby(_SORT_COLS[:2])["other_family_sales"].shift(7)
        )

        self.df = self.df.drop(columns=["store_total_sales", "other_family_sales"])

        return self

    def transform(self) -> pd.DataFrame:
        return self.df.copy()
