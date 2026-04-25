"""Feature engineering pipeline for Retail-IQ.

Performance contract
--------------------
- self.df is sorted by [store_nbr, family, date] exactly ONCE in __init__.
  No add_* method may re-sort.  All shift/rolling ops depend on this invariant.
- days_to_nearest_holiday uses np.searchsorted — O(N log K) not O(N×K).
- No mutable default arguments anywhere (None-sentinel pattern throughout).
- No df.copy() inside chained methods — copies only at API boundary (.transform()).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass  # avoid circular imports

logger = logging.getLogger(__name__)

# Sort key reused by __init__ and any future method that needs to document the invariant
_SORT_COLS = ["store_nbr", "family", "date"]


class FastFeatureEngineer:
    """Fluent feature engineering pipeline for the Favorita retail dataset.

    Usage::

        fe = FastFeatureEngineer(df, transactions=tx, oil_price=oil,
                                  holidays=holidays, store_meta=stores)
        features = (
            fe.add_temporal_features()
              .add_lag_and_rolling()
              .add_onpromotion_features()
              .add_macroeconomic_features()
              .add_transaction_features()
              .add_store_metadata()
              .add_cannibalization_features()
              .transform()
        )

    Sort invariant:
        self.df is sorted by [store_nbr, family, date] in __init__.
        All add_* methods RELY on this invariant — they MUST NOT re-sort.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transactions: pd.DataFrame | None = None,
        oil_price: pd.DataFrame | None = None,
        holidays: pd.DataFrame | None = None,
        store_meta: pd.DataFrame | None = None,
    ) -> None:
        """Initialise the engineer and establish the sort invariant.

        Args:
            df:           Merged training DataFrame (output of merge_datasets).
            transactions: Optional daily transactions per store.
            oil_price:    Optional daily oil price DataFrame.
            holidays:     Optional holiday events DataFrame.
            store_meta:   Optional store metadata DataFrame.
        """
        # CONTRACT: sorted once here.  All add_* methods assume this order.
        self.df = df.sort_values(_SORT_COLS).reset_index(drop=True)
        self.transactions = transactions
        self.oil_price = oil_price
        self.holidays = holidays
        self.store_meta = store_meta

    # ------------------------------------------------------------------
    # Public API — chainable add_* methods
    # ------------------------------------------------------------------

    def add_temporal_features(self) -> "FastFeatureEngineer":
        """Add calendar and holiday-proximity features.

        Features added:
            day_of_week, day_of_month, week_of_year, month, quarter, year,
            is_weekend, days_to_nearest_holiday (when holidays provided).

        Performance:
            days_to_nearest_holiday uses np.searchsorted — O(N log K) where
            N = number of rows, K = number of holiday dates.  The previous
            apply(lambda ...) was O(N × K) — up to 1000x slower on 3M rows.

        Returns:
            self (for method chaining).
        """
        d = self.df["date"]
        self.df["day_of_week"]   = d.dt.dayofweek
        self.df["day_of_month"]  = d.dt.day
        self.df["week_of_year"]  = d.dt.isocalendar().week.astype(int)
        self.df["month"]         = d.dt.month
        self.df["quarter"]       = d.dt.quarter
        self.df["year"]          = d.dt.year
        self.df["is_weekend"]    = (self.df["day_of_week"] >= 5).astype(np.int8)

        if self.holidays is not None:
            active = self.holidays[self.holidays["transferred"] == False]  # noqa: E712
            holiday_dates = active["date"].dropna().unique()

            if len(holiday_dates) > 0:
                # Vectorized binary-search approach — O(N log K)
                holiday_ns = np.sort(
                    pd.to_datetime(holiday_dates).asi8  # int64 nanoseconds — .asi8 is the correct attr
                )
                row_ns = self.df["date"].values.astype(np.int64)
                ns_per_day = np.int64(86_400 * 10 ** 9)

                idx = np.searchsorted(holiday_ns, row_ns)
                left  = np.abs(row_ns - holiday_ns[np.clip(idx - 1, 0, len(holiday_ns) - 1)])
                right = np.abs(row_ns - holiday_ns[np.clip(idx,     0, len(holiday_ns) - 1)])
                self.df["days_to_nearest_holiday"] = np.minimum(left, right) // ns_per_day
            else:
                self.df["days_to_nearest_holiday"] = 0

        return self

    def add_lag_and_rolling(
        self,
        lags: list[int] | None = None,
        windows: list[int] | None = None,
    ) -> "FastFeatureEngineer":
        """Add grouped lag and rolling-window features for 'sales'.

        Args:
            lags:    Lag periods in days.  Default [1, 7, 14, 28].
                     Note: lag_365d removed from defaults — creates NaN for
                     all of year-1 and is dropped by downstream notebooks.
            windows: Rolling-window sizes.  Default [7, 14, 28].

        Returns:
            self (for method chaining).

        Note:
            Relies on sort invariant set in __init__.  Do NOT call sort_values here.
        """
        if "sales" not in self.df.columns:
            return self

        lags    = lags    or [1, 7, 14, 28]
        windows = windows or [7, 14, 28]

        grouped_sales = self.df.groupby(_SORT_COLS[:2], sort=False)["sales"]  # store_nbr, family
        group_keys = [self.df["store_nbr"], self.df["family"]]

        for lag in lags:
            self.df[f"sales_lag_{lag}d"] = grouped_sales.shift(lag)

        sales_shift_1d = grouped_sales.shift(1)
        for window in windows:
            roll = sales_shift_1d.groupby(group_keys, sort=False).rolling(window)
            self.df[f"rolling_mean_{window}d"] = (
                roll.mean().reset_index(level=[0, 1], drop=True)
            )
            self.df[f"rolling_std_{window}d"] = (
                roll.std().reset_index(level=[0, 1], drop=True)
            )

        return self

    def add_onpromotion_features(self) -> "FastFeatureEngineer":
        """Add promotion lag and rolling features.

        Features added:
            onpromotion_lag_1d, onpromotion_rolling_7d.

        Returns:
            self (for method chaining).
        """
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
        """Add oil-price lag and rolling features.

        Features added:
            dcoilwtico_lag_7d, dcoilwtico_rolling_28d.

        Returns:
            self (for method chaining).
        """
        if "dcoilwtico" not in self.df.columns:
            if self.oil_price is not None:
                self.df = self.df.merge(self.oil_price, on="date", how="left")

        if "dcoilwtico" in self.df.columns:
            # Oil is market-level signal by date. Compute lag/rolling on unique date series,
            # then map back by date to avoid row-based leakage across store/family duplicates.
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
        """Add transaction-count lag feature per store.

        Features added:
            transactions_lag_7d.

        Returns:
            self (for method chaining).
        """
        if "transactions" not in self.df.columns:
            if self.transactions is not None:
                self.df = self.df.merge(self.transactions, on=["store_nbr", "date"], how="left")

        if "transactions" in self.df.columns:
            self.df["transactions_lag_7d"] = (
                self.df.groupby("store_nbr")["transactions"].shift(7)
            )
        return self

    def add_store_metadata(self) -> "FastFeatureEngineer":
        """Encode store type (A-E → 0-4) and merge remaining metadata if missing.

        Returns:
            self (for method chaining).
        """
        if self.store_meta is not None and "store_type" not in self.df.columns:
            self.df = self.df.merge(self.store_meta, on="store_nbr", how="left")

        if "type" in self.df.columns:
            self.df = self.df.rename(columns={"type": "store_type"})

        if "store_type" in self.df.columns and self.df["store_type"].dtype == object or self.df["store_type"].dtype.name in ("string", "str"):
            type_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            self.df["store_type"] = self.df["store_type"].map(type_map)

        return self

    def add_cannibalization_features(self) -> "FastFeatureEngineer":
        """Add cross-family cannibalization proxy via other-family sales lag.

        Computes store-level total sales per date, subtracts current family's
        sales to get 'other family sales', then lags by 7 days.

        Features added:
            other_family_sales_lag_7d.

        Returns:
            self (for method chaining).
        """
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
        """Return the fully engineered DataFrame.

        Returns:
            A copy of the internal DataFrame with all added features.
        """
        return self.df.copy()
