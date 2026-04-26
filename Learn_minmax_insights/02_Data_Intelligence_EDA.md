# 02 — Data Intelligence & EDA

## Mental Model

Favorita dataset = 54 stores × 33 product families × ~5 years of daily sales. Zero-sales rows are structural (store closed, item not stocked) — never remove them. Sales distribution is heavy-tailed: most rows are zero or low-volume; a few (GROCERY I, BEVERAGES) dominate total revenue.

## Dataset Breakdown

| File | Rows | Key Columns | Gap Risk |
|------|------|------------|----------|
| `train.csv` | 125M+ | store_nbr, family, date, sales, onpromotion | oil price gaps filled by ffill/bfill |
| `stores.csv` | 54 | store_nbr, type (A-E), cluster, city | Fixed metadata |
| `oil.csv` | ~1200 | date, dcoilwtico | Weekend gaps — market closed |
| `holidays.csv` | ~350 | date, locale, transferred | transferred=True → holiday moved |
| `transactions.csv` | ~83K | store_nbr, date, transactions | Sparse for some stores |

## EDA → Feature Mapping

| EDA Insight | Feature Engineered | Code Location |
|-------------|-------------------|--------------|
| Sales spike on Fri/Sat | `day_of_week`, `is_weekend` | `features.py:L94-L101` |
| Holiday proximity effect | `days_to_nearest_holiday` | `features.py:L103-L120` |
| Recent sales predict future | `sales_lag_1d`, `sales_lag_7d`, `rolling_mean_7d` | `features.py:L152-L163` |
| Oil price drives economy | `dcoilwtico_lag_7d`, `dcoilwtico_rolling_28d` | `features.py:L206-L217` |
| Promotion boosts immediate sales | `onpromotion_lag_1d`, `onpromotion_rolling_7d` | `features.py:L179-L189` |
| Transaction count = store traffic | `transactions_lag_7d` | `features.py:L234-L237` |
| Cannibalization between families | `other_family_sales_lag_7d` | `features.py:L273-L283` |

## Key Implementation Details

### Holiday Proximity (Binary Search — O(N log K))
```python
# features.py:L108-L118
holiday_ns = np.sort(pd.to_datetime(holiday_dates).asi8)  # nanoseconds
row_ns = self.df["date"].values.astype(np.int64)
idx = np.searchsorted(holiday_ns, row_ns)  # O(N log K)
# find nearest holiday in either direction
left  = np.abs(row_ns - holiday_ns[np.clip(idx - 1, 0, len(holiday_ns) - 1)])
right = np.abs(row_ns - holiday_ns[np.clip(idx, 0, len(holiday_ns) - 1)])
self.df["days_to_nearest_holiday"] = np.minimum(left, right) // ns_per_day
```
Previous approach: `apply(lambda ...)` was O(N × K) — 1000x slower on 3M rows.

### Oil Price — Date-Level Aggregation
```python
# features.py:L206-L217 — oil is market-level, not store-level
oil_by_date = self.df[["date", "dcoilwtico"]].drop_duplicates(subset=["date"])
oil_lag_7d = oil_by_date.shift(7)  # compute on unique dates
oil_roll_28d = oil_by_date.shift(1).rolling(28).mean()
self.df["dcoilwtico_lag_7d"] = self.df["date"].map(oil_lag_7d)
```
Common mistake: computing lag/rolling on the full df with duplicates — would leak values across store/family rows. Solution: deduplicate to date-level series first, then map back.

### Zero-Sales Retention
```python
# preprocessing.py — ZERO_RETENTION invariant
# detect_outliers_iqr() adds is_outlier flag but DOES NOT remove rows
# ZERO_RETENTION: Classify zeros only, never impute/remove
df["is_outlier"] = df["sales"] > (q3 + iqr_multiplier * iqr)  # only flags, no drop
```

> **Oil Weekend Gap — Spurious Correlation Risk:**
```python
# preprocessing.py:L86-L98 — clean_oil_prices()
df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
# Oil market closed Sat/Sun → no weekend data
# Sales exist for weekends (retail stores open)
# ffill propagates Friday price into Sat/Sun
```
**Problem:** Saturday sales get Friday oil price. If oil price is a proxy for general economic activity, this introduces a **lagged correlation** that doesn't reflect true Saturday market conditions.
**Mitigation:** Oil features (lag/rolling) use date-level deduplication first (`features.py:L208-L217`), so ffill impact is isolated to oil feature computation. However, the oil price for weekend sales is still forward-filled and may introduce subtle bias.

**Alternative:** Use only Monday oil price for weekend rows, or treat weekend as missing and let the model learn to ignore it.

## Distribution Characteristics

- **Heavy-tailed**: Most families average < 100 units/day. GROCERY I, BEVERAGES, PRODUCE dominate.
- **Zero-inflated**: ~40-60% of rows are zero sales (structural, not missing).
- **Skewed errors**: RMSLE penalizes relative error — better for skewed distributions than MSE (which over-penalizes high-volume items).
- **Seasonal**: Year-end holidays (Dec), Easter (variable), back-to-school (Aug-Sep).

## Feature Importance (From SHAP)

Expected top features from `evaluation.py:L94-L147`:
1. `sales_lag_7d` — weekly seasonality
2. `rolling_mean_7d` — trend capture
3. `onpromotion` / `onpromotion_lag_1d` — promo signal
4. `day_of_week` — weekly pattern
5. `transactions_lag_7d` — store traffic
