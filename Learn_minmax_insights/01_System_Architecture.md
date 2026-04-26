# 01 — System Architecture

## Mental Model

Retail-IQ = DAG pipeline. Raw CSVs → merge → feature engineering → model → evaluation. Each stage is a pure function: same input → same output, no hidden state.

```
┌─────────────────────────────────────────────────────────────┐
│                     ENTRY POINT                            │
│  notebooks/*.ipynb  (clean drivers, no inline logic)        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  src/retail_iq/config.py  — PATH_STRICT                    │
│  All I/O via constants. Never hardcode.                     │
│  PROJECT_ROOT, DATA_DIR, PARQUET_DATA_DIR, OUTPUT_DIR...    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  src/retail_iq/preprocessing.py                            │
│  load_raw_data() → prefers Parquet (10-20x faster)         │
│  clean_oil_prices() → ffill/bfill                         │
│  merge_datasets() → joins stores, oil, holidays, tx        │
│  strict_temporal_holdout_split() → train < Aug16, test ≥   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  src/retail_iq/features.py — FastFeatureEngineer           │
│  Fluent API, chainable, sort-invariant                      │
│  add_temporal_features()                                    │
│  add_lag_and_rolling(lags, windows)                         │
│  add_onpromotion_features()                                 │
│  add_macroeconomic_features()                              │
│  add_transaction_features()                                │
│  add_cannibalization_features()                             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  src/retail_iq/models.py                                    │
│  GD_Linear — gradient descent from scratch (JAX/NumPy)     │
│  SeasonalNaive — persistence baseline (shift 365)          │
│  Advanced: XGBoost + LightGBM via notebooks                  │
│  (NO LSTM, NO Prophet — these do not exist in this repo)   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  src/retail_iq/evaluation.py                                │
│  evaluate_model() — RMSLE, RMSE, MAPE, R²                   │
│  generate_shap_summary() — lazy import shap                │
│  plot_residuals()                                           │
└─────────────────────────────────────────────────────────────┘
```

## Execution Flow (Line-Reference)

| Step | File | Function | Key Detail |
|------|------|----------|------------|
| 1 | `config.py:L1-L28` | Path constants | All dirs created if missing |
| 2 | `preprocessing.py:L17-L61` | `load_raw_data()` | Parquet → CSV fallback |
| 3 | `preprocessing.py:L101-L149` | `merge_datasets()` | Left-joins, ffill transactions |
| 4 | `features.py:L28-L73` | `FastFeatureEngineer.__init__` | Sort once: `[store_nbr, family, date]` |
| 5 | `features.py:L79-L122` | `add_temporal_features()` | `np.searchsorted` holiday proximity |
| 6 | `features.py:L124-L165` | `add_lag_and_rolling()` | `.shift(1).rolling(w)` anti-leak |
| 7 | `models.py:L43-L181` | `GD_Linear.fit/predict` | L1+L2 in gradient step |
| 8 | `evaluation.py:L19-L57` | `evaluate_model()` | RMSLE primary metric |

## Critical Invariants

| ID | Rule | File:Line |
|----|------|-----------|
| V1 | Time-order preserved everywhere | `preprocessing.py:L211-L219` |
| V2 | Rolling features use `.shift(1).rolling(w)` | `features.py:L155-L163` |
| V5 | No zero-sales row removal | `preprocessing.py:L152-L183` |
| V6 | Seed locked at 42 | `config.py:L30-L35` |

## Data Shape

```
Favorita Store Sales (Ecuador)
├── train.csv:     125M+ rows  (store_nbr, family, date, sales, onpromotion)
├── test.csv:      ~28K rows    (submission target)
├── stores.csv:    54 rows      (store_nbr, type, cluster)
├── oil.csv:       ~1200 rows   (daily oil price)
├── holidays.csv:  ~350 rows    (locale, transferred flag)
└── transactions:  ~83K rows    (daily tx per store)

Temporal holdout: Train < 2017-08-16 | Test = Aug 16-31 2017 (15 days inclusive)

> **Holdout Period Justification:** 15 days chosen to cover one full fortnight (pay cycle in Ecuador). Alternatives: 7-day (weekly), 30-day (monthly). 15 days balances: enough data to detect weekly patterns × enough test rows to be statistically significant. Weakness: single fortnight may not capture end-of-month effects.
```

## Key Design Decisions

1. **Parquet over CSV**: `scripts/convert_to_parquet.py` — columnar, compressed, multi-threaded read via Polars
2. **Sort once invariant**: `FastFeatureEngineer.__init__` sorts `df` by `[store_nbr, family, date]`. No `add_*` method may re-sort. All shift/rolling ops rely on this.
3. **JAX backend for GD_Linear**: `@jax.jit` compiles gradient step to XLA. Falls back to pure NumPy if JAX unavailable.
4. **Lazy shap import**: `evaluation.py:L110` — avoids 200ms import overhead in notebooks that only need metrics.

## Systems Thinking

- **Bottleneck**: CSV I/O (10-20x slower than Parquet). Run `convert_to_parquet.py` once.
- **Leakage risk surface**: `add_lag_and_rolling` uses `.shift(1)` to prevent same-day leakage. If you remove the `.shift(1)`, you leak the current period.
- **Memory guard**: `SAMPLE_N_CORR=50_000`, `SAMPLE_N_DIST=100_000` in `config.py:L21-L22` cap plot data volume.
- **Zero-inflation blindspot**: 40-60% of rows are zero-sales. All models (GD_Linear, XGBoost, LightGBM) output continuous values — they cannot model `P(sales=0)` separately from `E(sales|sales>0)`. This systematically under-predicts zero periods. Future work: two-stage model (classifier + regressor).
