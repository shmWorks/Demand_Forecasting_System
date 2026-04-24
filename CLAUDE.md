# CLAUDE.md

Project Overview

Retail-IQ = ML sales forecasting for retail time-series. Favorita store sales data. XGBoost + Optuna tuning, SHAP evaluation, cannibalization analysis.

## Setup

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
pytest tests/
```

## Architecture

```
src/retail_iq/           # All logic here — NOT in notebooks
├── config.py             # Path constants (always use for I/O)
├── preprocessing.py      # load, merge, clean, outlier detection
├── features.py           # FastFeatureEngineer (fluent API)
├── models.py             # GD_Linear (JAX/NumPy), SeasonalNaive
├── evaluation.py         # metrics, SHAP, residual analysis
└── visualization.py      # plotting utilities

notebooks/                # Clean drivers only — no inline logic
```

## Key Constraints

1. **PATH_STRICT**: All I/O via `config.py` constants. Never hardcode.
2. **ANTI_LEAKAGE**: Rolling features use `.shift(1).rolling(w)`. Scaler fit on train only.
3. **TEMPORAL_HOLDOUT**: Train < 2017-08-16, Test = 2017-08-16 to 2017-08-31. No shuffle.
4. **FROM_SCRATCH**: GD_Linear = NumPy only. No sklearn.LinearRegression.
5. **ZERO_RETENTION**: Never impute/remove zero sales rows.
6. **SEED_LOCK**: random_state=42 everywhere.
7. **CV_TEMPORAL**: TimeSeriesSplit(n_splits≥3). KFold(shuffle=True) forbidden.
8. **API_REUSE**: FastFeatureEngineer fluent API. Chain methods. No inline feature code in notebooks.

## Core APIs

### FastFeatureEngineer (features.py)

```python
fe = FastFeatureEngineer(df, transactions=tx_df, oil_price=oil_df,
                         holidays=hol_df, store_meta=stores_df)
fe.add_temporal_features()
fe.add_lag_and_rolling(lags=[1,7,14,365], windows=[7,14,28])
fe.add_transaction_features()
fe.add_macroeconomic_features()
fe.add_store_metadata()
fe.add_cannibalization_features()
df = fe.transform()
```

Sort invariant: `self.df` sorted by `[store_nbr, family, date]` once in `__init__`. No `add_*` method may re-sort.

### Models (models.py)

- `GD_Linear`: Gradient descent linear regression. JAX backend (auto-falls back to NumPy). L1+L2 in gradient. `fit(X, y)` with log1p target. `predict(X)`.
- `SeasonalNaive`: Persistence baseline. `predict(df)` returns `groupby(['store_nbr','family'])['sales'].shift(365)`.

### Config (config.py)

```python
from retail_iq.config import PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from retail_iq.config import PARQUET_DATA_DIR, OUTPUT_DIR, PLOT_DIR, MODEL_DIR
```

`PARQUET_DATA_DIR` = fast Polars I/O path. Run `scripts/convert_to_parquet.py` once to convert CSVs.

Performance guards: `SAMPLE_N_CORR=50_000`, `SAMPLE_N_DIST=100_000` — max rows for plots.

## Cannibalization Analysis (SPEC §B, §STAGE_8)

```python
# Find cannibal pairs
find_cannibal_pairs(df, promo_threshold=0, corr_threshold=-0.35)

# Compute promotional lift
compute_promo_lift(df, window_pre=28)
```

## Cursor Rules

`.cursor/rules/always-caveman-style.mdc` forces caveman response style (terse, no fluff). Always active. Code/commits unaffected.

## Development Rules

- Logic in `src/retail_iq/`, not notebooks.
- Sync `requirements.txt` and `pyproject.toml` before installing deps.
- notebooks/ = clean drivers only. Atomic rerun by cell.