# SPEC.md

## G Goal
- Build stable, fast retail demand forecasting pipeline.
- Cut crash risk, leakage risk, compute waste.
- Run same on local Windows + Kaggle + Colab.
- Keep outputs reproducible, auditable, business-usable.

## C Constraints
- No temporal leakage. Time-order strict.
- Keep zero sales rows (valid signal).
- Favor low RAM footprint, bounded CPU/GPU use.
- Prefer reusable `src/retail_iq` logic over notebook duplication.
- Notebook flow must support atomic rerun by cell.

## I Inputs
- Data: Favorita dataset files in project data paths.
- Docs truth: roadmap PDF, proposal PDF, project slides.
- Current code: `src/retail_iq`, `notebooks`, `scripts`, tests.

## V Invariants
- V1: Any split/train/eval step preserves time direction.
- V2: Feature generation no cross-time leakage.
- V3: No synthetic row explosion (no blind full-data replication).
- V4: Dependencies for fast paths align across `pyproject.toml` and `requirements.txt`.
- V5: Notebook cells single-purpose and restart-safe.
- V6: Seed and config path deterministic for reproducibility.

## T Tasks
- T1 Audit: Build severity-ranked findings register from docs-vs-code.
- T2 Stabilize: Fix crashers and correctness bugs first.
- T3 Optimize: Reduce memory copies, dtype bloat, wasteful loops.
- T4 Refactor: Split target notebooks into atomic script-backed cells.
- T5 Portability: Harden Kaggle/Colab bootstrap + imports + deps.
- T6 Verify: Run tests + smoke runs + lint on touched files.

## B Backprop Bug Log
- B1 Memory blowup risk: advanced notebook data replication path.
- B2 Eval split risk: cutoff date may produce invalid/empty eval slice.
- B3 Leakage risk: lag feature computation path needs strict temporal guard.
- B4 Portability risk: dependency drift blocks fast paths on hosted runtimes.
# SPEC.md — Retail-IQ

## [CURRENT_STATE]

status: ALL_PHASES_COMPLETE
done: |
config.py, preprocessing.py, features.py, visualization.py,
models.py (GD_Linear, SeasonalNaive), evaluation.py,
notebooks/eda.ipynb, baseline_models.ipynb, advanced_models.ipynb,
evaluation.ipynb, cannibalization.ipynb
next: EXECUTE_NOTEBOOKS → GENERATE_REPORTS → SUBMIT

---

## [CHECKPOINT_LOG]

| Date       | Phase | Status                                        |
| ---------- | ----- | --------------------------------------------- |
| 2026-04-20 | 1-4   | DONE — Core modules, EDA complete             |
| 2026-04-20 | 5     | DONE — Baseline (SeasonalNaive, GD_Linear)    |
| 2026-04-20 | 6     | DONE — Advanced (XGBoost, LightGBM + Optuna)  |
| 2026-04-20 | 7     | DONE — Evaluation + SHAP framework            |
| 2026-04-20 | 8     | DONE — Cannibalization analysis               |
| NEXT       | 9     | Finalize best model, generate reports, submit |

---

## [API_INDEX]

```python
# config.py — Path constants
PROJECT_ROOT: Path
DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR: Path
OUTPUT_DIR, PLOT_DIR, MODEL_DIR, LOG_DIR: Path

# preprocessing.py
load_raw_data() -> Tuple[train, test, stores, oil, holidays, transactions]
preprocess_dates(dfs: List[DataFrame]) -> List[DataFrame]
clean_oil_prices(oil_df: DataFrame) -> DataFrame
merge_datasets(train, stores, oil, holidays, transactions) -> DataFrame
detect_outliers_iqr(df: DataFrame) -> DataFrame with 'is_outlier' col

# features.py: FastFeatureEngineer
__init__(df, transactions=None, oil_price=None, holidays=None, store_meta=None)
.add_temporal_features() -> self  # day_of_week, week_of_year, month, quarter, year, is_weekend
.add_lag_and_rolling(lags=[1,7,14,365], windows=[7,14,28]) -> self  # sales_lag_Nd, sales_roll_mean/std_Nd
.add_onpromotion_features() -> self  # onpromotion_lag_1d, onpromotion_rolling_7d
.add_macroeconomic_features() -> self  # requires oil_price df; adds dcoilwtico_lag_7d, rolling_28d
.add_transaction_features() -> self  # requires transactions df; adds transactions_lag_7d
.add_store_metadata() -> self  # requires store_meta; encodes store_type to categorical codes
.add_cannibalization_features(top_n=3) -> self  # cross-family corr, adds top_corr_mean
.transform() -> DataFrame
```

---

## [NEXT_STAGES]

### STAGE_5: Baseline Models (WEEK_5)

deliver: baseline_models.ipynb + /models/naive.pkl, /models/gd_linear.pkl

| Model         | Impl                                                   | Key Requirements                                   |
| ------------- | ------------------------------------------------------ | -------------------------------------------------- |
| SeasonalNaive | df.groupby(['store_nbr','family'])['sales'].shift(365) | Zero training, persistence forecast                |
| GD_Linear     | NumPy only, no sklearn.LinearRegression                | L1+L2 in gradient, log1p(y), 1000 iter, loss curve |

metrics: RMSLE, RMSE, MAPE, R² on test (Aug 16-31 2017)

### STAGE_6: Advanced + Tuning (WEEK_6)

deliver: advanced_models.ipynb + /models/xgb_tuned.pkl, /models/lgb_tuned.pkl + JSON params

| Model         | CV                          | Search Space                                                                                          |
| ------------- | --------------------------- | ----------------------------------------------------------------------------------------------------- |
| XGBRegressor  | TimeSeriesSplit(n_splits=5) | n_estimators(200-1000), max_depth(3-10), lr(0.01-0.3), subsample(0.5-1.0), colsample(0.5-1.0)         |
| LGBMRegressor | TimeSeriesSplit(n_splits=5) | num_leaves(20-150), max_depth(3-12), lr(0.01-0.3), min_child_samples(5-50), feature_fraction(0.5-1.0) |

tuning: Optuna TPE (Bayesian) preferred
invariant: random_state=42, scaler fit on train ONLY

### STAGE_7: Evaluation + SHAP (WEEK_7)

deliver: evaluation.ipynb + SHAP plots + residual analysis

reqs:

- metrics table (4 models × 4 metrics)
- actual vs predicted plots (top 5 store-family by volume)
- residual plots (check bias: |mean residual| > 5% of mean actual)
- SHAP TreeExplainer on best model → summary_plot bar
- SHAP validation: ≥1 promo feature AND ≥1 temporal/lag in top-5

### STAGE_8: Cannibalization Analysis (WEEK_7-8)

deliver: cannibalization_report.md + figures

| Analysis          | Method                                                                          | Threshold             |
| ----------------- | ------------------------------------------------------------------------------- | --------------------- |
| Cross-family corr | Pearson on residual sales (actual - 28d rolling mean), filtered onpromotion>0   | r < -0.35 = candidate |
| Promotional lift  | (actual - counterfactual) / counterfactual; counterfactual = 4wk pre-promo mean | ≥10 distinct events   |

reqs:

- ≥3 cannibal pairs with r, p-value
- lift scatter: promotion magnitude vs lift
- markdown report 3-5 pages

### STAGE_9: Finalization (WEEK_8)

deliver: best_model.pkl + models/README.md + final_report.ipynb

checklist:

- joblib save with reload verification (bit-for-bit)
- models/README.md: name, date, data range, metrics, features, seeds
- notebook Restart & Run All → zero errors
- export HTML/PDF ≤20MB

---

## [CONSTRAINTS]

1. PATH_STRICT: All I/O via config.py constants. Never hardcode paths.
2. TEMPORAL_HOLDOUT: Train < 2017-08-16, Test = 2017-08-16 to 2017-08-31. No shuffle.
3. ANTI_LEAKAGE: All rolling features use .shift(1).rolling(w). Scaler fit on train only.
4. FROM_SCRATCH: GD_Linear = NumPy only, no sklearn.LinearRegression.
5. ZERO_RETENTION: Do NOT impute/remove zero sales. Classify only (structural/event/intermittent).
6. TRANSFERRED_EXCLUSION: holidays[holidays['transferred'] == False] for active flags.
7. SEED_LOCK: random_state=42, PYTHONHASHSEED=42 in all model constructors.
8. API_REUSE: Use FastFeatureEngineer fluent API. Chain methods. No inline feature code in notebooks.
9. CV_TEMPORAL: TimeSeriesSplit(n_splits≥3). KFold(shuffle=True) explicitly forbidden.
10. METRIC_PRIMARY: RMSLE (handles zeros, asymmetric penalty). MAPE secondary (exclude zero actuals).

---

## [CRITICAL_IMPLEMENTATIONS]

### GD_Linear (from scratch)

```python
import numpy as np

def gradient_descent(X, y, lr=0.001, iterations=1000, l1=0.0, l2=0.0):
    """X: (n_samples, n_features), y: (n_samples,) — log1p(sales) target"""
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    for i in range(iterations):
        preds = X @ theta
        errors = preds - y
        # L2: + (l2/m)*theta, L1: + (l1/m)*sign(theta)
        grad = (2/m) * (X.T @ errors) + (l2/m)*theta + (l1/m)*np.sign(theta)
        theta -= lr * grad
        losses.append(np.mean(errors**2))
    return theta, losses
```

### Cannibalization Detection

```python
def find_cannibal_pairs(df, promo_threshold=0, corr_threshold=-0.35):
    """df cols: store_nbr, family, date, sales, onpromotion"""
    stores = df['store_nbr'].unique()
    pairs = []
    for s in stores:
        df_s = df[df['store_nbr']==s]
        # Residual sales (detrended)
        df_s['sales_resid'] = df_s['sales'] - df_s.groupby('family')['sales'].transform(
            lambda x: x.shift(1).rolling(28, min_periods=1).mean()
        )
        # Filter promotion periods
        promo_mask = df_s['onpromotion'] > promo_threshold
        df_promo = df_s[promo_mask].pivot(index='date', columns='family', values='sales_resid')
        corr = df_promo.corr()
        # Find negative correlations
        for i, fam_i in enumerate(corr.columns):
            for fam_j in corr.columns[i+1:]:
                r = corr.loc[fam_i, fam_j]
                if r < corr_threshold:
                    pairs.append({'store':s, 'family_i':fam_i, 'family_j':fam_j, 'r':r})
    return pd.DataFrame(pairs)
```

### Promotional Lift

```python
def compute_promo_lift(df, window_pre=28):
    """Counterfactual = rolling mean of 28 days pre-promotion"""
    df = df.sort_values(['store_nbr','family','date'])
    df['baseline'] = df.groupby(['store_nbr','family'])['sales'].transform(
        lambda x: x.shift(1).rolling(window_pre, min_periods=1).mean()
    )
    df['lift'] = (df['sales'] - df['baseline']) / df['baseline']
    promo_events = df[df['onpromotion'] > 0][['store_nbr','family','date','sales','baseline','lift']]
    return promo_events
```

---

## [FILE_MANIFEST]

src/retail_iq/
├── **init**.py # Package marker
├── config.py # PATH constants — DONE
├── preprocessing.py # load, merge, clean, outlier — DONE
├── features.py # FastFeatureEngineer — DONE
├── visualization.py # Stub — needs impl
├── models.py # MISSING — add GD_Linear, model training wrappers
└── evaluation.py # MISSING — metrics, SHAP, residual plots

notebooks/
├── eda.ipynb # EXISTS
├── baseline_models.ipynb # CREATE — Stage 5
├── advanced_models.ipynb # CREATE — Stage 6
├── evaluation.ipynb # CREATE — Stage 7
└── cannibalization.ipynb # CREATE — Stage 8

models/ # .gitignore large files
├── naive*baseline.pkl
├── gd_linear_YYYYMMDD_v1.pkl
├── xgb_tuned_YYYYMMDD_v1.pkl
├── lgb_tuned_YYYYMMDD_v1.pkl
├── best_params*\*.json
└── README.md

outputs/
├── eda/ # 10+ figures from EDA
├── plots/ # model comparison, residuals, SHAP
└── reports/ # cannibalization_report.md

---

## [CHECKPOINT_LOG]

| Date       | Phase | Status                                         |
| ---------- | ----- | ---------------------------------------------- |
| 2026-04-20 | 1-4   | DONE — Core modules, EDA complete              |
| NEXT       | 5     | Baseline modeling — Seasonal Naive + GD_Linear |
| NEXT       | 6     | Advanced models + Optuna tuning                |
| NEXT       | 7     | Evaluation + SHAP analysis                     |
| NEXT       | 8     | Cannibalization analysis + final report        |

---

## [REPRODUCIBILITY]

env: requirements.txt pinned (xgboost, lightgbm, optuna, shap, joblib)
seeds: random*state=42, PYTHONHASHSEED=42
data: SHA-256 hashes in data/checksums.txt
exec: notebooks numbered (01*, 02\_) for order
