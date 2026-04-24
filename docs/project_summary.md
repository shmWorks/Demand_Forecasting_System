# 🚀 Retail-IQ: Hyper-Agent Manifest (SPOT v2.0)

> [!IMPORTANT]
> **AGENT_PROTOCOL**: Read first. Use API manifest plan tasks. Read no source files. Update STATE after each major turn.

## [META]

- **Goal**: Time-series forecasting for Favorita sales. XGBoost.
- **Arch**: Modular Python Package (`retail_iq`).
- **Stack**: Python 3.10, XGBoost, Scikit-learn, Flask.

## [STATE]

- **PHASE**: 3 (Insight/EDA) -> 4 (Learning/Training)
- **STATUS**: Dataset refactored. Modularized. EDA plots generated.
- **ACTIVE_TASK**: Transitioning to model training.
- **BLOCKERS**: None.

## [API_MANIFEST]

| File               | Members (Signatures)                                                                                                                                                                                                                                           | Description                 |
| :----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------- |
| `config.py`        | `PROJECT_ROOT`, `DATA_DIR`, `LOG_DIR`, `PLOT_DIR`                                                                                                                                                                                                              | Path constants (Pathlib).   |
| `preprocessing.py` | `load_raw_data()`, `preprocess_dates(dfs)`, `clean_oil_prices(oil_df)`, `merge_datasets(tr, st, oil, hol, tx)`, `detect_outliers_iqr(df)`                                                                                                                      | Raw ingestion to merged df. |
| `features.py`      | `FastFeatureEngineer(df, tx, oil, hol, st)`<br>`.add_temporal_features()`, `.add_lag_and_rolling()`, `.add_onpromotion_features()`, `.add_macroeconomic_features()`, `.add_transaction_features()`, `.add_store_metadata()`, `.add_cannibalization_features()` | Feature pipeline class.     |
| `visualization.py` | `plot_ts_decomposition(df, s, f)`, `plot_correlation_heatmap(df)`, `plot_sales_distribution(df)`                                                                                                                                                               | Automated plotting.         |

## [INDEX]

- `data/raw/`: Original Kaggle CSVs. GitIgnored.
- `notebooks/eda.ipynb`: Interactive analysis driver.
- `outputs/`: Figures, logs, models. GitIgnored.
- `tests/`: Unit/integration testing suite.
- `REFLECTIONS.md`: Design logic and lessons.
- `pyproject.toml`: Package/build configuration.

## [RULES]

1. **NO_FLUFF**: Dense reasoning. No filler.
2. **API_FIRST**: Use API_MANIFEST signatures before opening files.
3. **PATH_STRICT**: Always use `config.py` constants for file I/O.
4. **STATE_UPDATE**: Append task completion to STATE if phase changes.