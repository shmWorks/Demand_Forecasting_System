# ADR Matrix — Smart Retail Demand Forecasting

## Corporación Favorita
### Ayesha Khalid 23L-0667 | Uma E Rubab 23L-0928 | Sheraz Malik 23L-0572

---

## Part I: ADR Matrix

> Each row = immutable design decision. "Supporting Lit" = annotation from Literature Review. "Implementation Cost" = Low/Medium/High. "Failure Mode" = corruption if reversed.

| # | Component | Decision | Supporting Lit | Impl Cost | Failure Mode |
|---|-----------|----------|----------------|-----------|--------------|
| ADR-01 | Primary metric | RMSLE | [8] asymmetric under-prediction penalty = retail stockout cost > overstock; [1] RMSLE dominant competition metric for right-skewed, zero-inflated retail | Low | MAPE division-by-zero on 23–30% zero-sale rows; RMSE fails to penalise under-prediction asymmetrically |
| ADR-02 | Primary models | XGBoost + LightGBM co-primary; LightGBM preferred at scale | [1] XGB/LGB most employed across 516-paper mapping; [2] LightGBM deployed across 100Ks SKUs including Favorita; [22] LightGBM outperforms TimeGPT on 16-day Favorita horizon | Medium | Foundation model (TimeGPT/Chronos) as primary sacrifices supervised performance gains on this specific dataset |
| ADR-03 | Zero-sale treatment | Retain all zeros; classify 3-category taxonomy (structural / event-driven / intermittent) | [9] out-of-stock zeros as true zeros biases lag features; [10] classify-then-forecast reduces RMSLE 8.3% on zero-heavy families | Medium | Imputing zeros corrupts rolling stats; removing destroys temporal continuity; treating all zeros as demand inflates stockout forecasts |
| ADR-04 | Earthquake anomaly | Flag 2016-04-16 to 2016-05-15 as `is_outlier=1`; retain rows, add flag as feature | [9] external-disruption zeros must be distinguished from demand zeros; [10] third zero category addresses supply-chain shock zeros | Low | Treating earthquake demand spike/crash as normal data biases rolling means/lag features for all Ecuadorian stores Apr–May 2016, propagating corruption to next 365 days |
| ADR-05 | Rolling feature construction | `.shift(1).rolling(w).mean()` — mandatory pre-shift before every rolling | [14] sequences without pre-shift inflate RMSE improvements up to 20.5% under K-fold CV; [24] leakage distorts LSTM vs tree comparisons | Low | Omitting `.shift(1)` makes today's sales predict itself; training loss collapses (near-zero) while test RMSLE catastrophically high |
| ADR-06 | CV strategy | `TimeSeriesSplit(n_splits=5)` — minimum 3 splits enforced as pipeline invariant | [14] KFold with shuffle=True allows future data into training, inflating apparent performance 15–20%; [24] deep learning appears to outperform trees under leaky CV but gap dissolves under clean temporal splits | Low | `KFold(shuffle=True)` on time-series produces optimistically biased hyperparameter selection; tuned models underperform on held-out test |
| ADR-07 | Holiday feature granularity | Three independent binary flags: `is_national_holiday`, `is_regional_holiday`, `is_local_holiday` + `days_to_nearest_holiday` continuous feature | [8] promotional + holiday periods interact; [7] hierarchical promotional data at every aggregation level materially reduces SKU-level error | Low | Collapsing to single `is_holiday` binary loses 31% vs 9% differential between national and regional holidays; encoding transferred holidays as active introduces systematic errors |
| ADR-08 | Transferred holiday handling | `holidays[holidays['transferred'] == False]` — hard exclusion before flag construction | [8] service-level anchoring biases from incorrect holiday classification; FR-02 explicit requirement | Low | Including transferred holidays as active marks workdays as holidays, adding spurious demand spikes/drops; pipeline enforces as INV-10 |
| ADR-09 | Lag feature windows | 1d, 7d, 14d, 365d per (store_nbr, family) group | [13] SHAP at Colombian retailer identifies 7-day lag as top predictor; [17] same finding on grocery data; ACF/PACF confirms weekly and annual autocorrelation | Low | Omitting 365-day lag loses annual seasonality; computing lags globally (no groupby) silently crosses group boundaries |
| ADR-10 | Promotion feature encoding | Numerical count (onpromotion items) + duration (rolling count consecutive promo days) — NOT binary flag | [17] promotions < 3 days yield negligible lift, duration must be numerical; [5] `days_since_last_promotion` and `days_to_next_promotion` rank among highest SHAP predictors | Low | Binary flag loses promotion lifecycle signal (pre-stockpile, in-promotion lift, post-promotion dip); reduces RMSE 18% when modelled vs spike-only |
| ADR-11 | Promotional lift counterfactual | 4-week rolling pre-promotion mean as counterfactual baseline | [5] post-promotion demand dip documented; simple pre-period mean standard counterfactual; FR-08 spec | Low | Global store average as baseline ignores seasonal baseline drift; same-week prior-year baseline conflates holiday + promotion effects |
| ADR-12 | Cannibalization identification | Pearson r < −0.35 during `onpromotion > 0` periods on residual sales | [2] cross-SKU interaction patterns in Favorita; FR-08 threshold; [7] promotional data improves hierarchical forecast at SKU level | Medium | Simple negative-correlation screen on raw sales conflates seasonal co-movement with promotional cannibalization; filtering to promotion-active periods isolates causal mechanism |
| ADR-13 | Oil price imputation | Forward-fill then backward-fill after sorting by date globally | FR-02; [4] external contextual variables yield largest accuracy gain in context-augmented XGBoost | Low | Filling within store groups propagates store-specific timing artifacts; forward-fill before backward-fill ensures most recent known price used preferentially |
| ADR-14 | Hyperparameter tuning engine | Optuna TPE sampler (Bayesian) preferred over GridSearchCV | [15] TPE identifies near-optimal XGB/LGB configs in 3–5× fewer trials than grid; [28] multi-objective sampling allows simultaneous RMSLE + inventory cost minimisation | Medium | Grid search over 5-parameter XGBoost space (4 values each) = 4^5 = 1,024 evaluations × 5 CV folds = 5,120 model fits; prohibitive on ~3M rows |
| ADR-15 | Scaler fit domain | StandardScaler.fit() exclusively on training rows; transform applied to train and test | [14] fitting on full dataset makes test-set statistics influence training normalization (leakage); [24] particularly distorts LSTM vs tree comparisons | Low | Fitting scaler on full data leaks test-set mean/variance into training normalization; methodological violation reviewers flag |
| ADR-16 | Baseline models | Seasonal Naive (365-day lag) + NumPy-only Gradient Descent Linear Regression | [3] all deep learning variants require strong naive baseline for relative improvement quantification; [1] RMSE improvement over naive benchmark framing | Low | Without baseline, no "improvement" claim substantiated; from-scratch GD implementation mandatory deliverable (proposal §5.5) cannot be replaced by sklearn.LinearRegression |
| ADR-17 | Feature selection (trees vs linear) | VIF pruning for linear model only (VIF > 10 dropped); SHAP-guided post-hoc pruning for tree models | [17] SHAP dependence plots enable attribution-guided pruning; [19] attribution-guided selection reduces input dimensionality 30% without accuracy loss | Medium | Applying VIF pruning to tree models removes correlated lag features carrying independent signal at different time horizons; trees collinearity-tolerant by construction |
| ADR-18 | Model persistence format | joblib (.pkl) with bit-for-bit reload verification + params JSON | FR-09; reproducibility plan §5.1 | Low | Pickle without verification silently fails if serialized in incompatible library version; saving params separately enables hyperparameter auditability |
| ADR-19 | Retraining cadence | Weekly retraining recommended; daily retraining explicitly rejected | [29] daily retraining introduces quantile flip-flopping disrupting safety-stock calculations; weekly preserves accuracy with substantially higher stability | Low | Daily retraining in production causes SMQC metric to spike, indicating unstable probabilistic forecasts making downstream inventory planning unreliable |
| ADR-20 | Foundation model role | Chronos-2 and TimeGPT reserved as Future Work zero-shot baselines; excluded from primary pipeline | [22] LightGBM with hand-crafted features outperforms TimeGPT in supervised setting on Favorita 16-day horizon; [23] multivariate foundation frontier; [20] training-free reference point | Low | Foundation model as primary sacrifices interpretability and SHAP-based cannibalization analysis constituting highest-differentiation deliverables |
| ADR-21 | SHAP implementation | SHAP TreeExplainer (not feature permutation importance) | [13] global + local attributions for business stakeholders; [17] dependence plots reveal non-linear promotion × weekday interaction; [30] SHAP validated as auditable interpretability layer | Low | Built-in tree feature importance inconsistent — same model ranks features differently depending on tree construction order; SHAP values only theoretically grounded alternative |
| ADR-22 | Hierarchical reconciliation | Future Work: robust reconciliation via semidefinite program | [26] semidefinite program ensures coherent forecasts across store → family → national hierarchy while minimising worst-case squared error | High (Future Work) | Without reconciliation, store-level predictions do not sum to family-level aggregates; management uses inconsistent numbers from different model outputs |
| ADR-23 | Intermittent demand handling | Binary `is_zero_intermittent` flag as feature; hurdle-model architecture reserved for Future Work | [11] Switch-Hurdle outperforms standard regression 14% RMSLE on sparse grocery families; [12] dual-CNN mirrors hurdle concept for low-frequency families | Medium (core) / High (Future Work) | Treating intermittent demand with standard regressor systematically underestimates occurrence probability for families like BABY CARE and MAGAZINES; binary flag lets tree partially learn two-regime structure |
| ADR-24 | Training/inference window alignment | Lag windows, rolling windows, holiday proximity features must be constructed identically in training and inference code paths | [25] misaligned look-back horizons inflate test errors 8–15% on strongly seasonal datasets; ensemble across look-back lengths provides robustness | Medium | Common bug: training uses 14-day rolling means but inference recomputes on shorter available window; feature distribution shift causes silent accuracy degradation only detectable by monitoring RMSLE in production |
| ADR-25 | Experiment tracking | MLflow (recommended) or manually maintained experiments_log.csv | §5.3; [29] SMQC metric requires tracking probabilistic outputs across retraining cycles | Low | Without experiment tracking, hyperparameter-to-metric lineage lost; reproducing best model requires re-running all experiments |

---

## Part II: IEEE Methodology — Guidance and Draft

### IEEE Paper Structure
Methodology section (Section III) communicates five things: (1) dataset + split rationale, (2) feature engineering tied to literature, (3) model selection + training protocol, (4) evaluation framework + formulas, (5) validity + reproducibility controls.

---

### Draft — III. METHODOLOGY

**A. Dataset and Experimental Configuration**

Corporación Favorita Grocery Sales dataset: ~3.0M daily sales, Jan 2013–Aug 2017, 54 stores, 33 product families. Six relational tables joined on `(store_nbr, date)`: transactions, store metadata, WTI crude oil prices, holiday events, foot traffic. Primary modelling unit = `(store_nbr, family, date)` triple → 1,782 distinct time series.

Temporal holdout: August 16–31, 2017 (16 days). All training uses data strictly preceding August 16. No temporal shuffling. Enforced as runtime invariant.

**B. Preprocessing and Data Integrity**

~44 missing oil price values (weekend/holiday gaps) → forward-fill then backward-fill globally by date. Transaction counts (~2% missing) → forward-fill within store groups.

Holiday events: `transferred=True` rows excluded from all active-holiday flags (per roadmap spec; treating transferred as active introduces systematic feature errors). Zero-sales retained without imputation, classified per Svetunkov and Sroginis [10]: structural zeros (zero mean across entire series history), event-driven zeros (Apr–May 2016 earthquake, flagged `is_outlier`), intermittent zeros (residual sparse demand). Outlier detection: per `(store_nbr, family)` IQR with 3× threshold; flagged rows retained with `is_outlier` covariate.

**C. Feature Engineering**

Nine feature groups grounded in prior literature:

- **Temporal**: day-of-week, week-of-year, month, quarter, year, weekend indicator — per Fatima and Salam [4] contextual augmentation framework (largest accuracy gains from weekday/holiday augmentation).
- **Lag features**: 1-, 7-, 14-, 365-day within `(store_nbr, family)` group. 7-day lag = single highest-importance predictor per SHAP analyses [13][17]; inclusion non-negotiable.
- **Rolling statistics**: 7-, 14-, 28-day rolling mean and std via mandatory `.shift(1).rolling(w)` pattern. Without pre-shift, current target contaminates own rolling mean → 20.5% RMSE inflation under CV [14].
- **Promotional**: raw `onpromotion` count (numerical, not binary), 1-day lag, 7-day rolling mean, `promo_duration_rolling` (consecutive promo days in 14-day window). Duration-aware encoding required per [17]: promotions < 3 days negligible lift; `days_since_last_promotion` reduces RMSE 18% vs spike-only [5].
- **Macroeconomic**: WTI crude oil price, 7-day lag, 28-day rolling mean — Ecuadorian retail sensitive to oil price fluctuations [4].
- **Transaction volume**: daily store foot traffic with 7-day lag as leading indicator [4].
- **Store metadata**: store type (label-encoded A–E), cluster (1–17), target-encoded city/state. Target encoding fit exclusively on training partition [14].
- **Cannibalization proxy**: 7-day lagged mean sales of top-3 most negatively correlated families during `onpromotion > 0` periods. Correlation map derived from training partition only.

First calendar year per series (invalid 365-day lag) dropped before training. Expected: ~4 years of data per series remain.

**D. Model Development**

Four models in increasing complexity order:

1. **Seasonal Naïve**: each day = value 365 days prior within `(store_nbr, family)`. Performance floor [3].
2. **Gradient Descent Linear Regression**: from scratch NumPy, L1+L2 regularisation analytically in gradient update. Target = log1p(sales); predictions = expm1. 1,000 iterations with loss trajectory validation.
3. **XGBoost Regressor**: `hist` tree method, n_estimators=500, early stopping 50 rounds.
4. **LightGBM Regressor**: 63 leaves, min_child_samples=20, feature_fraction=0.8.

Advanced models: default params → Optuna TPE Bayesian optimisation [15]. Search spaces: XGBoost — n_estimators (200–1,000), max_depth (3–10), learning_rate (0.01–0.3 log-uniform), subsample (0.5–1.0), colsample_bytree (0.5–1.0); LightGBM — num_leaves (20–150), max_depth (3–12), learning_rate, min_child_samples (5–50), feature_fraction (0.5–1.0).

CV: `TimeSeriesSplit(n_splits=5)`. KFold with shuffle explicitly rejected — future info in training inflates apparent RMSE 20.5% [14]. StandardScaler fit on training only, applied to train+test.

**E. Evaluation Framework**

Held-out 16-day window (Aug 16–31, 2017). Four metrics:

$$\text{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\log(1+\hat{y}_i) - \log(1+y_i))^2}$$

RMSLE preferred: target right-skewed, structural zeros, retail stockout cost > overstock cost → all three favour logarithmic compression + asymmetric under-prediction penalty [8]. MAPE (excludes zero-actual rows), R², RMSE reported. Mean residual exceeding 5% of mean actual sales → systematic bias warranting documentation.

**F. Cannibalization and Promotional Impact Analysis**

Cross-family correlation matrix on residual sales (actual − 28-day rolling mean, net of trend) during `onpromotion > 0` periods. Pearson r < −0.35 → cannibalization candidate; p-values reported. Expected candidates: BEVERAGES vs LIQUOR/WINE/BEER — consistent with EDA cross-family structure.

Promotional lift:

$$\text{lift} = \frac{s_{\text{actual}} - \hat{s}_{\text{counterfactual}}}{\hat{s}_{\text{counterfactual}}}$$

$\hat{s}_{\text{counterfactual}}$ = 4-week rolling pre-promotion mean. Minimum 10 distinct promotion events per acceptance criteria.

**G. Interpretability via SHAP Analysis**

SHAP TreeExplainer on best tree model. Global importance = mean absolute SHAP bar chart [13]. Validation criteria: at least one promotional feature AND at least one temporal/lag feature in top-5 SHAP contributors (convergent finding across [13][17]). Dependence plots for promotion × weekday interaction.

**H. Reproducibility Controls**

`random_state=42`, `PYTHONHASHSEED=42`. Package versions pinned in `requirements.txt`. CSV SHA-256 hashes logged to `data/checksums.txt`. Best model saved via joblib + bit-for-bit reload verification programmatically.

---

## Part III: Key Framing Points

1. **Temporal holdout vs random split**: cite [14][24] — future data contaminates lag features during CV. Demonstrates engineering rigour.
2. **RMSLE vs RMSE**: economic justification (stockout asymmetry) from [8], not mathematical convenience.
3. **Retain zeros**: three-category taxonomy (structural/event-driven/intermittent) distinguishes practitioner from student [9][10].
4. **XGBoost + LightGBM vs neural network**: [3][22] — deep learning gains narrow substantially on horizons ≤ 2 weeks. 16-day horizon critical mediating factor.
5. **Cannibalization analysis**: novel contribution. Most forecasting papers stop at model comparison. Work extends to causal business insight — which promotions shift demand vs create it.
6. **Future Work**: Chronos-2 [23] + TimeGPT [22] zero-shot baselines; Switch-Hurdle [11] for intermittent families; semidefinite hierarchical reconciliation [26]; probabilistic quantile extensions [2]; weekly retraining cadence [29].

### IEEE Formatting Checklist
- Equations numbered: `(1)`, `(2)`, etc.
- Table captions above, figure captions below
- Citations numbered in order of first appearance: `[1]`, `[2]`, etc.
- Methodology section: ~800–1,200 words (conference), ~2,000 words (journal)
- Do NOT describe pandas/sklearn in methodology (→ Implementation Details section or footnote)
- Passive voice: "models were trained", "features were constructed"
- State equations, then reference in prose: "...computed using (1), where ŷᵢ denotes..."

---

_Document: March 2026 | Smart Retail Demand Forecasting — Data Science Semester Project_
