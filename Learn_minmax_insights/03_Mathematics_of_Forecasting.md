# 03 — The Mathematics of Forecasting

## Mental Model

RMSLE = root mean squared error on log1p(sales). log1p compresses large values, making the metric sensitive to **relative** errors. A 10-unit error on a 10-unit sale (100% error) is penalized 10x more than a 10-unit error on a 1000-unit sale (1% error). This matches retail reality: a stockout on a low-volume item matters less than on a high-volume item.

## RMSLE Derivation (Maximum Likelihood)

假設誤差服從 log-normal 分佈：

$$\epsilon_i = \log(y_{pred,i} + 1) - \log(y_{actual,i} + 1) \sim \mathcal{N}(0, \sigma^2)$$

MLE 估計 $\sigma^2$（對數似然函數取極值）：

$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n \epsilon_i^2 = \frac{1}{n}\sum_{i=1}^n (\log(y_{pred}+1) - \log(y_{actual}+1))^2$$

RMSLE = $\sqrt{\hat{\sigma}^2}$ = RMSE on log-space

**為何 log-normal 假設合理於 sales：**
- Sales 為非負、right-skewed（GROCERY I 日均 $10^4$ vs LINGERIE 日均 $10^1$）
- 乘法過程自然產生 log-normal（多獨立因素複合效應）
- 零售需求受天氣、促銷、競爭、假日等因素影響，複合 → log-normal

> **Warning — Zero-Inflation Gap:** Favorita 40-60% rows 為零銷售。log-normal 假設連續分佈，無視 zero-inflation。實際預測時，模型低估零時期真實值。**Two-stage model**（先分類 P(sales>0)，再回歸 E[sales|sales>0]）為正確方向，此 repo 未實現。

## Why RMSLE Over MSE or MAE?

| Metric | Formula | Problem it Solves | Weakness |
|--------|---------|------------------|----------|
| **RMSLE** | `sqrt(mean((log1p(pred) - log1p(actual))²))` | Handles zeros, asymmetric penalty, scale-independence | Assumes log-normal-like error distribution |
| **MSE** | `mean((pred - actual)²)` | Unbiased for Gaussian errors | Over-penalizes large-volume items; infinite gradient at zero |
| **MAE** | `mean(|pred - actual|)` | Robust to outliers | Treats 10-unit error same on 10-unit sale as on 1000-unit sale |
| **MAPE** | `mean(|pred - actual| / actual) × 100` | Percentage terms | Division by zero when actual=0; explodes for small actuals |

**RMSLE chosen**: Store sales is heavy-tailed, zero-inflated, and scale varies across families (GROCERY I vs PLAYERS AND ELECTRONICS). RMSLE treats all families on equal relative footing.

```python
# evaluation.py:L44-L45 — RMSLE implementation
y_pred_clipped = np.clip(y_pred, 0, None)  # no negative predictions
rmsle = float(np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true_clipped)) ** 2)))
```

## GD_Linear: Gradient Descent from Scratch

```python
# models.py:L91-L98 — JAX-compiled update step
@jax.jit
def _step(theta, X, y, lr, l1, l2):
    preds  = X @ theta
    errors = preds - y
    grad   = (2.0/m) * (X.T @ errors) + (l2/m)*theta + (l1/m)*jnp.sign(theta)
    new_theta = theta - lr * grad
    loss      = jnp.mean(errors ** 2)
    return new_theta, loss
```

**Key properties**:
- L1 + L2 both in gradient (not in loss). `+ (l1/m)*sign(theta)` for Lasso-like sparsity, `+ (l2/m)*theta` for Ridge-like shrinkage.
- Target is `log1p(sales)` — model predicts **log-space**, then `expm1()` to recover.
- 1000 iterations default — loss history tracks convergence.
- JAX fallback: if JAX unavailable, pure NumPy loop at `models.py:L151-L165` (3-5x slower).

> **API Contract — CRITICAL:** `predict()` at `models.py:L167-L181` returns `X @ theta` directly — **log-space predictions**. Caller must apply `np.expm1()` to get actual sales. This is undocumented in the docstring. Failure to inverse-transform produces catastrophically wrong RMSLE.

## SeasonalNaive: Persistence Baseline

```python
# models.py:L207-L230
def predict(self, df: pd.DataFrame) -> pd.Series:
    return df.groupby(["store_nbr", "family"])["sales"].shift(self.period)
```

- Zero training. Predicts same-period-last-year sales.
- Used as lower bound — if GD_Linear doesn't beat this, something is wrong.
- Period=365 for yearly seasonality (default).

## Advanced Models: XGBoost + LightGBM with Optuna

```python
# advanced_models.ipynb — Optuna TPE Bayesian search
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50, timeout=600)
```

| Model | Key Hyperparams | CV Strategy |
|-------|----------------|-------------|
| XGBoost | n_estimators(200-1000), max_depth(3-10), lr(0.01-0.3), subsample(0.5-1.0) | TimeSeriesSplit(n_splits=5) |
| LightGBM | num_leaves(20-150), max_depth(3-12), lr(0.01-0.3), min_child_samples(5-50) | TimeSeriesSplit(n_splits=5) |

**Why TimeSeriesSplit?**
```python
# ANTI_LEAKAGE: KFold(shuffle=True) forbidden
# Temporal data must not mix past and future in CV
cv = TimeSeriesSplit(n_splits=5)  # forward-chaining train/test splits
```

> **Optuna Timeout Risk:** `timeout=600` (10 minutes) but `n_trials=50`. If each trial > 12s (XGBoost on large data can be 30s+), search **incompletes** with no early stopping. Increase timeout or reduce n_trials. No partial-result saving in this implementation.

## SHAP Explanation (TreeExplainer)

```python
# evaluation.py:L110-L128 — lazy import avoids 200ms notebook overhead
import shap  # only loaded when generate_shap_summary() called
explainer = shap.Explainer(model)  # TreeExplainer for XGB/LGBM
shap_values = explainer(X_test)
```

- SHAP validates that promo features AND temporal/lag features appear in top-5 importance.
- If only one category appears, the model may be overfitting to a single signal type.
- **TreeExplainer fallback chain** at `evaluation.py:L112-L128`:
  1. `TreeExplainer` (fast, XGB/LGBM native)
  2. If fails → `KernelExplainer` (slow, model-agnostic, 10x+ slower)
  3. If fails → skip silently
  - **Risk:** KernelExplainer on 50 samples is statistically unstable; run repeated times to validate.

> **SHAP Lazy Import Risk:** `import shap` at `evaluation.py:L110` inside the function (not module-top). If `requirements.txt` missing `shap`, `generate_shap_summary()` fails only at **runtime**, not import time. Verify `shap` is listed in dependencies.

## Loss Curve Diagnostics

- **GD_Linear loss not converging**: learning rate too high → loss explodes. Reduce lr.
- **Loss plateaus early**: learning rate too low or insufficient iterations. Increase iterations.
- **Loss jumps**: gradient clipping issue in JAX. Fallback to NumPy.
- **SeasonalNaive RMSLE high for new stores**: no 365-day history available → NaN predictions.
