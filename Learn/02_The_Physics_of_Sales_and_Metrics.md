# 02. The Physics of Sales and Metrics: A Mathematical Autopsy

## 1. Intuition: Why RMSLE?

In retail forecasting, the target variable (`sales`) exhibits significant right-skewness. You have many days with 0 or 5 sales, and occasionally days with 5,000 sales due to promotions or holidays.

The standard Root Mean Squared Error (RMSE) metric penalizes large absolute errors heavily. If a model predicts 0 when the truth is 5, the squared error is 25. If a model predicts 4000 when the truth is 5000, the squared error is 1,000,000. RMSE forces the model to obsess over high-volume days, completely ignoring the low-volume "long tail" products.

Root Mean Squared Logarithmic Error (RMSLE) fixes this by computing the RMSE on the $\log(1 + x)$ transformed target.
$\text{RMSLE} = \sqrt{\frac{1}{n} \sum (\log(p_i + 1) - \log(a_i + 1))^2}$

RMSLE measures *relative* (percentage) error, not absolute error. The penalty for predicting 0 instead of 5 is similar to predicting 4000 instead of 5000. Furthermore, RMSLE imposes an **asymmetric penalty**: under-predicting is penalized slightly more than over-predicting.

## 2. Implementation: The Evaluation Pipeline

In `src/retail_iq/evaluation.py:L26`:
```python
y_pred_clipped = np.clip(y_pred, 0, None)
y_true_clipped = np.clip(y_true, 0, None)
# RMSLE — primary metric per SPEC (handles zeros, asymmetric penalty)
rmsle = float(np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true_clipped)) ** 2)))
```
[Source: `src/retail_iq/evaluation.py:L31`]

The implementation correctly clips negative predictions to 0 before the `log1p` transform (since log of a negative number is undefined). It also computes MAPE (Mean Absolute Percentage Error), strictly filtering out true zeros to avoid division-by-zero explosions.

## 3. Forensic Critique: The Illusion of Statistical Accuracy

While RMSLE is a mathematically sound choice for the Kaggle *Favorita* competition, its strict minimization leads to a catastrophic divergence from **Net Business Profit**.

**The Gap Between Statistics and Business Utility:**
1.  **RMSLE ignores margin:** An RMSLE-optimized model treats a 10% error on low-margin toilet paper exactly the same as a 10% error on high-margin prime beef.
2.  **The Asymmetry Flaw:** RMSLE natively penalizes under-prediction slightly more than over-prediction. In retail, the cost of under-predicting (stockout, lost revenue) is rarely equal to the cost of over-predicting (spoilage, inventory holding cost). For highly perishable goods (e.g., milk), over-predicting destroys margin. For durables, under-predicting destroys customer loyalty.
3.  **Gaming the Metric:** Because models train on MSE but we evaluate on RMSLE, the standard technique is to log-transform the target *before* training (`np.log1p(y)`), fit an MSE-based regressor (like `GD_Linear` or XGBoost), and exponentiate the predictions (`np.expm1(y_pred)`). The model is mathematically oblivious to the absolute units.

**Code Reality:** The system reports RMSLE, RMSE, MAPE, and $R^2$, but nowhere in the evaluation module is there a calculation for "Dollars Lost due to Mis-forecasting".

## 4. Sovereign Extension: From Accuracy to ROI

To transform this system from an academic exercise into a C-suite prescriptive engine, we must construct a custom loss function that bridges the statistical-business gap.

### Step-by-Step Actionable Insights

*   **Insight 1 (The Asymmetric Profit Loss):** We need to discard purely symmetric statistical metrics during the final model selection. Implement a custom asymmetric loss function for XGBoost/LightGBM.
    $L(y, \hat{y}) = c_{over} \max(\hat{y} - y, 0) + c_{under} \max(y - \hat{y}, 0)$
    Where $c_{over}$ is the cost of spoilage/holding, and $c_{under}$ is the opportunity cost of a stockout.
*   **Insight 2 (Margin-Weighted Evaluation):** Extend `evaluate_model` to accept a `margin` vector. Weight the errors not just by relative volume, but by the actual dollar value at risk.
*   **Insight 3 (The Zero-Inflated Problem):** Kaggle datasets have many structural zeros (days a product simply isn't stocked). MAPE explodes, and RMSLE gets noisy. We should decouple the problem: Train a classification model to predict $P(\text{sales} > 0)$, and a regression model to predict volume given it is stocked.