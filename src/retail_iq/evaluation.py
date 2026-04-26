"""Model evaluation utilities for Retail-IQ.

Note: ``import shap`` is lazy (inside generate_shap_summary) — avoids ~200ms
startup overhead on every notebook import of evaluate_model.
"""

"""Before:
Pure statistical evaluation
Treats all errors equally
After:
✅ Business-aware (over vs under prediction)
✅ Financial-aware (margin weighting)
✅ Still compatible with your pipeline"""


from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    margin: Optional[np.ndarray] = None,
    c_over: float = 1.0,
    c_under: float = 2.0,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics including business-aware asymmetric loss
    and optional margin-weighted errors.

    Args:
        y_true: Ground-truth values
        y_pred: Predictions
        model_name: Name for logging
        margin: Optional margin/importance weights per sample
        c_over: Cost of overprediction
        c_under: Cost of underprediction

    Returns:
        Dict of metrics
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # 🔒 Clip negatives (important for stability)
    y_pred_clipped = np.clip(y_pred, 0, None)
    y_true_clipped = np.clip(y_true, 0, None)

    if np.any(y_true < 0):
        logger.warning("evaluate_model: negative y_true detected; clipping to 0.")

    # -------------------------
    # Standard metrics
    # -------------------------
    rmsle = float(np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true_clipped)) ** 2)))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    mask = y_true > 0
    mape = (
        float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100)
        if mask.sum() > 0 else float("nan")
    )

    r2 = float(r2_score(y_true, y_pred))

    # -------------------------
    # 🔥 Asymmetric loss (IMPORTANT)
    # -------------------------
    over = np.maximum(y_pred - y_true, 0)
    under = np.maximum(y_true - y_pred, 0)
    asym_loss = float(np.mean(c_over * over + c_under * under))

    # -------------------------
    # 💰 Margin-weighted error
    # -------------------------
    if margin is not None:
        margin = np.asarray(margin, dtype=float)
        weights = margin / (np.sum(margin) + 1e-8)  # normalize

        weighted_mae = float(np.sum(weights * np.abs(y_true - y_pred)))
        weighted_rmse = float(np.sqrt(np.sum(weights * (y_true - y_pred) ** 2)))
    else:
        weighted_mae = float("nan")
        weighted_rmse = float("nan")

    # -------------------------
    # Print results
    # -------------------------
    print(
        f"{model_name}: "
        f"RMSLE={rmsle:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, R²={r2:.4f}, "
        f"AsymLoss={asym_loss:.2f}, "
        f"W-MAE={weighted_mae:.2f}, W-RMSE={weighted_rmse:.2f}"
    )

    return {
        "model": model_name,
        "RMSLE": rmsle,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "AsymmetricLoss": asym_loss,
        "Weighted_MAE": weighted_mae,
        "Weighted_RMSE": weighted_rmse,
    }

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot residuals vs predicted values to check for systematic bias.

    Args:
        y_true:    Ground-truth values.
        y_pred:    Predicted values.
        save_path: If given, save figure here instead of displaying.
    """
    residuals = y_true - y_pred
    mean_residual = np.mean(residuals)
    mean_actual = np.mean(y_true)
    bias_pct = (mean_residual / mean_actual) * 100 if mean_actual != 0 else float("nan")

    logger.info("Mean residual: %.4f (%.2f%% of mean actual)", mean_residual, bias_pct)
    print(f"Mean residual: {mean_residual:.4f} ({bias_pct:.2f}% of mean actual)")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, s=5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Sales")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Sales")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def generate_shap_summary(
    model: Any,
    X_test: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Generate SHAP summary bar plot for a tree model.

    Args:
        model:     Fitted tree model (XGBoost, LightGBM, etc.).
        X_test:    Feature DataFrame used for SHAP values.
        save_path: If given, save figure here instead of displaying.

    Note:
        ``shap`` is imported lazily here — avoids 200ms startup cost on notebooks
        that only import evaluate_model.
    """
    import shap  # Lazy import — only paid when this function is called

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
    except Exception as exc:
        # Try KernelExplainer as fallback for corrupted models or version conflicts.
        # This is slower but more robust to serialization issues.
        try:
            logger.warning(
                "generate_shap_summary: TreeExplainer failed (%s), "
                "falling back to KernelExplainer (slower).",
                str(exc)[:100],
            )
            explainer = shap.KernelExplainer(model.predict, X_test[:min(50, len(X_test))])
            shap_values = explainer.shap_values(X_test)
        except Exception as exc2:  # pragma: no cover - both methods failed
            logger.warning("generate_shap_summary skipped: both explainers failed: %s", exc2)
            return

    plt.figure(figsize=(12, 8))
    try:
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    except Exception as exc:
        logger.warning("generate_shap_summary: summary_plot failed, using waterfall: %s", exc)
        try:
            # Try simpler plotting as last resort
            shap.plots.force(explainer.expected_value, shap_values[:10], X_test[:10], show=False)
        except Exception:
            logger.error("generate_shap_summary: all plotting methods failed")
            plt.close()
            return

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
