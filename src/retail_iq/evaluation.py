"""Model evaluation utilities for Retail-IQ.

Note: ``import shap`` is lazy (inside generate_shap_summary) — avoids ~200ms
startup overhead on every notebook import of evaluate_model.
"""
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
) -> Dict[str, Any]:
    """Compute and print RMSLE, RMSE, MAPE, and R² for a set of predictions.

    Negative predictions are clipped to 0 for RMSLE/log computation.
    MAPE excludes zero actuals to avoid division by zero.

    Args:
        y_true:     Ground-truth sales values.
        y_pred:     Predicted sales values (may be negative — will be clipped).
        model_name: Label printed in the output line.

    Returns:
        Dict with keys 'model', 'RMSLE', 'RMSE', 'MAPE', 'R2'.
    """
    y_pred_clipped = np.clip(y_pred, 0, None)

    # RMSLE — primary metric per SPEC (handles zeros, asymmetric penalty)
    rmsle = float(np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true)) ** 2)))

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # MAPE — exclude zero actuals
    mask = y_true > 0
    mape = float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100) if mask.sum() > 0 else float("nan")

    r2 = float(r2_score(y_true, y_pred))

    print(f"{model_name}: RMSLE={rmsle:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, R²={r2:.4f}")

    return {"model": model_name, "RMSLE": rmsle, "RMSE": rmse, "MAPE": mape, "R2": r2}


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

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
