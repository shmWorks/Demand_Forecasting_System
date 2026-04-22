import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
    """
    Evaluates the given predictions against truth, calculating RMSLE, RMSE, MAPE, and R2.
    """
    # Fix negative predictions for log and RMSLE
    y_pred_clipped = np.clip(y_pred, 0, None)

    # RMSLE uses log1p
    rmsle = np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true)) ** 2))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE: exclude zero actuals to avoid division by zero
    mask = y_true > 0
    if np.sum(mask) > 0:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    else:
        mape = np.nan

    r2 = r2_score(y_true, y_pred)

    print(f"{model_name}: RMSLE={rmsle:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, R²={r2:.4f}")

    return {
        'model': model_name,
        'RMSLE': rmsle,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """
    Plots residuals to check for bias.
    """
    residuals = y_true - y_pred
    mean_residual = np.mean(residuals)
    mean_actual = np.mean(y_true)

    print(f"Mean residual: {mean_residual:.4f} ({(mean_residual/mean_actual)*100:.2f}% of mean actual)")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Sales')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def generate_shap_summary(model, X_test: pd.DataFrame, save_path: str = None):
    """
    Generates and saves SHAP summary plot for tree models.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
