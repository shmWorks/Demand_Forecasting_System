"""Forecasting models for Retail-IQ — implemented from scratch.

GD_Linear: Gradient descent linear regression using JAX @jax.jit.
    - Algorithm is from-scratch (no sklearn.LinearRegression).
    - JAX is the computational backend: XLA compiles the update step.
    - Public interface (.fit / .predict) is identical to the NumPy version.
    - Backward-compatible: if JAX unavailable, falls back to NumPy.

SeasonalNaive: Persistence baseline — predict last-period same-weekday value.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JAX availability probe — done once at import time, cached
# ---------------------------------------------------------------------------
def _try_import_jax() -> tuple[Any, Any, bool]:
    """Return (jax, jnp, jax_available).  Silent if JAX not installed."""
    try:
        import jax
        import jax.numpy as jnp
        logger.info("JAX backend: %s  |  devices: %s", jax.default_backend(), jax.devices())
        return jax, jnp, True
    except ImportError:
        logger.warning("JAX not installed — GD_Linear will use NumPy fallback (~3-5x slower).")
        return None, None, False


_jax, _jnp, _JAX_AVAILABLE = _try_import_jax()


# ---------------------------------------------------------------------------
# GD_Linear
# ---------------------------------------------------------------------------
class GD_Linear:
    """Linear regression via gradient descent — from scratch, XLA-compiled.

    The algorithm:
        theta ← theta - lr * grad
        grad  = (2/m) * Xᵀ(Xθ - y) + (l2/m)*θ + (l1/m)*sign(θ)

    JAX's @jax.jit compiles the update step to an XLA kernel (CPU or GPU).
    If JAX is unavailable, falls back to a pure NumPy loop automatically.

    Constraints:
        - No sklearn.LinearRegression or equivalent.
        - JAX is the backend, not the algorithm.

    Args:
        lr:           Learning rate.  Default 0.001.
        iterations:   Number of gradient steps.  Default 1000.
        l1:           L1 regularisation coefficient.  Default 0.0.
        l2:           L2 regularisation coefficient.  Default 0.0.
        random_state: Seed for reproducibility (SPEC: 42).
    """

    def __init__(
        self,
        lr: float = 0.001,
        iterations: int = 1000,
        l1: float = 0.0,
        l2: float = 0.0,
        random_state: int = 42,
    ) -> None:
        self.lr = lr
        self.iterations = iterations
        self.l1 = l1
        self.l2 = l2
        self.random_state = random_state
        self.theta: np.ndarray | None = None
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------
    # JAX update step (compiled once, reused for all iterations)
    # ------------------------------------------------------------------
    @staticmethod
    def _make_jax_step():
        """Build and return a jit-compiled update step function."""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def _step(theta, X, y, lr, l1, l2):
            """XLA-compiled gradient descent step.  Pure function, no side effects."""
            m = X.shape[0]
            preds  = X @ theta
            errors = preds - y
            grad   = (2.0 / m) * (X.T @ errors) + (l2 / m) * theta + (l1 / m) * jnp.sign(theta)
            new_theta = theta - lr * grad
            loss      = jnp.mean(errors ** 2)
            return new_theta, loss

        return _step

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GD_Linear":
        """Train using gradient descent.

        Args:
            X: Feature matrix (n_samples, n_features).  Should be standardized.
            y: Target vector (n_samples,).  Apply log1p externally.

        Returns:
            self (method chaining).
        """
        np.random.seed(self.random_state)
        self.loss_history = []

        if _JAX_AVAILABLE:
            self._fit_jax(X, y)
        else:
            self._fit_numpy(X, y)

        return self

    def _fit_jax(self, X: np.ndarray, y: np.ndarray) -> None:
        """JAX-accelerated training loop."""
        jnp = _jnp
        step = self._make_jax_step()

        X_j = jnp.array(X, dtype=jnp.float32)
        y_j = jnp.array(y, dtype=jnp.float32)
        theta = jnp.zeros(X_j.shape[1], dtype=jnp.float32)

        lr  = float(self.lr)
        l1  = float(self.l1)
        l2  = float(self.l2)

        for _ in range(self.iterations):
            theta, loss = step(theta, X_j, y_j, lr, l1, l2)
            self.loss_history.append(float(loss))

        self.theta = np.array(theta)

    def _fit_numpy(self, X: np.ndarray, y: np.ndarray) -> None:
        """Pure NumPy fallback training loop (no JAX dependency)."""
        m, n = X.shape
        theta = np.zeros(n, dtype=np.float64)
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        for _ in range(self.iterations):
            preds  = X @ theta
            errors = preds - y
            grad   = (2.0 / m) * (X.T @ errors) + (self.l2 / m) * theta + (self.l1 / m) * np.sign(theta)
            theta  -= self.lr * grad
            self.loss_history.append(float(np.mean(errors ** 2)))

        self.theta = theta

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Prediction array (n_samples,) as numpy float64.

        Raises:
            RuntimeError: If called before fit().
        """
        if self.theta is None:
            raise RuntimeError("Call fit() before predict().")
        return X.astype(np.float64) @ self.theta.astype(np.float64)


# ---------------------------------------------------------------------------
# SeasonalNaive
# ---------------------------------------------------------------------------
class SeasonalNaive:
    """Seasonal naive baseline: predict last season's same-period value.

    Zero training required.  Used as a lower-bound benchmark.

    Args:
        period: Seasonal lag in days.  Default 7 (weekly).
    """

    def __init__(self, period: int = 7) -> None:
        self.period = period

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> "SeasonalNaive":
        """No-op.  SeasonalNaive requires no training.

        Returns:
            self (for API compatibility).
        """
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Shift sales by 'period' days within each (store_nbr, family) group.

        Args:
            df: DataFrame with 'store_nbr', 'family', 'sales' columns.
                Must be sorted by date within each group.

        Returns:
            Series of lagged sales values (NaN where lag exceeds history).

        Raises:
            ValueError: If 'sales' column is missing.
        """
        if "sales" not in df.columns:
            raise ValueError("DataFrame must contain 'sales' column.")
        return df.groupby(["store_nbr", "family"])["sales"].shift(self.period)
