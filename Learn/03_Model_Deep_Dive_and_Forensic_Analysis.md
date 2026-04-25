# 03. Model Deep Dive and Forensic Analysis: Deconstructing `GD_Linear`

## 1. Intuition: Linear Regression from Scratch

The foundational model in this repository is `GD_Linear`, a custom implementation of Linear Regression optimized via Gradient Descent, completely eschewing standard libraries like `scikit-learn` in favor of raw matrix math.

Linear regression operates on the premise that the target variable ($y$) is a linear combination of the features ($X$) plus an error term.
$y = X\theta + \epsilon$

To find the optimal weights ($\theta$), we minimize the Mean Squared Error (MSE) loss function. Instead of computing the closed-form Normal Equation (which requires an expensive $O(n^3)$ matrix inversion), we iteratively update $\theta$ by taking steps in the direction of the negative gradient of the loss function.

## 2. Implementation: The JAX-Accelerated Engine

The class `GD_Linear` in `src/retail_iq/models.py` implements this.

```python
grad = (2.0 / m) * (X.T @ errors) + (l2 / m) * theta + (l1 / m) * jnp.sign(theta)
```
[Source: `src/retail_iq/models.py:L74`]

**Key technical decisions:**
1.  **Gradient Descent Step:** It computes the gradients via the dot product of the transposed feature matrix and the error vector, scaled by $\frac{2}{m}$.
2.  **Regularization:** It seamlessly incorporates both L1 (Lasso) and L2 (Ridge) penalties directly into the gradient calculation, preventing overfitting by squashing weights towards zero.
3.  **JAX XLA Compilation:** By wrapping the `_step` function in `@jax.jit`, the Python loops are completely bypassed. JAX traces the operations and compiles them into a highly optimized XLA (Accelerated Linear Algebra) kernel that runs natively on the CPU (or GPU, if available). This provides orders of magnitude speedup over raw NumPy loops.

## 3. Forensic Critique: The Hidden Bottlenecks

While the JAX compilation is an impressive piece of low-level optimization, an architectural audit reveals critical flaws that limit its effectiveness in a production retail environment.

### Flaw 1: The Standard Scaler Contract
Gradient descent is notoriously sensitive to feature scaling. If `dcoilwtico` is ranging from 40 to 100, while `is_weekend` is 0 or 1, the gradient landscape becomes a narrow, elongated valley. The algorithm will oscillate wildly or require a microscopic learning rate.
The docstring states: `X: Feature matrix (n_samples, n_features). Should be standardized.` [Source: `src/retail_iq/models.py:L82`]
However, **the model does not enforce this**. Relying on the user to externally scale features is a massive data leakage and deployment risk. The model object should own its scaling logic to prevent skew during inference.

### Flaw 2: The Batch Processing Illusion
The `_step` function computes the gradient across the *entire* dataset `X` simultaneously. [Source: `src/retail_iq/models.py:L74`]
For 3 million rows and 50 features, `X` takes up a significant chunk of RAM. JAX will attempt to load the entire matrix into the accelerator's memory. This is **Batch Gradient Descent**, not Stochastic or Mini-Batch. As the dataset scales to 100 million rows, this implementation will hard-crash with an Out-Of-Memory (OOM) error.

### Flaw 3: Static Compilation Rigidity
Because `@jax.jit` compiles the graph based on the *shape* of the input arrays, passing a differently sized batch during inference or training will trigger a costly re-compilation.

## 4. Sovereign Extension: The True Production Regressor

To elevate this model to a robust, Top 0.000001% implementation, we must abandon naïve batch processing.

### Step-by-Step Actionable Insights

*   **Insight 1 (Mini-Batch & Stochasticity):** Rewrite the training loop to implement Mini-Batch Gradient Descent. Yield chunks of `X` and `y` (e.g., 10,000 rows at a time). This provides stochasticity (helping escape local minima) and bounds memory usage to $O(1)$ relative to dataset size.
*   **Insight 2 (Internal State Encapsulation):** The model class must maintain an internal state for feature means and standard deviations. It should automatically `fit_transform` during `.fit()` and apply the learned scalars during `.predict()`.
*   **Insight 3 (Adaptive Learning Rates):** Hardcoding a fixed learning rate (`self.lr`) is inefficient. Implement a learning rate scheduler (e.g., exponential decay or Adam optimizer logic) inside the JAX kernel to accelerate convergence in the early epochs while fine-tuning in the later ones.