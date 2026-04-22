import numpy as np

class GD_Linear:
    """
    Linear Regression with Gradient Descent (implemented from scratch using NumPy).
    Target should be log1p(y) externally or handled via wrappers if needed.
    """
    def __init__(self, lr=0.001, iterations=1000, l1=0.0, l2=0.0, random_state=42):
        self.lr = lr
        self.iterations = iterations
        self.l1 = l1
        self.l2 = l2
        self.random_state = random_state
        self.theta = None
        self.loss_history = []

    def fit(self, X, y):
        # Set seed for consistency
        np.random.seed(self.random_state)

        m, n = X.shape
        self.theta = np.zeros(n)
        self.loss_history = []

        for _ in range(self.iterations):
            preds = X @ self.theta
            errors = preds - y
            # Gradient with L1 and L2 regularization
            grad = (2/m) * (X.T @ errors) + (self.l2/m) * self.theta + (self.l1/m) * np.sign(self.theta)
            self.theta -= self.lr * grad
            self.loss_history.append(np.mean(errors**2))

        return self

    def predict(self, X):
        return X @ self.theta


class SeasonalNaive:
    """
    Seasonal Naive predictor (persistence model — predict last year's same-week value).
    Assuming daily data and 365-day seasonality.
    """
    def __init__(self, period=365):
        self.period = period

    def fit(self, X, y=None):
        # Seasonal Naive doesn't require fitting
        pass

    def predict(self, df):
        """
        Assumes df is passed directly and has grouping columns (store_nbr, family)
        and 'sales' column to shift. In practice, prediction logic is done via shifting
        in the feature pipeline. This is a helper wrapper.
        """
        if 'sales' not in df.columns:
            raise ValueError("Dataframe must contain 'sales' column for Seasonal Naive shift.")

        # Shift 365 days
        y_pred = df.groupby(['store_nbr', 'family'])['sales'].shift(self.period)
        return y_pred
