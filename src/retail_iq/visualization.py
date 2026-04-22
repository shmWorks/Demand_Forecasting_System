import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from typing import Optional

def plot_ts_decomposition(df: pd.DataFrame, store_nbr: int, family: str, period: int = 365, save_path: Optional[str] = None):
    """
    Plots seasonal decomposition for a specific store and family.
    """
    subset = df[(df['store_nbr'] == store_nbr) & (df['family'] == family)]
    if subset.empty:
        print(f"No data for store_nbr={store_nbr}, family={family}")
        return

    ts = subset.set_index('date').sort_index()
    if 'sales' not in ts.columns:
        print("Missing 'sales' column.")
        return

    result = seasonal_decompose(ts['sales'].fillna(0), model='additive', period=period)

    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"Time-Series Decomposition: Store {store_nbr}, Family {family}", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plots correlation heatmap for numeric features.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap of Numerical Features")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_sales_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plots sales distribution.
    """
    if 'sales' not in df.columns:
        return

    plt.figure(figsize=(10, 5))
    sns.histplot(df['sales'], bins=50, kde=True)
    plt.title("Distribution of Sales")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
