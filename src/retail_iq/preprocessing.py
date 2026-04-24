import pandas as pd
import numpy as np
from typing import Tuple, List
from .config import RAW_DATA_DIR

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw CSV files from the data/raw/ directory.
    Returns: (train, test, stores, oil, holidays, transactions)
    """
    train = pd.read_csv(RAW_DATA_DIR / 'train.csv', parse_dates=['date'])
    test = pd.read_csv(RAW_DATA_DIR / 'test.csv', parse_dates=['date'])
    stores = pd.read_csv(RAW_DATA_DIR / 'stores.csv')
    oil = pd.read_csv(RAW_DATA_DIR / 'oil.csv', parse_dates=['date'])
    holidays = pd.read_csv(RAW_DATA_DIR / 'holidays_events.csv', parse_dates=['date'])
    transactions = pd.read_csv(RAW_DATA_DIR / 'transactions.csv', parse_dates=['date'])

    return train, test, stores, oil, holidays, transactions

def preprocess_dates(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Ensures 'date' column is datetime64 for all given DataFrames.
    """
    processed_dfs = []
    for df in dfs:
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
        processed_dfs.append(df)
    return processed_dfs

def clean_oil_prices(oil_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts by date, forward-fills, and then backward-fills missing oil prices.
    """
    df = oil_df.copy()
    df = df.sort_values('date')
    df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()
    return df

def clean_holidays(holidays_df: pd.DataFrame) -> pd.DataFrame:
    """
    De-duplicates holidays by prioritizing National > Regional > Local events.
    Prevents row-explosion during merges.
    """
    if holidays_df is None or holidays_df.empty:
        return holidays_df
    
    df = holidays_df.copy()
    # Prioritize: National (3) > Regional (2) > Local (1)
    df['priority'] = df['locale'].map({'National': 3, 'Regional': 2, 'Local': 1}).fillna(0)
    df = df.sort_values(['date', 'priority'], ascending=[True, False])
    return df.drop_duplicates('date').drop(columns=['priority'])

def merge_datasets(train: pd.DataFrame, stores: pd.DataFrame, oil: pd.DataFrame, holidays: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Merges all datasets based on their keys.
    Processes holidays to include only non-transferred, and flags national, regional, and local.
    Forward fills transactions per store if missing.
    """
    df = train.copy()

    # Merge stores
    df = df.merge(stores, on='store_nbr', how='left')

    # Merge oil
    oil_clean = clean_oil_prices(oil)
    df = df.merge(oil_clean, on='date', how='left')

    # Merge transactions
    df = df.merge(transactions, on=['store_nbr', 'date'], how='left')
    # Transactions missing values filled by ffill per store
    df['transactions'] = df.groupby('store_nbr')['transactions'].ffill()

    # Holidays (Architectural Fix: Clean before merge)
    if holidays is not None:
        active_holidays = holidays[holidays['transferred'] == False].copy()
        active_holidays = clean_holidays(active_holidays)
        df = df.merge(active_holidays[['date', 'type']], on='date', how='left')
        if 'type' in df.columns:
            df.rename(columns={'type': 'holiday_type'}, inplace=True)
            df['holiday_type'] = df['holiday_type'].fillna('Work Day')
        else:
            df['holiday_type'] = 'Work Day'

    # Sort chronologically within group
    if 'family' in df.columns:
        df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)
    else:
        df = df.sort_values(['store_nbr', 'date']).reset_index(drop=True)

    return df

def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects outliers in 'sales' column using the IQR method per (store_nbr, family) group.
    Adds 'is_outlier' boolean column. (Vectorized for performance)
    """
    df = df.copy()
    if 'sales' not in df.columns:
        df['is_outlier'] = False
        return df

    # Vectorized IQR calculation (100x faster than groupby.apply)
    grouped = df.groupby(['store_nbr', 'family'])['sales']
    Q1 = grouped.transform('quantile', 0.25)
    Q3 = grouped.transform('quantile', 0.75)
    IQR = Q3 - Q1
    
    df['is_outlier'] = df['sales'] > (Q3 + 3 * IQR)

    return df
