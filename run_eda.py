import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from retail_iq.preprocessing import load_raw_data, preprocess_dates, merge_datasets, detect_outliers_iqr
from retail_iq.features import FastFeatureEngineer
from retail_iq.visualization import plot_ts_decomposition, plot_correlation_heatmap, plot_sales_distribution
from retail_iq.config import PLOT_DIR, OUTPUT_DIR

def run_eda():
    print("Loading data...")
    train, test, stores, oil, holidays, tx = load_raw_data()
    train, test, oil, holidays, tx = preprocess_dates([train, test, oil, holidays, tx])
    
    print("Merging data...")
    df = merge_datasets(train, stores, oil, holidays, tx)
    
    print("Engineering features...")
    fe = FastFeatureEngineer(df, transactions=tx, oil_price=oil, holidays=holidays, store_meta=stores)
    fe.add_temporal_features()\
      .add_lag_and_rolling()\
      .add_onpromotion_features()\
      .add_macroeconomic_features()\
      .add_transaction_features()\
      .add_store_metadata()\
      .add_cannibalization_features()
      
    train_features = fe.transform()
    train_features = detect_outliers_iqr(train_features)
    
    eda_dir = OUTPUT_DIR / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    
    # Time-series decomposition
    store_family_samples = train_features.groupby(['store_nbr','family']).size().head(3).index
    for store_nbr, family in store_family_samples:
        print(f"Plotting decomposition for Store {store_nbr}, Family {family}...")
        plot_ts_decomposition(train_features, store_nbr, family, period=7, save_path=str(eda_dir / f'ts_decompose_{store_nbr}_{family}.png'))
        
    # Correlation heatmap
    print("Plotting correlation heatmap...")
    plot_correlation_heatmap(train_features, save_path=str(eda_dir / 'correlation_heatmap.png'))
    
    # Sales distribution
    print("Plotting sales distribution...")
    plot_sales_distribution(train_features, save_path=str(eda_dir / 'sales_distribution.png'))
    
    # Holiday lift
    if 'holiday_type' in train_features.columns:
        print("Plotting holiday lift analysis...")
        plt.figure(figsize=(10,5))
        sns.barplot(x='holiday_type', y='sales', data=train_features.groupby('holiday_type')['sales'].mean().reset_index())
        plt.title("Holiday Lift Analysis")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(eda_dir / "holiday_lift.png")
        plt.close()
    
    # Oil vs Sales
    if 'dcoilwtico' in train_features.columns:
        print("Plotting oil vs sales...")
        daily_sales = train_features.groupby('date')['sales'].sum().reset_index()
        daily_data = daily_sales.merge(oil, on='date', how='left')
        plt.figure(figsize=(12,6))
        sns.scatterplot(x='dcoilwtico', y='sales', data=daily_data)
        plt.title("Oil Price vs Aggregate Sales")
        plt.savefig(eda_dir / "oil_vs_sales.png")
        plt.close()
        
    print("EDA pipeline completed.")

if __name__ == '__main__':
    run_eda()
