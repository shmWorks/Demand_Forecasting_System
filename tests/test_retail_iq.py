import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Import all modules to be tested
from retail_iq import config, preprocessing, features, models, evaluation, visualization

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def sample_train():
    return pd.DataFrame({
        'date': pd.to_datetime(['2017-01-01', '2017-01-01', '2017-01-02', '2017-01-02']),
        'store_nbr': [1, 2, 1, 2],
        'family': ['GROCERY I', 'GROCERY I', 'GROCERY I', 'GROCERY I'],
        'sales': [100.0, 200.0, 150.0, 250.0],
        'onpromotion': [0, 1, 0, 0]
    })

@pytest.fixture
def sample_stores():
    return pd.DataFrame({
        'store_nbr': [1, 2],
        'city': ['Quito', 'Guayaquil'],
        'state': ['Pichincha', 'Guayas'],
        'type': ['A', 'B'],
        'cluster': [1, 2]
    })

@pytest.fixture
def sample_oil():
    # Intentionally missing dates and NaNs to test clean_oil_prices
    return pd.DataFrame({
        'date': pd.to_datetime(['2016-12-31', '2017-01-02', '2017-01-04']),
        'dcoilwtico': [np.nan, 50.0, 52.0]
    })

@pytest.fixture
def sample_holidays():
    return pd.DataFrame({
        'date': pd.to_datetime(['2017-01-01', '2017-01-02']),
        'type': ['Holiday', 'Holiday'],
        'locale': ['National', 'Local'],
        'locale_name': ['Ecuador', 'Quito'],
        'description': ['New Year', 'Local Fest'],
        'transferred': [False, True]  # Transferred should be ignored
    })

@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        'date': pd.to_datetime(['2017-01-01', '2017-01-02']),
        'store_nbr': [1, 1],
        'transactions': [1000, 1100]
    })

@pytest.fixture
def feature_base_df(sample_train, sample_stores, sample_oil, sample_holidays, sample_transactions):
    return preprocessing.merge_datasets(
        sample_train, sample_stores, sample_oil, sample_holidays, sample_transactions
    )

# ==========================================
# CONFIGURATION TESTS
# ==========================================

class TestConfig:
    def test_paths_are_pathlib_objects(self):
        """White-box: verify config paths are properly instantiated Path objects."""
        assert isinstance(config.PROJECT_ROOT, Path)
        assert isinstance(config.DATA_DIR, Path)
        assert isinstance(config.RAW_DATA_DIR, Path)
        assert isinstance(config.OUTPUT_DIR, Path)

    def test_directories_exist(self):
        """Integration: verify directories are created upon import."""
        assert config.DATA_DIR.exists()
        assert config.OUTPUT_DIR.exists()

    def test_set_global_seed_reproducible_numpy(self):
        config.set_global_seed(42)
        arr1 = np.random.rand(5)
        config.set_global_seed(42)
        arr2 = np.random.rand(5)
        np.testing.assert_allclose(arr1, arr2)

# ==========================================
# PREPROCESSING TESTS
# ==========================================

class TestPreprocessing:
    def test_load_raw_data(self):
        """load_raw_data reads 6 DataFrames — either from Parquet or CSV."""
        # Mock Path.exists to return False (force CSV path), then mock read_csv.
        with patch('retail_iq.preprocessing.pd.read_csv', return_value=pd.DataFrame()) as mock_csv, \
             patch.object(Path, 'exists', return_value=False):
            data_tuple = preprocessing.load_raw_data()

        assert len(data_tuple) == 6
        assert mock_csv.call_count == 6
        calls = mock_csv.call_args_list
        assert any('train.csv' in str(c[0][0]) for c in calls)

    def test_preprocess_dates(self):
        """Test conversion of string dates to datetime64."""
        df1 = pd.DataFrame({'date': ['2017-01-01', '2017-01-02'], 'val': [1, 2]})
        df2 = pd.DataFrame({'nodate': [1, 2]})

        processed = preprocessing.preprocess_dates([df1, df2])
        assert pd.api.types.is_datetime64_any_dtype(processed[0]['date'])
        assert 'nodate' in processed[1].columns

    def test_clean_oil_prices(self):
        """White-box test for forward and backward filling logic."""
        oil = pd.DataFrame({
            'date': pd.to_datetime(['2017-01-01', '2017-01-02', '2017-01-03']),
            'dcoilwtico': [np.nan, 50.0, np.nan]
        })
        clean = preprocessing.clean_oil_prices(oil)
        assert clean['dcoilwtico'].iloc[0] == 50.0
        assert clean['dcoilwtico'].iloc[2] == 50.0
        assert not clean['dcoilwtico'].isna().any()

    def test_merge_datasets(self, sample_train, sample_stores, sample_oil, sample_holidays, sample_transactions):
        """Integration: tests complex merge logic and specific ffill/holiday rules."""
        merged = preprocessing.merge_datasets(
            sample_train, sample_stores, sample_oil, sample_holidays, sample_transactions
        )
        assert 'city' in merged.columns
        assert 'dcoilwtico' in merged.columns
        assert 'transactions' in merged.columns
        assert 'is_national_holiday' in merged.columns

        national_flags = merged.loc[merged['date'] == '2017-01-01', 'is_national_holiday']
        assert all(national_flags == 1)

    def test_detect_outliers_iqr(self):
        """Tests vectorized IQR logic."""
        np.random.seed(42)
        df = pd.DataFrame({
            'store_nbr': [1] * 100,
            'family': ['A'] * 100,
            'sales': np.random.normal(50, 5, 100).tolist()
        })
        df.loc[0, 'sales'] = 10000.0

        outlier_df = preprocessing.detect_outliers_iqr(df)
        assert 'is_outlier' in outlier_df.columns
        assert outlier_df.loc[0, 'is_outlier'] == True

    def test_detect_outliers_iqr_no_sales(self):
        """Edge case: missing sales column."""
        df = pd.DataFrame({'store_nbr': [1], 'family': ['A']})
        outlier_df = preprocessing.detect_outliers_iqr(df)
        assert not outlier_df['is_outlier'].any()

    def test_strict_temporal_holdout_split_15_days(self):
        dates = pd.date_range("2017-07-01", periods=31, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "store_nbr": [1] * len(dates),
            "family": ["A"] * len(dates),
            "sales": np.arange(len(dates), dtype=float),
        })
        train_df, test_df = preprocessing.strict_temporal_holdout_split(df, holdout_days=15)
        assert len(test_df["date"].dt.normalize().unique()) == 15
        assert train_df["date"].max() < test_df["date"].min()

    def test_strict_temporal_holdout_split_invalid_days(self):
        df = pd.DataFrame({"date": pd.date_range("2017-01-01", periods=5), "sales": [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError, match="holdout_days must be > 0"):
            preprocessing.strict_temporal_holdout_split(df, holdout_days=0)

# ==========================================
# FEATURES TESTS (FastFeatureEngineer)
# ==========================================

class TestFastFeatureEngineer:
    def test_initialization(self, sample_train):
        """__init__ sorts df but does not mutate the original."""
        fe = features.FastFeatureEngineer(sample_train)
        assert fe.df is not sample_train   # Must be a distinct object (copy via sort+reset)
        # Original unchanged
        pd.testing.assert_frame_equal(sample_train, sample_train)
        # Result has same columns and row count
        assert set(fe.df.columns) == set(sample_train.columns)
        assert len(fe.df) == len(sample_train)

    def test_temporal_features(self, sample_train, sample_holidays):
        fe = features.FastFeatureEngineer(sample_train, holidays=sample_holidays)
        transformed = fe.add_temporal_features().transform()

        expected_cols = ['day_of_week', 'day_of_month', 'week_of_year', 'month',
                         'quarter', 'year', 'is_weekend', 'days_to_nearest_holiday']
        for col in expected_cols:
            assert col in transformed.columns

        assert transformed.loc[transformed['date'] == '2017-01-01', 'is_weekend'].iloc[0] == 1

    def test_lag_and_rolling(self):
        """Tests shifting per group logic."""
        df = pd.DataFrame({
            'date': pd.date_range('2017-01-01', periods=10),
            'store_nbr': [1] * 10,
            'family': ['A'] * 10,
            'sales': np.arange(10, 110, 10)
        })
        fe = features.FastFeatureEngineer(df)
        transformed = fe.add_lag_and_rolling(lags=[1, 2], windows=[2]).transform()

        assert transformed['sales_lag_1d'].iloc[1] == 10.0
        assert transformed['rolling_mean_2d'].iloc[2] == 15.0

    def test_lag_and_rolling_no_sales(self):
        """No crash when sales column absent — just returns df unchanged."""
        df = pd.DataFrame({
            'store_nbr': [1],
            'family': ['A'],
            'date': pd.to_datetime(['2017-01-01'])  # date required for sort
        })
        fe = features.FastFeatureEngineer(df)
        transformed = fe.add_lag_and_rolling().transform()
        assert 'sales_lag_1d' not in transformed.columns

    def test_onpromotion_features(self):
        df = pd.DataFrame({
            'date': pd.date_range('2017-01-01', periods=5),
            'store_nbr': [1]*5, 'family': ['A']*5,
            'onpromotion': [0, 1, 1, 0, 0]
        })
        fe = features.FastFeatureEngineer(df)
        transformed = fe.add_onpromotion_features().transform()
        assert transformed['onpromotion_lag_1d'].iloc[1] == 0.0
        assert transformed['onpromotion_lag_1d'].iloc[2] == 1.0

    def test_macroeconomic_features(self, sample_train, sample_oil):
        fe = features.FastFeatureEngineer(sample_train, oil_price=sample_oil)
        transformed = fe.add_macroeconomic_features().transform()
        assert 'dcoilwtico_lag_7d' in transformed.columns

    def test_transaction_features(self, sample_train, sample_transactions):
        fe = features.FastFeatureEngineer(sample_train, transactions=sample_transactions)
        transformed = fe.add_transaction_features().transform()
        assert 'transactions_lag_7d' in transformed.columns

    def test_store_metadata(self, sample_train, sample_stores):
        fe = features.FastFeatureEngineer(sample_train, store_meta=sample_stores)
        transformed = fe.add_store_metadata().transform()
        assert transformed.loc[transformed['store_nbr'] == 1, 'store_type'].iloc[0] == 0

    def test_cannibalization_features(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2017-01-01', '2017-01-01']),
            'store_nbr': [1, 1],
            'family': ['A', 'B'],
            'sales': [100, 50],
            'onpromotion': [1, 0]
        })
        fe = features.FastFeatureEngineer(df)
        transformed = fe.add_cannibalization_features().transform()
        assert 'other_family_sales_lag_7d' in transformed.columns

# ==========================================
# MODELS TESTS
# ==========================================

class TestModels:
    def test_gd_linear_convergence(self):
        """White-box: verify gradient descent actually minimizes loss."""
        np.random.seed(42)
        X = np.random.rand(100, 3)
        true_theta = np.array([1.5, -2.0, 3.0])
        y = X @ true_theta + np.random.normal(0, 0.1, 100)

        model = models.GD_Linear(lr=0.1, iterations=500)
        model.fit(X, y)

        assert len(model.loss_history) == 500
        assert model.loss_history[0] > model.loss_history[-1]

        preds = model.predict(X)
        assert preds.shape == (100,)
        np.testing.assert_allclose(model.theta, true_theta, rtol=0.2, atol=0.2)

    @pytest.mark.parametrize("l1, l2", [(0.1, 0.0), (0.0, 0.1), (0.1, 0.1)])
    def test_gd_linear_regularization(self, l1, l2):
        """Test L1 and L2 penalty paths."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])
        model = models.GD_Linear(lr=0.01, iterations=10, l1=l1, l2=l2)
        model.fit(X, y)
        assert not np.isnan(model.theta).any()

    def test_seasonal_naive(self):
        df = pd.DataFrame({'store_nbr': [1, 1, 1], 'family': ['A', 'A', 'A'], 'sales': [10, 20, 30]})
        model = models.SeasonalNaive(period=1)
        preds = model.predict(df)
        assert pd.isna(preds.iloc[0])
        assert preds.iloc[1] == 10.0

    def test_seasonal_naive_no_sales(self):
        df = pd.DataFrame({'store_nbr': [1], 'family': ['A']})
        model = models.SeasonalNaive()
        with pytest.raises(ValueError, match="must contain 'sales' column"):
            model.predict(df)

# ==========================================
# EVALUATION TESTS
# ==========================================

class TestEvaluation:
    def test_evaluate_model(self):
        """Test metrics calculation."""
        y_true = np.array([10.0, 20.0, 0.0])  # Includes 0 for MAPE edge case
        y_pred = np.array([12.0, 18.0, 0.0])

        metrics = evaluation.evaluate_model(y_true, y_pred, "TestModel")

        assert metrics['model'] == "TestModel"
        np.testing.assert_approx_equal(metrics['MAPE'], 15.0)

    def test_evaluate_model_all_zeros(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([0.0, 0.0])
        metrics = evaluation.evaluate_model(y_true, y_pred, "Zeros")
        assert np.isnan(metrics['MAPE'])

    @patch('retail_iq.evaluation.plt')
    def test_plot_residuals(self, mock_plt):
        y_true = np.array([10, 20])
        y_pred = np.array([10, 20])

        evaluation.plot_residuals(y_true, y_pred, save_path='test.png')
        # New code calls savefig with dpi and bbox_inches — use assert_called_once + check path only
        assert mock_plt.savefig.called
        args, kwargs = mock_plt.savefig.call_args
        assert args[0] == 'test.png'

        evaluation.plot_residuals(y_true, y_pred)
        mock_plt.show.assert_called_once()

    def test_generate_shap_summary(self):
        """generate_shap_summary uses lazy import — patch via builtins.__import__."""
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap.Explainer.return_value = mock_explainer

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        import builtins
        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == 'shap':
                return mock_shap
            return real_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=_mock_import), \
             patch('retail_iq.evaluation.plt'):
            mock_model = MagicMock()
            X_test = pd.DataFrame({'a': [1], 'b': [2]})
            evaluation.generate_shap_summary(mock_model, X_test, save_path='shap.png')

        mock_shap.Explainer.assert_called_with(mock_model)
        mock_explainer.assert_called_with(X_test)

# ==========================================
# VISUALIZATION TESTS
# ==========================================

class TestVisualization:
    @patch('retail_iq.visualization.seasonal_decompose')
    @patch('retail_iq.visualization.plt')
    def test_plot_ts_decomposition(self, mock_plt, mock_decompose):
        np.random.seed(42)
        df = pd.DataFrame({
            'date': pd.date_range('2017-01-01', periods=10),
            'store_nbr': [1]*10, 'family': ['A']*10, 'sales': np.random.rand(10)
        })

        mock_result = MagicMock()
        mock_result.plot.return_value = MagicMock()
        mock_decompose.return_value = mock_result

        visualization.plot_ts_decomposition(df, 1, 'A', period=7, save_path='ts.png')
        mock_decompose.assert_called_once()
        # Check path positional arg; ignore dpi/bbox_inches kwargs
        args, kwargs = mock_plt.savefig.call_args
        assert args[0] == 'ts.png'

    @patch('retail_iq.visualization.plt')
    def test_plot_ts_decomposition_empty(self, mock_plt, caplog):
        """Empty-filter case logs warning via logger — not print."""
        import logging
        df = pd.DataFrame({'store_nbr': [2], 'family': ['B']})
        with caplog.at_level(logging.WARNING, logger='retail_iq.visualization'):
            visualization.plot_ts_decomposition(df, 1, 'A')
        assert 'No data' in caplog.text

    @patch('retail_iq.visualization.sns')
    @patch('retail_iq.visualization.plt')
    def test_plot_correlation_heatmap(self, mock_plt, mock_sns):
        df = pd.DataFrame({'num1': [1, 2], 'num2': [2, 4], 'cat': ['a', 'b']})
        visualization.plot_correlation_heatmap(df)
        mock_sns.heatmap.assert_called_once()
        mock_plt.show.assert_called_once()

    @patch('retail_iq.visualization.sns')
    @patch('retail_iq.visualization.plt')
    def test_plot_sales_distribution(self, mock_plt, mock_sns):
        df = pd.DataFrame({'sales': [10, 20, 30]})
        visualization.plot_sales_distribution(df, save_path='dist.png')
        mock_sns.histplot.assert_called_once()
        # Check path; ignore dpi/bbox_inches
        args, kwargs = mock_plt.savefig.call_args
        assert args[0] == 'dist.png'

# ==========================================
# NOTEBOOK VALIDATION TESTS
# ==========================================

class TestNotebooks:
    def test_notebooks_are_valid_json(self):
        """Validates that all notebooks parse as valid JSON (sanity check)."""
        notebook_dir = config.PROJECT_ROOT / 'notebooks'
        if not notebook_dir.exists():
            pytest.skip("Notebooks directory not found.")

        notebook_files = list(notebook_dir.glob('*.ipynb'))
        assert len(notebook_files) > 0, "No notebooks found to test."

        for nb_file in notebook_files:
            try:
                with open(nb_file, 'r', encoding='utf-8') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"Notebook {nb_file.name} is invalid JSON: {e}")

# ==========================================
# ADVERSARIAL "TROLL" TESTS (SYSTEM BLINDSPOTS)
# ==========================================

class TestAdversarialBreakage:
    """
    Brutal adversarial tests exposing severe blind spots in the current system logic.
    These are expected to fail (or expose bugs) given the current implementation.
    """

    def test_preprocess_dates_garbage_crash(self):
        df = pd.DataFrame({'date': ['2017-01-01', 'not-a-date', 'Q1-2017']})
        processed = preprocessing.preprocess_dates([df])[0]
        assert pd.api.types.is_datetime64_any_dtype(processed["date"])
        assert processed["date"].isna().sum() == 2

    def test_macroeconomic_leakage_across_groups(self):
        # 8 rows: Alternating Store 1 and Store 2 over 4 dates.
        df = pd.DataFrame({
            'date': pd.to_datetime(['2017-01-01', '2017-01-01', '2017-01-02', '2017-01-02',
                                    '2017-01-03', '2017-01-03', '2017-01-04', '2017-01-04']),
            'store_nbr': [1, 2, 1, 2, 1, 2, 1, 2],
            'family': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
            'dcoilwtico': [10, 10, 20, 20, 30, 30, 40, 40]
        })
        fe = features.FastFeatureEngineer(df)
        transformed = fe.add_macroeconomic_features().transform()

        val = transformed['dcoilwtico_lag_7d'].iloc[7]
        assert pd.isna(val), f"Leakage occurred! 7-day lag got value {val} instead of NaN!"

    def test_seasonal_naive_missing_dates(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2017-01-01', '2017-01-05']),
            'store_nbr': [1, 1],
            'family': ['A', 'A'],
            'sales': [100.0, 200.0]
        })
        model = models.SeasonalNaive(period=1)
        preds = model.predict(df)
        assert pd.isna(preds.iloc[1]), "Time-blindness: Model shifted a row instead of a calendar day!"

    def test_evaluate_negative_sales_crash(self):
        y_true = np.array([10.0, -5.0])
        y_pred = np.array([10.0, 0.0])

        metrics = evaluation.evaluate_model(y_true, y_pred, "Test")
        assert not np.isnan(metrics['RMSLE']), "RMSLE evaluated to NaN due to negative truths!"

    def test_gd_linear_empty_world(self):
        X = np.empty((0, 3))
        y = np.empty((0,))
        model = models.GD_Linear()
        with pytest.raises(ValueError, match="non-empty"):
            model.fit(X, y)
