import pytest
import pandas as pd
import numpy as np
from retail_iq.models import GD_Linear
from retail_iq.features import FastFeatureEngineer
from retail_iq.evaluation import evaluate_model

def test_gd_linear():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    y_log = np.log1p(y)

    model = GD_Linear(lr=0.01, iterations=1000)
    model.fit(X, y_log)
    preds = np.expm1(model.predict(X))

    assert len(preds) == len(y)

def test_fast_feature_engineer():
    df = pd.DataFrame({
        'date': pd.date_range('2017-08-01', '2017-08-10'),
        'store_nbr': [1] * 10,
        'family': ['GROCERY I'] * 10,
        'sales': np.random.rand(10) * 100
    })

    fe = FastFeatureEngineer(df)
    fe.add_temporal_features()
    transformed = fe.transform()

    assert 'day_of_week' in transformed.columns
    assert 'month' in transformed.columns
    assert 'year' in transformed.columns
