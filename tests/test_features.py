import pandas as pd
import pytest

@pytest.fixture
def test_data():
    return pd.read_parquet("data/splits/test.parquet")

def test_rolling_avg_non_negative(test_data):
    assert (test_data['rolling_avg_10'] >= 0).all(), "Sanity check failed: Rolling average contains negative values."

def test_volume_sum_non_negative(test_data):
    assert (test_data['volume_sum_10'] >= 0).all(), "Sanity check failed: Volume sum contains negative values."
