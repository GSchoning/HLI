import numpy as np
import pandas as pd
import pytest
from libraries.ES import LMEnsembleSmoother

def dummy_model(params):
    return {"obs1": params["p1"] * 2}

@pytest.fixture
def smoother():
    param_df = pd.DataFrame({
        "prior_mean": [1.0],
        "prior_std": [0.1]
    }, index=["p1"])

    obs_df = pd.DataFrame({
        "value": [2.0],
        "std": [0.1]
    }, index=["obs1"])

    # Instantiate the Ensemble Smoother
    # We set num_ensemble to 10 for basic testing
    return LMEnsembleSmoother(dummy_model, param_df, obs_df, num_ensemble=10)

def test_check_failures_clean(smoother):
    # A clean matrix with standard values
    # Shape is (num_ensemble, nobs_total) - here (10, 1)
    S_good = np.ones((10, 1)) * 2.0
    # Should return True and S should remain unchanged
    result = smoother._check_failures(S_good)
    assert result is True
    assert np.allclose(S_good, np.ones((10, 1)) * 2.0)

def test_check_failures_with_nan(smoother):
    # A matrix containing NaN values
    S_nan = np.ones((10, 1)) * 2.0
    S_nan[0, 0] = np.nan
    S_nan[1, 0] = np.nan

    # safe runs mean = 2.0
    result = smoother._check_failures(S_nan)

    assert result is True
    # The NaN values should have been replaced with the mean of the safe runs
    assert not np.isnan(S_nan).any()
    assert np.allclose(S_nan, np.ones((10, 1)) * 2.0)

def test_check_failures_with_inf(smoother):
    # A matrix containing Inf values
    S_inf = np.ones((10, 1)) * 2.0
    S_inf[0, 0] = np.inf

    result = smoother._check_failures(S_inf)

    assert result is True
    assert not np.isinf(S_inf).any()
    assert np.allclose(S_inf, np.ones((10, 1)) * 2.0)

def test_check_failures_below_safe_min(smoother):
    # A matrix containing values below safe_min
    S_min = np.ones((10, 1)) * 2.0
    S_min[0, 0] = -2e9

    result = smoother._check_failures(S_min, safe_min=-1e9, safe_max=1e9)

    assert result is True
    # The extreme low value should be replaced by mean of others (2.0)
    assert np.allclose(S_min, np.ones((10, 1)) * 2.0)

def test_check_failures_above_safe_max(smoother):
    # A matrix containing values above safe_max
    S_max = np.ones((10, 1)) * 2.0
    S_max[0, 0] = 2e9

    result = smoother._check_failures(S_max, safe_min=-1e9, safe_max=1e9)

    assert result is True
    # The extreme high value should be replaced by mean of others (2.0)
    assert np.allclose(S_max, np.ones((10, 1)) * 2.0)

def test_check_failures_all_bad(smoother):
    # A matrix where all rows are bad
    S_all_bad = np.ones((10, 1)) * np.nan

    result = smoother._check_failures(S_all_bad)

    # Should return False since there are no good indices to compute mean from
    assert result is False

def test_check_failures_all_bad_minmax(smoother):
    # All rows are bad due to exceeding bounds
    S_all_bad = np.ones((10, 1)) * 2e9

    result = smoother._check_failures(S_all_bad, safe_min=-1e9, safe_max=1e9)

    # Should return False
    assert result is False
