import numpy as np
import pytest
from src.models.arps import (
    exponential_decline,
    hyperbolic_decline,
    modified_arps,
    stretched_exponential,
    ArpsModeler,
)

def test_exponential_decline():
    t = np.array([0, 1, 2])
    qi = 1000.0
    di = 0.1
    res = exponential_decline(t, qi, di)
    assert res[0] == pytest.approx(1000.0)
    assert res[1] == pytest.approx(1000.0 * np.exp(-0.1))
    assert res[2] < res[1]

def test_hyperbolic_decline_boundary():
    t = np.array([0, 10, 20])
    qi = 5000.0
    di = 0.05
    b = 1.2
    res = hyperbolic_decline(t, qi, di, b)
    assert res[0] == pytest.approx(5000.0)
    assert np.all(res > 0)
    assert np.all(np.diff(res) < 0)  # Monotonically decreasing

def test_modified_arps_transition():
    t = np.arange(0, 100)
    qi = 6000.0
    di = 0.15
    b = 1.1
    d_term = 0.008  # ~10% annual / 12
    res = modified_arps(t, qi, di, b, d_term)
    assert len(res) == 100
    assert res[0] == pytest.approx(qi)
    assert np.all(res > 0)
    assert np.all(np.diff(res) <= 0)

def test_stretched_exponential():
    t = np.array([0, 12, 24, 36])
    qi = 4000.0
    tau = 15.0
    n = 0.6
    res = stretched_exponential(t, qi, tau, n)
    assert res[0] == pytest.approx(qi)
    assert np.all(res > 0)
    assert np.all(np.diff(res) < 0)

def test_arps_modeler_fit_and_predict():
    t = np.arange(0, 30)
    # Synthetic production with known parameters
    q_true = 5000.0 / (1.0 + 1.2 * 0.1 * t) ** (1.0 / 1.2)
    
    model = ArpsModeler(model_type='modified')
    params = model.fit(t, q_true)
    
    assert 'qi' in params
    assert 'b' in params
    assert params['qi'] > 0
    
    pred = model.predict(t)
    assert len(pred) == len(t)
    # Check EUR integration
    eur = model.calculate_eur(60)
    assert eur > 0
