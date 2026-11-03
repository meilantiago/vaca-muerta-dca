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
    t = np.arange(0, 300)
    qi = 6000.0
    di = 0.15
    b = 1.1
    d_term = 0.008  # ~10% annual / 12
    t_switch = (di / d_term - 1) / (b * di)  # ~107.6 meses
    res = modified_arps(t, qi, di, b, d_term)
    assert len(res) == 300
    assert res[0] == pytest.approx(qi)
    assert np.all(res > 0)
    assert np.all(np.diff(res) <= 0)

    # Continuidad en el punto de cambio y declinación exponencial pura después
    eps = 1e-6
    q_before = modified_arps(np.array([t_switch - eps]), qi, di, b, d_term)[0]
    q_after = modified_arps(np.array([t_switch + eps]), qi, di, b, d_term)[0]
    assert q_after == pytest.approx(q_before, rel=1e-6)
    late = res[t > t_switch + 1]
    assert np.allclose(late[1:] / late[:-1], np.exp(-d_term))


def test_modified_arps_initial_decline_below_terminal():
    t = np.arange(0, 24)
    qi = 1000.0
    res = modified_arps(t, qi, di=0.001, b=1.0, d_term=0.0067)
    assert res[0] == pytest.approx(qi)
    assert np.allclose(res, qi * np.exp(-0.0067 * t))

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
