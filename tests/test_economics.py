import numpy as np
import pytest
from src.models.economics import EconomicEvaluator

def test_economic_cash_flow():
    evaluator = EconomicEvaluator(oil_price_usd_bbl=70.0, opex_usd_bbl=6.0, royalty_rate=0.15)
    # 100 m3 de producción mensual
    q_m3 = np.array([100.0])
    cf = evaluator.calculate_cash_flow(q_m3)
    
    # 100 m3 * 6.2898 = 628.98 bbl
    bbl = 100.0 * 6.2898
    gross = bbl * 70.0
    net_rev = gross * 0.85
    opex = bbl * 6.0
    expected_cf = net_rev - opex
    
    assert cf[0] == pytest.approx(expected_cf, rel=1e-3)

def test_npv_discounting():
    evaluator = EconomicEvaluator(oil_price_usd_bbl=70.0, opex_usd_bbl=6.0, discount_rate_annual=0.10)
    q_m3 = np.array([100.0, 100.0, 100.0])
    npv = evaluator.calculate_npv(q_m3)
    
    # NPV should be less than simple undiscounted sum of cash flows for t > 0
    cf_undiscounted = np.sum(evaluator.calculate_cash_flow(q_m3))
    assert npv < cf_undiscounted
    assert npv > 0

def test_payout_time():
    evaluator = EconomicEvaluator(oil_price_usd_bbl=70.0, opex_usd_bbl=6.0)
    # Constant production
    q_m3 = np.full(50, 500.0)
    capex = 500_000.0  # 500k USD
    payout_month = evaluator.calculate_payout_time(q_m3, capex)
    
    assert payout_month > 0
    assert payout_month < 50
