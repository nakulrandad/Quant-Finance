import numpy as np
import pandas as pd
import pytest

from quant.optimizers import MeanVarianceOptimizer
from quant.portfolio import Portfolio


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = np.random.normal(0.001, 0.02, (100, 3))
    df = pd.DataFrame(data, index=dates, columns=["A", "B", "C"])
    return df


def test_portfolio_init(sample_returns):
    port = Portfolio(returns=sample_returns)
    assert port.assets == ["A", "B", "C"]
    assert port.rebalance_freq == "M"


def test_mvo_weights(sample_returns):
    port = Portfolio(returns=sample_returns)
    weights = port.mvo_weights(rebalance_freq="custom")
    assert isinstance(weights, pd.Series)
    assert len(weights) == 3
    assert np.isclose(weights.sum(), 1.0)
    # Check if weights are within bounds [0, 1]
    assert (weights >= -1e-6).all() and (weights <= 1 + 1e-6).all()


def test_kelly_weights(sample_returns):
    port = Portfolio(returns=sample_returns)
    weights = port.kelly_weights(rebalance_freq="custom")
    assert isinstance(weights, pd.Series)
    assert len(weights) == 3
    assert np.isclose(weights.sum(), 1.0)


def test_risk_budget_weights(sample_returns):
    port = Portfolio(returns=sample_returns)
    weights = port.risk_budget_weights(rebalance_freq="custom")
    assert isinstance(weights, pd.Series)
    assert len(weights) == 3
    assert np.isclose(weights.sum(), 1.0)


def test_optimize_generic(sample_returns):
    optimizer = MeanVarianceOptimizer()
    # Test that optimizer can be used directly with returns
    weights = optimizer.optimize(sample_returns)
    assert isinstance(weights, pd.Series)
    assert np.isclose(weights.sum(), 1.0)
