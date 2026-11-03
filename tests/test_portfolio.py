"""Metrics, backtester, optimiser and factor regressions on synthetic data
(no internet connection needed)."""

import numpy as np
import pandas as pd
import pytest

from portfolio_analytics.backtest import simulate_portfolio
from portfolio_analytics.factors import capm_regression, ff3_regression, rolling_beta
from portfolio_analytics.metrics import (
    asset_summary,
    correlation_matrix,
    max_drawdown,
    portfolio_summary,
    sharpe_ratio,
)
from portfolio_analytics.optimizer import (
    efficient_frontier,
    min_variance_portfolio,
    tangency_portfolio,
)


def synthetic_returns(n_days=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    mu = np.array([0.08, 0.05, 0.03]) / 252
    vol = np.array([0.20, 0.12, 0.06]) / np.sqrt(252)
    corr = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    data = rng.multivariate_normal(mu, np.outer(vol, vol) * corr, size=n_days)
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C"])


# --- metrics ---------------------------------------------------------------

def test_sharpe_definition():
    r = synthetic_returns()["A"]
    expected = (r.mean() * 252 - 0.02) / (r.std(ddof=1) * np.sqrt(252))
    assert sharpe_ratio(r, rf=0.02) == pytest.approx(expected)


def test_max_drawdown_known_path():
    prices = pd.DataFrame({"X": [100.0, 120.0, 90.0, 110.0, 130.0]},
                          index=pd.bdate_range("2020-01-01", periods=5))
    assert max_drawdown(prices["X"]) == pytest.approx(90 / 120 - 1)
    summary = asset_summary(prices)
    assert summary.loc["X", "max_drawdown"] == pytest.approx(90 / 120 - 1)
    assert summary.loc["X", "total_return"] == pytest.approx(0.30)


def test_max_drawdown_counts_loss_on_first_day():
    # a drop right after the start must count (peak = starting value)
    s = pd.Series([100.0, 80.0, 90.0])
    assert max_drawdown(s) == pytest.approx(-0.2)


def test_correlation_matrix_is_symmetric_with_unit_diagonal():
    prices = (1 + synthetic_returns()).cumprod() * 100
    c = correlation_matrix(prices)
    assert np.allclose(np.diag(c), 1.0)
    assert np.allclose(c, c.T)


def test_cagr_of_constant_growth():
    days = 252 * 3 + 1
    s = pd.Series(100 * 1.10 ** (np.arange(days) / 252),
                  index=pd.bdate_range("2018-01-01", periods=days))
    summary = portfolio_summary(s)
    assert summary["cagr"] == pytest.approx(0.10, abs=1e-12)
    assert summary["max_drawdown"] == pytest.approx(0.0)


# --- backtest --------------------------------------------------------------

def test_buy_and_hold():
    idx = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 200.0, 200.0], "B": [100.0, 100.0, 50.0]}, index=idx)
    value = simulate_portfolio(prices, {"A": 0.5, "B": 0.5}, rebalance="N", initial_capital=1000)
    # 5 units of A and 5 of B, never rebalanced -> 5*200 + 5*50
    assert value.iloc[-1] == pytest.approx(1250.0)


def test_yearly_rebalancing_restores_weights():
    idx = pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"])
    prices = pd.DataFrame({"A": [100.0, 200.0, 200.0, 400.0],
                           "B": [100.0, 100.0, 100.0, 100.0]}, index=idx)
    value = simulate_portfolio(prices, {"A": 0.5, "B": 0.5}, rebalance="Y", initial_capital=1000)
    # first trading day of 2021: value 1500 -> back to 50/50 = 3.75 A + 7.5 B
    assert value.iloc[-1] == pytest.approx(3.75 * 400 + 7.5 * 100)


def test_unknown_rebalance_frequency_raises():
    prices = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.bdate_range("2020-01-01", periods=2))
    with pytest.raises(ValueError):
        simulate_portfolio(prices, {"A": 1.0}, rebalance="W")


# --- optimizer -------------------------------------------------------------

def test_min_variance_matches_closed_form():
    r = synthetic_returns()
    cov = r.cov().to_numpy() * 252
    w_exact = np.linalg.solve(cov, np.ones(len(cov)))
    w_exact /= w_exact.sum()
    assert (w_exact > 0).all(), "test case must be interior so long-only is not binding"

    w = np.array(list(min_variance_portfolio(r)["weights"].values()))
    assert np.allclose(w, w_exact, atol=1e-4)


def returns_with_exact_moments(mu_annual, cov_annual, n_days=1000, seed=0):
    """Daily returns whose sample mean and covariance equal the targets exactly."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_days, len(mu_annual)))
    z -= z.mean(axis=0)
    z = z @ np.linalg.inv(np.linalg.cholesky(np.cov(z, rowvar=False))).T  # sample cov = I
    data = mu_annual / 252 + z @ np.linalg.cholesky(cov_annual / 252).T
    return pd.DataFrame(data, columns=["A", "B", "C"])


def test_tangency_matches_closed_form():
    # without binding constraints, tangency weights are proportional to Sigma^-1 (mu - rf)
    mu = np.array([0.08, 0.06, 0.04])
    vol = np.array([0.20, 0.12, 0.06])
    corr = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    cov = np.outer(vol, vol) * corr
    r = returns_with_exact_moments(mu, cov)

    w_exact = np.linalg.solve(cov, mu - 0.02)
    w_exact /= w_exact.sum()
    assert (w_exact > 0).all(), "test case must be interior so long-only is not binding"
    w = np.array(list(tangency_portfolio(r, rf=0.02)["weights"].values()))
    assert np.allclose(w, w_exact, atol=1e-3)


def test_tangency_beats_random_portfolios():
    r = synthetic_returns()
    tan = tangency_portfolio(r, rf=0.02)
    w = np.array(list(tan["weights"].values()))
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-9).all()

    mu, cov = r.mean().to_numpy() * 252, r.cov().to_numpy() * 252
    z = np.random.default_rng(3).dirichlet(np.ones(3), size=2000)
    sharpe_z = (z @ mu - 0.02) / np.sqrt(np.einsum("ij,jk,ik->i", z, cov, z))
    assert (sharpe_z <= tan["sharpe"] + 1e-6).all()


def test_frontier_is_increasing_and_starts_at_mvp():
    r = synthetic_returns()
    vols, rets, weights = efficient_frontier(r, n_points=20)
    assert np.all(np.diff(rets) > 0)
    assert np.all(np.diff(vols) > -1e-9)
    assert vols[0] == pytest.approx(min_variance_portfolio(r)["vol"], rel=1e-4)
    assert np.allclose(weights.sum(axis=1), 1.0)


# --- factors ---------------------------------------------------------------

def test_capm_recovers_true_beta():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2015-01-01", periods=3000)
    market = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    asset = 0.0001 + 1.3 * market + pd.Series(rng.normal(0, 0.005, len(idx)), index=idx)
    fit = capm_regression(asset, market)
    assert fit["beta"] == pytest.approx(1.3, abs=0.03)
    assert fit["r_squared"] > 0.8


def test_ff3_recovers_true_loadings():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=3000)
    f = pd.DataFrame(rng.normal(0, 0.01, (len(idx), 3)), index=idx,
                     columns=["Mkt-RF", "SMB", "HML"])
    f["RF"] = 0.0001
    asset = f["RF"] + f @ pd.Series({"Mkt-RF": 1.0, "SMB": 0.4, "HML": -0.3, "RF": 0.0}) \
        + rng.normal(0, 0.002, len(idx))
    fit = ff3_regression(asset, f)
    assert fit["beta_mkt"] == pytest.approx(1.0, abs=0.02)
    assert fit["beta_smb"] == pytest.approx(0.4, abs=0.02)
    assert fit["beta_hml"] == pytest.approx(-0.3, abs=0.02)


def test_rolling_beta_constant_relation():
    idx = pd.bdate_range("2020-01-01", periods=300)
    market = pd.Series(np.random.default_rng(2).normal(0, 0.01, len(idx)), index=idx)
    beta = rolling_beta(2.0 * market, market, window=63)
    assert beta.iloc[:62].isna().all()
    assert np.allclose(beta.dropna(), 2.0)
