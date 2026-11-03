"""Black-Scholes prices and Greeks against parity, bounds and finite
differences; implied vol as a round trip; Monte Carlo against the closed form."""

import numpy as np
import pytest

from portfolio_analytics.options import (
    bs_call,
    bs_price,
    bs_put,
    delta,
    gamma,
    implied_vol,
    mc_price,
    rho,
    theta,
    vega,
)

CASES = [
    (100.0, 100.0, 1.0, 0.05, 0.20),  # at the money
    (120.0, 100.0, 0.5, 0.03, 0.30),  # call in the money
    (80.0, 100.0, 2.0, 0.01, 0.15),   # call out of the money
]


@pytest.mark.parametrize("S,K,T,r,sigma", CASES)
def test_put_call_parity(S, K, T, r, sigma):
    assert bs_call(S, K, T, r, sigma) - bs_put(S, K, T, r, sigma) == pytest.approx(
        S - K * np.exp(-r * T), abs=1e-10
    )


@pytest.mark.parametrize("S,K,T,r,sigma", CASES)
def test_no_arbitrage_bounds(S, K, T, r, sigma):
    c = bs_call(S, K, T, r, sigma)
    assert max(S - K * np.exp(-r * T), 0.0) <= c <= S


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,T,r,sigma", CASES)
def test_greeks_match_finite_differences(S, K, T, r, sigma, kind):
    def V(S=S, T=T, r=r, sigma=sigma):
        return bs_price(S, K, T, r, sigma, kind)

    h = 1e-4
    assert delta(S, K, T, r, sigma, kind) == pytest.approx(
        (V(S=S + h) - V(S=S - h)) / (2 * h), rel=1e-5)
    assert gamma(S, K, T, r, sigma) == pytest.approx(
        (V(S=S + h) - 2 * V() + V(S=S - h)) / h**2, rel=1e-3)
    assert vega(S, K, T, r, sigma) == pytest.approx(
        (V(sigma=sigma + h) - V(sigma=sigma - h)) / (2 * h), rel=1e-5)
    # theta is the derivative in calendar time t, and T = maturity - t
    assert theta(S, K, T, r, sigma, kind) == pytest.approx(
        -(V(T=T + h) - V(T=T - h)) / (2 * h), rel=1e-5)
    assert rho(S, K, T, r, sigma, kind) == pytest.approx(
        (V(r=r + h) - V(r=r - h)) / (2 * h), rel=1e-5)


def test_prices_accept_arrays():
    S = np.linspace(50, 150, 11)
    prices = bs_call(S, 100.0, 1.0, 0.05, 0.2)
    assert prices.shape == S.shape
    assert np.all(np.diff(prices) > 0)  # call price increases in spot


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.8])
def test_implied_vol_round_trip(kind, sigma):
    S, K, T, r = 100.0, 105.0, 0.75, 0.02
    price = bs_price(S, K, T, r, sigma, kind)
    assert implied_vol(price, S, K, T, r, kind) == pytest.approx(sigma, abs=1e-8)


def test_implied_vol_below_intrinsic_value_is_nan():
    assert np.isnan(implied_vol(1.0, 120.0, 100.0, 1.0, 0.0, "call"))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        bs_call(100.0, 100.0, 0.0, 0.05, 0.2)
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1.0, 0.05, 0.2, kind="straddle")


@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_within_confidence_interval(kind):
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    exact = bs_price(S, K, T, r, sigma, kind)
    est, se = mc_price(S, K, T, r, sigma, kind, n_paths=200_000, antithetic=True, seed=1)
    assert abs(est - exact) < 4 * se


def test_antithetic_variates_reduce_standard_error():
    args = (100.0, 100.0, 1.0, 0.05, 0.20, "call", 100_000)
    _, se_plain = mc_price(*args, antithetic=False, seed=7)
    _, se_anti = mc_price(*args, antithetic=True, seed=7)
    assert se_anti < se_plain
