"""European options under Black-Scholes: prices, Greeks, implied volatility,
Monte Carlo.

Parameters follow the usual notation: spot S, strike K, time to maturity T
in years, continuously compounded risk-free rate r, volatility sigma (both
annualised decimals). Prices and Greeks accept numpy arrays for S. No
dividends are modelled.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

CALL, PUT = "call", "put"


def _check_kind(kind: str) -> None:
    if kind not in (CALL, PUT):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _d1_d2(S, K, T, r, sigma):
    if np.any(np.asarray(T) <= 0) or np.any(np.asarray(sigma) <= 0):
        raise ValueError("T and sigma must be positive")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


# --- prices ----------------------------------------------------------------

def bs_price(S, K, T, r, sigma, kind: str = CALL):
    """Black-Scholes price of a European call or put."""
    _check_kind(kind)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc_K = K * np.exp(-r * T)
    if kind == CALL:
        return S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    return disc_K * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call(S, K, T, r, sigma):
    return bs_price(S, K, T, r, sigma, CALL)


def bs_put(S, K, T, r, sigma):
    return bs_price(S, K, T, r, sigma, PUT)


# --- Greeks ----------------------------------------------------------------

def delta(S, K, T, r, sigma, kind: str = CALL):
    """dV/dS. Call delta lies in (0, 1), put delta in (-1, 0)."""
    _check_kind(kind)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) if kind == CALL else norm.cdf(d1) - 1.0


def gamma(S, K, T, r, sigma):
    """d2V/dS2, identical for calls and puts. Largest at the money near expiry."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """dV/dsigma per unit of volatility (divide by 100 for one vol point)."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, kind: str = CALL):
    """dV/dt per year in calendar time, i.e. -dV/dT. Divide by 365 for daily."""
    _check_kind(kind)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    decay = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    carry = r * K * np.exp(-r * T)
    if kind == CALL:
        return decay - carry * norm.cdf(d2)
    return decay + carry * norm.cdf(-d2)


def rho(S, K, T, r, sigma, kind: str = CALL):
    """dV/dr: positive for calls, negative for puts."""
    _check_kind(kind)
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if kind == CALL:
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


# --- implied volatility ----------------------------------------------------

def implied_vol(price: float, S: float, K: float, T: float, r: float,
                kind: str = CALL, lo: float = 1e-4, hi: float = 5.0) -> float:
    """Volatility that reproduces the given option price, via Brent's method.

    The Black-Scholes price is strictly increasing in sigma, so the root is
    unique if it exists. Returns nan if the price lies outside the no-arbitrage
    bounds (no sigma in [lo, hi] matches it).
    """
    _check_kind(kind)

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, kind) - price

    try:
        return brentq(objective, lo, hi, xtol=1e-12)
    except ValueError:
        return float("nan")


# --- Monte Carlo -----------------------------------------------------------

def mc_price(S: float, K: float, T: float, r: float, sigma: float,
             kind: str = CALL, n_paths: int = 100_000, antithetic: bool = False,
             seed: int | None = None) -> tuple[float, float]:
    """Monte Carlo price of a European option and its standard error.

    Terminal prices are sampled exactly under the risk-neutral measure:
        S_T = S exp((r - sigma^2/2) T + sigma sqrt(T) Z),  Z ~ N(0, 1).

    With antithetic variates, Z and -Z are not independent. The independent
    samples are the pair averages (f(Z) + f(-Z)) / 2, so the standard error
    is computed from those n_paths/2 values.
    """
    _check_kind(kind)
    rng = np.random.default_rng(seed)

    if antithetic:
        z_half = rng.standard_normal(n_paths // 2)
        z = np.concatenate([z_half, -z_half])
    else:
        z = rng.standard_normal(n_paths)

    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(S_T - K, 0.0) if kind == CALL else np.maximum(K - S_T, 0.0)
    discounted = np.exp(-r * T) * payoff

    if antithetic:
        m = len(z_half)
        samples = 0.5 * (discounted[:m] + discounted[m:])
    else:
        samples = discounted

    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(len(samples)))
