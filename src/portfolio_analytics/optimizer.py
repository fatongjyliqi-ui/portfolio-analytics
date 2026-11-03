"""Mean-variance (Markowitz) optimisation, long-only.

mu and Sigma are estimated from the full sample of daily returns and
annualised with 252 days. The resulting frontier is therefore in-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import TRADING_DAYS


def estimate_moments(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Annualised mean vector and covariance matrix of daily returns."""
    return returns.mean().to_numpy() * TRADING_DAYS, returns.cov().to_numpy() * TRADING_DAYS


def _solve(objective, n: int, extra_constraints=()) -> np.ndarray:
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra_constraints]
    res = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"optimisation failed: {res.message}")
    return res.x


def _result(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, tickers, rf=None) -> dict:
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    out = {"weights": dict(zip(tickers, w, strict=True)), "return": ret, "vol": vol}
    if rf is not None:
        out["sharpe"] = (ret - rf) / vol
    return out


def min_variance_portfolio(returns: pd.DataFrame) -> dict:
    """Long-only minimum-variance portfolio: weights, return, vol."""
    mu, cov = estimate_moments(returns)
    w = _solve(lambda w: w @ cov @ w, len(mu))
    return _result(w, mu, cov, returns.columns)


def tangency_portfolio(returns: pd.DataFrame, rf: float = 0.02) -> dict:
    """Long-only portfolio with the highest Sharpe ratio for risk-free rate rf."""
    mu, cov = estimate_moments(returns)
    w = _solve(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), len(mu))
    return _result(w, mu, cov, returns.columns, rf=rf)


def efficient_frontier(
    returns: pd.DataFrame, n_points: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Efficient frontier from the minimum-variance portfolio upwards.

    For each target return between the MVP return and the highest single
    asset return, minimise variance subject to that return.
    Returns (vols, returns, weights).
    """
    mu, cov = estimate_moments(returns)
    n = len(mu)
    mvp = min_variance_portfolio(returns)

    vols, rets, weights = [], [], []
    for target in np.linspace(mvp["return"], mu.max(), n_points):
        target_constraint = {"type": "eq", "fun": lambda w, t=target: w @ mu - t}
        try:
            w = _solve(lambda w: w @ cov @ w, n, [target_constraint])
        except RuntimeError:
            continue
        vols.append(np.sqrt(w @ cov @ w))
        rets.append(target)
        weights.append(w)
    return np.array(vols), np.array(rets), np.array(weights)


def plot_frontier(returns: pd.DataFrame, rf: float = 0.02, n_random: int = 5000, seed: int = 42):
    """Frontier, capital market line, MVP, tangency portfolio and single assets.

    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    mu, cov = estimate_moments(returns)
    mvp = min_variance_portfolio(returns)
    tan = tangency_portfolio(returns, rf=rf)
    fr_vol, fr_ret, _ = efficient_frontier(returns)

    rng = np.random.default_rng(seed)
    w_rand = rng.dirichlet(np.ones(len(mu)), size=n_random)
    rand_ret = w_rand @ mu
    rand_vol = np.sqrt(np.einsum("ij,jk,ik->i", w_rand, cov, w_rand))

    cml_vol = np.linspace(0, tan["vol"] * 1.5, 100)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(rand_vol * 100, rand_ret * 100, s=8, alpha=0.15, color="grey",
               label="Random portfolios")
    ax.plot(fr_vol * 100, fr_ret * 100, color="navy", linewidth=2.5,
            label="Efficient frontier")
    ax.plot(cml_vol * 100, (rf + tan["sharpe"] * cml_vol) * 100, "r--", linewidth=1.5,
            label=f"Capital market line (rf = {rf:.1%})")
    ax.scatter(tan["vol"] * 100, tan["return"] * 100, color="red", s=250, marker="*",
               zorder=5, label=f"Tangency (Sharpe {tan['sharpe']:.2f})")
    ax.scatter(mvp["vol"] * 100, mvp["return"] * 100, color="green", s=120, marker="D",
               zorder=5, label="Minimum variance")

    for i, ticker in enumerate(returns.columns):
        x, y = np.sqrt(cov[i, i]) * 100, mu[i] * 100
        ax.scatter(x, y, marker="x", s=80, color="black")
        ax.annotate(ticker, (x, y), xytext=(7, 7), textcoords="offset points")

    ax.set(title="Efficient frontier and capital market line",
           xlabel="Volatility (annualised, %)", ylabel="Return (annualised, %)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
