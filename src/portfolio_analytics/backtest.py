"""Portfolio backtests with periodic rebalancing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import drawdown, portfolio_summary

_PERIOD_KEYS = {
    "Y": lambda d: d.year,
    "Q": lambda d: (d.year, d.quarter),
    "M": lambda d: (d.year, d.month),
}


def simulate_portfolio(
    prices: pd.DataFrame,
    weights: dict[str, float],
    rebalance: str = "Y",
    initial_capital: float = 10_000.0,
) -> pd.Series:
    """Daily value of a portfolio that is rebalanced to target weights.

    Args:
        prices: price table, one column per ticker.
        weights: target weights {ticker: weight}; normalised to sum to 1.
        rebalance: "Y" yearly, "Q" quarterly, "M" monthly, "N" buy-and-hold.
            Rebalancing happens at the close of the first trading day of a
            new period.
        initial_capital: starting value.

    No transaction costs or taxes are modelled.
    """
    if rebalance not in (*_PERIOD_KEYS, "N"):
        raise ValueError(f"unknown rebalance frequency: {rebalance!r}")

    tickers = list(weights)
    w = np.array([weights[t] for t in tickers], dtype=float)
    w /= w.sum()

    px = prices[tickers].dropna()
    p = px.to_numpy()
    period_of = _PERIOD_KEYS.get(rebalance)
    periods = [period_of(d) for d in px.index] if period_of else None

    values = np.empty(len(px))
    units = initial_capital * w / p[0]
    values[0] = initial_capital
    for i in range(1, len(px)):
        values[i] = units @ p[i]
        if periods is not None and periods[i] != periods[i - 1]:
            units = values[i] * w / p[i]

    return pd.Series(values, index=px.index, name="portfolio")


def compare_strategies(
    prices: pd.DataFrame,
    strategies: dict[str, dict[str, float]],
    rebalance: str = "Y",
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Run several weightings on the same data.

    Returns a summary table (one row per strategy) and the value series.
    """
    series = {
        name: simulate_portfolio(prices, w, rebalance)
        for name, w in strategies.items()
    }
    table = pd.DataFrame({name: portfolio_summary(s) for name, s in series.items()}).T
    return table, series


def plot_comparison(series: dict[str, pd.Series]):
    """Cumulative performance and drawdown of several strategies.

    Returns the matplotlib figure; saving or showing is left to the caller.
    """
    import matplotlib.pyplot as plt

    fig, (ax_perf, ax_dd) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Strategy comparison: performance and drawdown",
                 fontsize=14, fontweight="bold")

    for name, s in series.items():
        normed = s / s.iloc[0]
        (line,) = ax_perf.plot(normed.index, normed.values, label=name, linewidth=1.6)
        dd = drawdown(normed) * 100
        ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.3, color=line.get_color())
        ax_dd.plot(dd.index, dd.values, color=line.get_color(), linewidth=0.9, label=name)

    ax_perf.set(ylabel="Value (start = 1)", title="Cumulative performance")
    ax_dd.set(ylabel="Drawdown (%)", title="Drawdown")
    ax_perf.legend(loc="upper left")
    ax_dd.legend(loc="lower left")
    for ax in (ax_perf, ax_dd):
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig
