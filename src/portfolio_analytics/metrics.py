"""Performance and risk metrics.

Conventions used throughout the project:
- returns are simple close-to-close daily returns
- annualisation with 252 trading days (mean * 252, volatility * sqrt(252))
- sample statistics (ddof=1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    rf: float = 0.0,
    periods: int = TRADING_DAYS,
):
    """Annualised Sharpe ratio from daily returns.

    Sharpe = (mean * periods - rf) / (std * sqrt(periods)), with the
    arithmetic mean. This is the same definition the optimiser uses, so
    numbers are comparable across modules.
    """
    mu = returns.mean() * periods
    sigma = returns.std(ddof=1) * np.sqrt(periods)
    return (mu - rf) / sigma


def drawdown(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Drawdown path relative to the running maximum (0 or negative)."""
    return values / values.cummax() - 1.0


def max_drawdown(values: pd.Series | pd.DataFrame):
    """Largest peak-to-trough loss, as a negative number (-0.2 = -20 %)."""
    return drawdown(values).min()


def cagr(values: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Compound annual growth rate of a value series."""
    years = (len(values) - 1) / periods
    return float((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1)


def asset_summary(prices: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    """Return and risk statistics for each column of a price table."""
    returns = prices.pct_change().dropna()
    return pd.DataFrame(
        {
            "ann_return": returns.mean() * TRADING_DAYS,
            "ann_vol": returns.std(ddof=1) * np.sqrt(TRADING_DAYS),
            "sharpe": sharpe_ratio(returns, rf=rf),
            "max_drawdown": max_drawdown(prices),
            "total_return": prices.iloc[-1] / prices.iloc[0] - 1,
            "n_days": len(prices),
        }
    )


def portfolio_summary(values: pd.Series, rf: float = 0.0) -> dict:
    """Statistics for a single portfolio value series.

    The annual return is reported as CAGR (what an investor actually earned
    over several years); the Sharpe ratio uses the arithmetic convention of
    sharpe_ratio() so it matches asset_summary() and the optimiser.
    """
    values = values.dropna()
    returns = values.pct_change().dropna()
    vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return {
        "total_return": values.iloc[-1] / values.iloc[0] - 1,
        "cagr": cagr(values),
        "ann_vol": vol,
        "sharpe": sharpe_ratio(returns, rf=rf) if vol > 0 else np.nan,
        "max_drawdown": max_drawdown(values),
    }


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Correlation of daily returns (not of price levels)."""
    return prices.pct_change().dropna().corr()
