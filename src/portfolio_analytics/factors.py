"""CAPM and Fama-French 3-factor regressions, rolling beta.

All alphas are returned as decimals (0.01 = 1 % per year).
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from .metrics import TRADING_DAYS


def capm_regression(asset: pd.Series, market: pd.Series) -> dict:
    """OLS fit of  R_asset = alpha + beta * R_market + eps  on daily returns."""
    idx = asset.index.intersection(market.index)
    X = sm.add_constant(market.loc[idx].rename("market"))
    model = sm.OLS(asset.loc[idx], X).fit()
    return {
        "alpha_daily": model.params["const"],
        "alpha_annual": model.params["const"] * TRADING_DAYS,
        "beta": model.params["market"],
        "r_squared": model.rsquared,
        "p_alpha": model.pvalues["const"],
        "p_beta": model.pvalues["market"],
    }


def ff3_regression(asset: pd.Series, factors: pd.DataFrame) -> dict:
    """Fama-French 3-factor regression on excess returns.

    R_asset - RF = alpha + b_mkt (Mkt-RF) + b_smb SMB + b_hml HML + eps
    """
    idx = asset.index.intersection(factors.index)
    excess = asset.loc[idx] - factors.loc[idx, "RF"]
    X = sm.add_constant(factors.loc[idx, ["Mkt-RF", "SMB", "HML"]])
    model = sm.OLS(excess, X).fit()
    return {
        "alpha_annual": model.params["const"] * TRADING_DAYS,
        "beta_mkt": model.params["Mkt-RF"],
        "beta_smb": model.params["SMB"],
        "beta_hml": model.params["HML"],
        "r_squared": model.rsquared,
        "p_alpha": model.pvalues["const"],
    }


def rolling_beta(asset: pd.Series, market: pd.Series, window: int = 63) -> pd.Series:
    """Rolling beta Cov(asset, market) / Var(market); 63 days is about 3 months."""
    idx = asset.index.intersection(market.index)
    a, m = asset.loc[idx], market.loc[idx]
    return a.rolling(window).cov(m) / m.rolling(window).var(ddof=1)
