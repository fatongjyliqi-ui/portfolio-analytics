"""
factor_analysis.py — Faktor-Analyse-Modul
==========================================
Werkzeug-Modul für CAPM- und Fama-French-3-Faktor-Analysen.

Funktionen:
    lade_ff_3_faktor_daily()
    capm_regression(asset_renditen, markt_renditen)
    drei_faktor_regression(asset_renditen, ff_daten)
    rolling_beta(asset_renditen, markt_renditen, fenster=63)
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


FF_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)


def lade_ff_3_faktor_daily() -> pd.DataFrame:
    """
    Lädt tägliche Fama-French-3-Faktor-Daten direkt von Ken Frenchs Server.

    Rückgabe
    --------
    pd.DataFrame mit DatetimeIndex und Spalten:
        Mkt-RF, SMB, HML, RF  (alle als Dezimalzahlen, nicht Prozent)
    """
    response = urllib.request.urlopen(FF_URL)
    with zipfile.ZipFile(io.BytesIO(response.read())) as zf:
        with zf.open(zf.namelist()[0]) as csv_file:
            df = pd.read_csv(csv_file, skiprows=4)

    df = df.rename(columns={df.columns[0]: "date"})
    df = df[df["date"].astype(str).str.match(r"^\d{8}$", na=False)]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")
    df = df.astype(float) / 100
    df.columns = [c.strip() for c in df.columns]
    return df


def capm_regression(
    asset_renditen: pd.Series,
    markt_renditen: pd.Series,
) -> dict:
    """
    Führt eine CAPM-Regression durch:
        R_asset = alpha + beta * R_markt + epsilon

    Parameter
    ---------
    asset_renditen  : pd.Series  Tägliche Asset-Renditen.
    markt_renditen  : pd.Series  Tägliche Markt-Renditen (z. B. SPY).

    Rückgabe
    --------
    dict mit: alpha, alpha_annual, beta, r_squared, p_alpha, p_beta
    """
    idx = asset_renditen.index.intersection(markt_renditen.index)
    y = asset_renditen.loc[idx]
    X = sm.add_constant(markt_renditen.loc[idx])
    model = sm.OLS(y, X).fit()

    const_key = model.params.index[0]   # "const"
    mkt_key   = model.params.index[1]

    return {
        "alpha":        model.params[const_key],
        "alpha_annual": model.params[const_key] * 252,
        "beta":         model.params[mkt_key],
        "r_squared":    model.rsquared,
        "p_alpha":      model.pvalues[const_key],
        "p_beta":       model.pvalues[mkt_key],
    }


def drei_faktor_regression(
    asset_renditen: pd.Series,
    ff_daten: pd.DataFrame,
) -> dict:
    """
    Führt eine Fama-French-3-Faktor-Regression durch:
        R_asset - RF = alpha + beta_mkt*(Mkt-RF) + beta_smb*SMB + beta_hml*HML + eps

    Parameter
    ---------
    asset_renditen : pd.Series
        Tägliche Asset-Renditen.
    ff_daten : pd.DataFrame
        Fama-French-Faktoren (von lade_ff_3_faktor_daily()).

    Rückgabe
    --------
    dict mit: alpha_annual, beta_mkt, beta_smb, beta_hml, r_squared, p_alpha
    """
    idx = asset_renditen.index.intersection(ff_daten.index)
    excess = asset_renditen.loc[idx] - ff_daten.loc[idx, "RF"]
    X = sm.add_constant(ff_daten.loc[idx, ["Mkt-RF", "SMB", "HML"]])
    model = sm.OLS(excess, X).fit()

    return {
        "alpha_annual": model.params["const"] * 252 * 100,   # in %
        "beta_mkt":     model.params["Mkt-RF"],
        "beta_smb":     model.params["SMB"],
        "beta_hml":     model.params["HML"],
        "r_squared":    model.rsquared,
        "p_alpha":      model.pvalues["const"],
    }


def rolling_beta(
    asset_renditen: pd.Series,
    markt_renditen: pd.Series,
    fenster: int = 63,
) -> pd.Series:
    """
    Berechnet das Rolling-Beta eines Assets gegen einen Markt-Index.

    Parameter
    ---------
    asset_renditen  : pd.Series  Tägliche Asset-Renditen.
    markt_renditen  : pd.Series  Tägliche Markt-Renditen.
    fenster         : int        Fenstergröße in Handelstagen (Standard: 63 ≈ 3 Monate).

    Rückgabe
    --------
    pd.Series mit Rolling-Beta-Werten (erste `fenster-1` Werte sind NaN).
    """
    idx = asset_renditen.index.intersection(markt_renditen.index)
    asset = asset_renditen.loc[idx]
    markt = markt_renditen.loc[idx]

    rolling_cov = asset.rolling(window=fenster).cov(markt)
    rolling_var = markt.rolling(window=fenster).var(ddof=1)
    return rolling_cov / rolling_var
