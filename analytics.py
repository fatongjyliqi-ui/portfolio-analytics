"""
Analytics - Werkzeug-Modul fuer Asset-Renditen-Analyse.

Funktionen zur Berechnung von Performance- und Risiko-Kennzahlen fuer
einzelne Assets oder ein Universum von Assets aus Preisdaten.

Teil des portfolio-analytics Projekts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


HANDELSTAGE_PRO_JAHR = 252


def sharpe_ratio(
    renditen: pd.Series | pd.DataFrame,
    rf: float = 0.0,
    tage_pro_jahr: int = HANDELSTAGE_PRO_JAHR,
):
    """
    Sharpe Ratio aus taeglichen Renditen - EINE Definition fuer das ganze Projekt.

        Sharpe = (mu_annual - rf) / sigma_annual

    mit mu_annual = arithmetisches Mittel der Tagesrenditen * 252 und
    sigma_annual = Stichproben-Std der Tagesrenditen * sqrt(252).
    Das ist dieselbe Konvention wie in optimizer.py (Markowitz-Parameter).

    Parameter:
        renditen: Tagesrenditen (Series fuer ein Asset, DataFrame fuer mehrere).
        rf: risikofreier Zins, annualisiert (Default 0 fuer reine Vergleiche).
        tage_pro_jahr: Annualisierungsfaktor.
    """
    mu = renditen.mean() * tage_pro_jahr
    sigma = renditen.std(ddof=1) * np.sqrt(tage_pro_jahr)
    return (mu - rf) / sigma


def analyse_dataframe(
    preise_df: pd.DataFrame,
    tage_pro_jahr: int = HANDELSTAGE_PRO_JAHR,
    rf: float = 0.0,
) -> pd.DataFrame:
    """
    Berechnet Performance- und Risiko-Kennzahlen pro Asset.

    Parameter:
        preise_df: DataFrame mit Datum-Index und einer Preisspalte pro Asset.
        tage_pro_jahr: Annualisierungsfaktor (Default 252 Handelstage).
        rf: risikofreier Zins fuer die Sharpe Ratio (annualisiert, Default 0).

    Returns:
        DataFrame mit Assets als Zeilen, Kennzahlen als Spalten:
        - annualisierte_rendite (Dezimal, 0.10 = 10%)
        - annualisierte_volatilitaet
        - sharpe_ratio
        - max_drawdown (negative Zahl, -0.15 = -15%)
        - anzahl_handelstage
        - gesamtrendite (Dezimal, kumulative Gesamtperformance)

    Hinweis: Volatilitaet wird auf RENDITEN berechnet, nicht auf Preisen.
    """
    renditen = preise_df.pct_change().dropna()

    # Mittlere taegliche Rendite und ihre Annualisierung
    mittlere_taegliche_rendite = renditen.mean()
    annualisierte_rendite = mittlere_taegliche_rendite * tage_pro_jahr

    # Volatilitaet auf den RENDITEN, nicht auf den Preisen
    taegliche_vol = renditen.std(ddof=1)
    annualisierte_volatilitaet = taegliche_vol * np.sqrt(tage_pro_jahr)

    # Sharpe Ratio (projektweite Definition, siehe sharpe_ratio())
    sharpe = sharpe_ratio(renditen, rf=rf, tage_pro_jahr=tage_pro_jahr)

    # Max Drawdown
    kum = (1 + renditen).cumprod()
    peak = kum.cummax()
    dd = (kum - peak) / peak
    max_drawdown = dd.min()

    # Gesamtrendite ueber den Zeitraum
    gesamtrendite = preise_df.iloc[-1] / preise_df.iloc[0] - 1

    return pd.DataFrame({
        "annualisierte_rendite":    annualisierte_rendite,
        "annualisierte_vol":        annualisierte_volatilitaet,
        "sharpe_ratio":             sharpe,
        "max_drawdown":             max_drawdown,
        "anzahl_handelstage":       len(preise_df),
        "gesamtrendite":            gesamtrendite,
    })


def korrelationsmatrix(preise_df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet die Korrelationsmatrix der Tagesrenditen.

    Parameter:
        preise_df: DataFrame mit Preisen.

    Returns:
        DataFrame mit Korrelationen zwischen den Assets.
    """
    renditen = preise_df.pct_change().dropna()
    return renditen.corr()
