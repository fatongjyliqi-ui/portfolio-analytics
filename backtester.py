"""
Backtester - Portfolio-Simulation mit Rebalancing.

Werkzeug-Modul fuer:
- Datenpipeline (yfinance, mit Cleaning)
- Buy-and-Hold und rebalancierte Portfolio-Strategien
- Performance- und Risiko-Kennzahlen
- Vergleichende Visualisierung

Teil des portfolio-analytics Projekts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


HANDELSTAGE_PRO_JAHR = 252


# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------

def lade_daten(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Laedt taeglich-bereinigte Schlusskurse via yfinance.

    Parameter:
        tickers: Liste von Yahoo-Finance-Symbolen, z.B. ["SPY", "TLT"]
        start, end: Datumsstrings im Format "YYYY-MM-DD"

    Returns:
        DataFrame mit Datum-Index und einer Spalte pro Ticker. Tage mit
        fehlenden Werten in einem der Ticker werden entfernt (gemeinsame
        Handelstage).
    """
    rohdaten = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )

    if isinstance(rohdaten.columns, pd.MultiIndex):
        preise = rohdaten["Close"]
    else:
        preise = rohdaten[["Close"]]
        preise.columns = tickers

    return preise.dropna()


# ---------------------------------------------------------------------------
# Portfolio-Simulation
# ---------------------------------------------------------------------------

def simuliere_portfolio(
    preise: pd.DataFrame,
    gewichte_dict: dict,
    rebal_freq: str = "Y",
    startkapital: float = 10_000.0,
) -> pd.Series:
    """
    Simuliert ein Portfolio mit periodischem Rebalancing.

    Parameter:
        preise: DataFrame mit Preisen, Spalten = Asset-Tickers
        gewichte_dict: {ticker: gewicht}, wird intern auf Summe 1 normalisiert
        rebal_freq: 'Y' (jaehrlich), 'Q' (quartalsweise), 'M' (monatlich),
                    'N' (kein Rebalancing - Buy-and-Hold)
        startkapital: Anfangsinvestition (Default 10.000)

    Returns:
        Series mit taeglichem Portfoliowert.
    """
    tickers = list(gewichte_dict.keys())
    weights = np.array(list(gewichte_dict.values()))
    weights = weights / weights.sum()

    preise_sub = preise[tickers].dropna()
    portfolio_wert = pd.Series(index=preise_sub.index, dtype=float)
    portfolio_wert.iloc[0] = startkapital

    anteile = (startkapital * weights) / preise_sub.iloc[0].values

    if rebal_freq == "Y":
        get_period = lambda d: d.year
    elif rebal_freq == "Q":
        get_period = lambda d: (d.year, d.quarter)
    elif rebal_freq == "M":
        get_period = lambda d: (d.year, d.month)
    else:
        get_period = None  # kein Rebalancing

    for i in range(1, len(preise_sub)):
        datum = preise_sub.index[i]
        vortag = preise_sub.index[i - 1]

        aktueller_wert = (anteile * preise_sub.iloc[i].values).sum()
        portfolio_wert.iloc[i] = aktueller_wert

        if get_period is not None and get_period(datum) != get_period(vortag):
            anteile = (aktueller_wert * weights) / preise_sub.iloc[i].values

    return portfolio_wert


# ---------------------------------------------------------------------------
# Kennzahlen
# ---------------------------------------------------------------------------

def kennzahlen(portfolio_serie: pd.Series) -> dict:
    """
    Berechnet Performance- und Risiko-Kennzahlen fuer eine Portfolio-Serie.

    Annualisierung verwendet geometrisches Mittel (Compound Annual Growth Rate),
    da dies bei mehrjaehrigen Zeitraeumen realistischer ist als arithmetisches.

    Returns:
        Dict mit Rohwerten (nicht gerundet) fuer:
        gesamt_rendite, annual_rendite, annual_vol, sharpe, max_drawdown
    """
    s = portfolio_serie.dropna()
    n_jahre = len(s) / HANDELSTAGE_PRO_JAHR

    gesamt = s.iloc[-1] / s.iloc[0] - 1
    annual_r = (1 + gesamt) ** (1 / n_jahre) - 1

    renditen = s.pct_change().dropna()
    annual_v = renditen.std(ddof=1) * np.sqrt(HANDELSTAGE_PRO_JAHR)
    sharpe_ratio = annual_r / annual_v if annual_v > 0 else np.nan

    normiert = s / s.iloc[0]
    max_dd = (normiert / normiert.cummax() - 1).min()

    return {
        "gesamt_rendite": gesamt,
        "annual_rendite": annual_r,
        "annual_vol":     annual_v,
        "sharpe":         sharpe_ratio,
        "max_drawdown":   max_dd,
    }


def vergleiche_strategien(
    preise: pd.DataFrame,
    strategien_dict: dict,
    rebal_freq: str = "Y",
) -> tuple:
    """
    Simuliert mehrere Strategien und gibt Kennzahlen + Portfoliowerte zurueck.

    Parameter:
        preise: DataFrame mit Preisen
        strategien_dict: {strategie_name: {ticker: gewicht}}
        rebal_freq: Rebalancing-Frequenz fuer alle Strategien

    Returns:
        Tuple (kennzahl_tabelle, serien_dict):
        - kennzahl_tabelle: DataFrame mit Strategien als Zeilen, Kennzahlen als Spalten
        - serien_dict: {strategie_name: portfolio_wert_serie} - bereit fuer Plotting
    """
    kennzahl_rows = {}
    serien = {}
    for name, gewichte in strategien_dict.items():
        serie = simuliere_portfolio(preise, gewichte, rebal_freq)
        serien[name] = serie
        kennzahl_rows[name] = kennzahlen(serie)
    return pd.DataFrame(kennzahl_rows).T, serien


# ---------------------------------------------------------------------------
# Visualisierung
# ---------------------------------------------------------------------------

def plotte_vergleich(
    ergebnisse_dict: dict,
    ausgabe_pfad: str = None,
) -> None:
    """
    Plottet kumulierte Performance und Drawdown mehrerer Strategien.

    Parameter:
        ergebnisse_dict: {name: portfolio_wert_serie}
        ausgabe_pfad: Wenn angegeben, wird der Plot als PNG gespeichert
                      (Ordner werden bei Bedarf erstellt). Ansonsten Anzeige.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Strategievergleich: Performance & Drawdown",
                  fontsize=14, fontweight="bold")

    farben = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (name, serie) in enumerate(ergebnisse_dict.items()):
        c = farben[idx % len(farben)]
        normiert = serie / serie.iloc[0]

        axes[0].plot(normiert.index, normiert.values,
                      label=name, color=c, linewidth=1.6)

        dd = (normiert / normiert.cummax() - 1) * 100
        axes[1].fill_between(dd.index, dd.values, 0,
                              alpha=0.35, color=c, label=name)
        axes[1].plot(dd.index, dd.values, color=c, linewidth=0.9, alpha=0.8)

    axes[0].set_ylabel("Normierter Wert (Start = 1)")
    axes[0].set_title("Kumulierte Performance")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_title("Drawdown-Verlauf")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()

    if ausgabe_pfad:
        os.makedirs(os.path.dirname(ausgabe_pfad) or ".", exist_ok=True)
        plt.savefig(ausgabe_pfad, dpi=150, bbox_inches="tight")
        print(f"Plot gespeichert: {ausgabe_pfad}")
        plt.close()
    else:
        plt.show()
