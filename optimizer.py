"""
optimizer.py — Portfolio-Optimierungs-Modul
============================================
Werkzeug-Modul für Markowitz-Portfoliotheorie.

Funktionen:
    min_variance_portfolio(renditen_df)
    tangency_portfolio(renditen_df, rf=0.02)
    efficient_frontier(renditen_df, n_points=50)
    plot_frontier(renditen_df, rf=0.02, ausgabe_pfad=None)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------

def _params(renditen_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    """Berechnet annualisierte Kenngrößen aus einem Renditen-DataFrame."""
    mu = renditen_df.mean().values * 252
    cov = renditen_df.cov().values * 252
    n = len(mu)
    tickers = list(renditen_df.columns)
    return mu, cov, n, tickers


def _rendite(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def _vola(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(w @ cov @ w))


def _constraint_sum(n: int) -> dict:
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1}


# ---------------------------------------------------------------------------
# Öffentliche Funktionen
# ---------------------------------------------------------------------------

def min_variance_portfolio(renditen_df: pd.DataFrame) -> dict:
    """
    Berechnet das Minimum-Varianz-Portfolio.

    Parameter
    ---------
    renditen_df : pd.DataFrame
        Tägliche Renditen (Spalten = Assets).

    Rückgabe
    --------
    dict mit:
        gewichte   : dict {ticker: Gewicht}
        rendite    : annualisierte Rendite
        vola       : annualisierte Volatilität
    """
    mu, cov, n, tickers = _params(renditen_df)
    bounds = [(0, 1)] * n

    res = minimize(
        lambda w: w @ cov @ w,
        x0=np.ones(n) / n,
        constraints=[_constraint_sum(n)],
        bounds=bounds,
    )
    w = res.x
    return {
        "gewichte": dict(zip(tickers, w)),
        "rendite":  _rendite(w, mu),
        "vola":     _vola(w, cov),
    }


def tangency_portfolio(renditen_df: pd.DataFrame, rf: float = 0.02) -> dict:
    """
    Berechnet das Tangency-Portfolio (maximale Sharpe Ratio).

    Parameter
    ---------
    renditen_df : pd.DataFrame
        Tägliche Renditen (Spalten = Assets).
    rf : float
        Risikofreier Zinssatz (annualisiert), Standard 2 %.

    Rückgabe
    --------
    dict mit:
        gewichte   : dict {ticker: Gewicht}
        rendite    : annualisierte Rendite
        vola       : annualisierte Volatilität
        sharpe     : Sharpe Ratio
    """
    mu, cov, n, tickers = _params(renditen_df)
    bounds = [(0, 1)] * n

    def neg_sharpe(w: np.ndarray) -> float:
        r = _rendite(w, mu)
        v = _vola(w, cov)
        return -(r - rf) / v

    res = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        constraints=[_constraint_sum(n)],
        bounds=bounds,
    )
    w = res.x
    r = _rendite(w, mu)
    v = _vola(w, cov)
    return {
        "gewichte": dict(zip(tickers, w)),
        "rendite":  r,
        "vola":     v,
        "sharpe":   (r - rf) / v,
    }


def efficient_frontier(
    renditen_df: pd.DataFrame,
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechnet die Efficient Frontier (ab MVP aufwärts).

    Parameter
    ---------
    renditen_df : pd.DataFrame
        Tägliche Renditen (Spalten = Assets).
    n_points : int
        Anzahl der Frontier-Punkte.

    Rückgabe
    --------
    (volas, renditen, gewichte) als numpy-Arrays.
    """
    mu, cov, n, _ = _params(renditen_df)
    bounds = [(0, 1)] * n

    mvp = min_variance_portfolio(renditen_df)
    ziele = np.linspace(mvp["rendite"], mu.max(), n_points)

    volas, renditen_arr, gewichte = [], [], []
    for ziel in ziele:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, z=ziel: w @ mu - z},
        ]
        res = minimize(
            lambda w: w @ cov @ w,
            x0=np.ones(n) / n,
            constraints=constraints,
            bounds=bounds,
        )
        if res.success:
            volas.append(np.sqrt(res.fun))
            renditen_arr.append(ziel)
            gewichte.append(res.x)

    return np.array(volas), np.array(renditen_arr), np.array(gewichte)


def plot_frontier(
    renditen_df: pd.DataFrame,
    rf: float = 0.02,
    ausgabe_pfad: str | Path | None = None,
) -> None:
    """
    Plottet Efficient Frontier, CML, Tangency- und Min-Variance-Portfolio.

    Parameter
    ---------
    renditen_df : pd.DataFrame
        Tägliche Renditen (Spalten = Assets).
    rf : float
        Risikofreier Zinssatz (annualisiert).
    ausgabe_pfad : str | Path | None
        Wenn angegeben, wird der Plot dort gespeichert (kein hardcoded Pfad).
    """
    mu, cov, n, tickers = _params(renditen_df)

    mvp = min_variance_portfolio(renditen_df)
    tan = tangency_portfolio(renditen_df, rf=rf)
    fr_volas, fr_renditen, _ = efficient_frontier(renditen_df)

    cml_volas = np.linspace(0, tan["vola"] * 1.5, 100)
    cml_renditen = rf + tan["sharpe"] * cml_volas

    # Random-Portfolios als Hintergrund
    np.random.seed(42)
    rv, rr = [], []
    for _ in range(5000):
        w = np.random.random(n)
        w /= w.sum()
        rv.append(_vola(w, cov))
        rr.append(_rendite(w, mu))

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(np.array(rv) * 100, np.array(rr) * 100,
               alpha=0.15, s=8, color="grey", label="Zufällige Portfolios")
    ax.plot(fr_volas * 100, fr_renditen * 100,
            color="navy", linewidth=2.5, label="Efficient Frontier")
    ax.plot(cml_volas * 100, cml_renditen * 100,
            color="red", linewidth=1.5, linestyle="--",
            label=f"CML (r_f={rf*100:.1f}%)")
    ax.scatter(tan["vola"] * 100, tan["rendite"] * 100,
               color="red", s=250, marker="*", zorder=5,
               label=f"Tangency (Sharpe={tan['sharpe']:.2f})")
    ax.scatter(mvp["vola"] * 100, mvp["rendite"] * 100,
               color="green", s=120, marker="D", zorder=5, label="Min-Variance")
    ax.scatter(0, rf * 100, color="black", s=80, zorder=5,
               label=f"r_f = {rf*100:.1f}%")

    for i, ticker in enumerate(tickers):
        ax.scatter(np.sqrt(cov[i, i]) * 100, mu[i] * 100,
                   marker="x", s=80, color="black")
        ax.annotate(ticker, xy=(np.sqrt(cov[i, i]) * 100, mu[i] * 100),
                    xytext=(7, 7), textcoords="offset points")

    ax.set_title("Efficient Frontier mit Capital Market Line")
    ax.set_xlabel("Volatilität (annualisiert, %)")
    ax.set_ylabel("Rendite (annualisiert, %)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if ausgabe_pfad is not None:
        ausgabe_pfad = Path(ausgabe_pfad)
        ausgabe_pfad.parent.mkdir(exist_ok=True)
        fig.savefig(ausgabe_pfad, dpi=150, bbox_inches="tight")

    plt.show()
