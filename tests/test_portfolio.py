"""
Tests fuer analytics.py, backtester.py, optimizer.py und factor_analysis.py.

Alle Tests laufen auf synthetischen Daten - keine Internetverbindung noetig.
"""

import numpy as np
import pandas as pd
import pytest

from analytics import analyse_dataframe, korrelationsmatrix, sharpe_ratio
from backtester import kennzahlen, simuliere_portfolio
from factor_analysis import capm_regression
from optimizer import min_variance_portfolio, tangency_portfolio


def synthetische_renditen(n_tage=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_tage)
    mu = np.array([0.08, 0.05, 0.03]) / 252
    vol = np.array([0.20, 0.12, 0.06]) / np.sqrt(252)
    korr = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    cov = np.outer(vol, vol) * korr
    daten = rng.multivariate_normal(mu, cov, size=n_tage)
    return pd.DataFrame(daten, index=idx, columns=["A", "B", "C"])


# --- analytics -------------------------------------------------------------

def test_sharpe_definition():
    r = synthetische_renditen()["A"]
    erwartet = (r.mean() * 252 - 0.02) / (r.std(ddof=1) * np.sqrt(252))
    assert sharpe_ratio(r, rf=0.02) == pytest.approx(erwartet)


def test_max_drawdown_bekannter_verlauf():
    preise = pd.DataFrame({"X": [100.0, 120.0, 90.0, 110.0, 130.0]},
                          index=pd.bdate_range("2020-01-01", periods=5))
    tab = analyse_dataframe(preise)
    assert tab.loc["X", "max_drawdown"] == pytest.approx(90 / 120 - 1)
    assert tab.loc["X", "gesamtrendite"] == pytest.approx(0.30)


def test_korrelationsmatrix_diagonale():
    preise = (1 + synthetische_renditen()).cumprod() * 100
    k = korrelationsmatrix(preise)
    assert np.allclose(np.diag(k), 1.0)
    assert np.allclose(k, k.T)


# --- backtester ------------------------------------------------------------

def test_buy_and_hold_ohne_rebalancing():
    idx = pd.bdate_range("2020-01-01", periods=3)
    preise = pd.DataFrame({"A": [100.0, 200.0, 200.0], "B": [100.0, 100.0, 50.0]}, index=idx)
    wert = simuliere_portfolio(preise, {"A": 0.5, "B": 0.5}, rebal_freq="N", startkapital=1000)
    # 5 Anteile A, 5 Anteile B -> Tag 3: 5*200 + 5*50
    assert wert.iloc[-1] == pytest.approx(1250.0)


def test_rebalancing_stellt_gewichte_wieder_her():
    idx = pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"])
    preise = pd.DataFrame({"A": [100.0, 200.0, 200.0, 400.0],
                           "B": [100.0, 100.0, 100.0, 100.0]}, index=idx)
    wert = simuliere_portfolio(preise, {"A": 0.5, "B": 0.5}, rebal_freq="Y", startkapital=1000)
    # Jahreswechsel: bei 1500 auf 50/50 zurueck -> 3.75 A und 7.5 B -> 3.75*400 + 7.5*100
    assert wert.iloc[-1] == pytest.approx(2250.0)


def test_cagr_bei_konstantem_wachstum():
    tage = 252 * 3 + 1
    serie = pd.Series(100 * 1.10 ** (np.arange(tage) / 252),
                      index=pd.bdate_range("2018-01-01", periods=tage))
    k = kennzahlen(serie)
    assert k["annual_rendite"] == pytest.approx(0.10, abs=2e-4)
    assert k["max_drawdown"] == pytest.approx(0.0)


# --- optimizer -------------------------------------------------------------

def test_min_variance_gegen_geschlossene_loesung():
    r = synthetische_renditen()
    cov = r.cov().values * 252
    eins = np.ones(len(cov))
    w_exakt = np.linalg.solve(cov, eins)
    w_exakt /= w_exakt.sum()
    assert (w_exakt > 0).all(), "Testfall soll im Inneren (long-only aktiv) liegen"

    w = np.array(list(min_variance_portfolio(r)["gewichte"].values()))
    assert np.allclose(w, w_exakt, atol=1e-3)


def test_tangency_schlaegt_zufallsportfolios():
    r = synthetische_renditen()
    tan = tangency_portfolio(r, rf=0.02)
    w = np.array(list(tan["gewichte"].values()))
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-9).all()

    mu, cov = r.mean().values * 252, r.cov().values * 252
    rng = np.random.default_rng(3)
    for _ in range(2000):
        z = rng.random(3)
        z /= z.sum()
        sharpe_z = (z @ mu - 0.02) / np.sqrt(z @ cov @ z)
        assert sharpe_z <= tan["sharpe"] + 1e-6


# --- factor_analysis -------------------------------------------------------

def test_capm_findet_wahres_beta():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2015-01-01", periods=3000)
    markt = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    asset = 0.0001 + 1.3 * markt + pd.Series(rng.normal(0, 0.005, len(idx)), index=idx)
    erg = capm_regression(asset, markt)
    assert erg["beta"] == pytest.approx(1.3, abs=0.03)
    assert erg["r_squared"] > 0.8
