"""
Tests fuer options.py: Black-Scholes, Greeks, implizite Vola, Monte Carlo.

Die Greeks werden gegen zentrale Differenzenquotienten des Preises geprueft,
die Preise gegen die Put-Call-Paritaet und der MC-Preis gegen die exakte Formel.
"""

import numpy as np
import pytest

from options import (
    black_scholes_call, black_scholes_put,
    delta, gamma, vega, theta, rho,
    implizite_vola, mc_preis,
)

PARAMS = [
    (100.0, 100.0, 1.0, 0.05, 0.20),   # ATM
    (120.0, 100.0, 0.5, 0.03, 0.30),   # ITM-Call
    (80.0, 100.0, 2.0, 0.01, 0.15),    # OTM-Call
]


def preis(S, K, T, r, sigma, typ):
    f = black_scholes_call if typ == "call" else black_scholes_put
    return f(S, K, T, r, sigma)


@pytest.mark.parametrize("S,K,T,r,sigma", PARAMS)
def test_put_call_paritaet(S, K, T, r, sigma):
    c = black_scholes_call(S, K, T, r, sigma)
    p = black_scholes_put(S, K, T, r, sigma)
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("S,K,T,r,sigma", PARAMS)
def test_preisgrenzen(S, K, T, r, sigma):
    c = black_scholes_call(S, K, T, r, sigma)
    assert max(S - K * np.exp(-r * T), 0.0) <= c <= S


@pytest.mark.parametrize("typ", ["call", "put"])
@pytest.mark.parametrize("S,K,T,r,sigma", PARAMS)
def test_greeks_gegen_finite_differenzen(S, K, T, r, sigma, typ):
    h = 1e-4
    d_num = (preis(S + h, K, T, r, sigma, typ) - preis(S - h, K, T, r, sigma, typ)) / (2 * h)
    g_num = (preis(S + h, K, T, r, sigma, typ) - 2 * preis(S, K, T, r, sigma, typ)
             + preis(S - h, K, T, r, sigma, typ)) / h**2
    v_num = (preis(S, K, T, r, sigma + h, typ) - preis(S, K, T, r, sigma - h, typ)) / (2 * h)
    # Theta = dV/dt = -dV/dT (Kalenderzeit vorwaerts, Restlaufzeit rueckwaerts)
    t_num = -(preis(S, K, T + h, r, sigma, typ) - preis(S, K, T - h, r, sigma, typ)) / (2 * h)
    r_num = (preis(S, K, T, r + h, sigma, typ) - preis(S, K, T, r - h, sigma, typ)) / (2 * h)

    assert delta(S, K, T, r, sigma, typ) == pytest.approx(d_num, rel=1e-5)
    assert gamma(S, K, T, r, sigma) == pytest.approx(g_num, rel=1e-3)
    assert vega(S, K, T, r, sigma) == pytest.approx(v_num, rel=1e-5)
    assert theta(S, K, T, r, sigma, typ) == pytest.approx(t_num, rel=1e-5)
    assert rho(S, K, T, r, sigma, typ) == pytest.approx(r_num, rel=1e-5)


@pytest.mark.parametrize("typ", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.8])
def test_implizite_vola_rundreise(typ, sigma):
    S, K, T, r = 100.0, 105.0, 0.75, 0.02
    marktpreis = preis(S, K, T, r, sigma, typ)
    assert implizite_vola(marktpreis, S, K, T, r, typ) == pytest.approx(sigma, abs=1e-8)


def test_implizite_vola_ohne_loesung():
    # Preis unter dem inneren Wert -> keine Volatilitaet erzeugt diesen Preis
    assert np.isnan(implizite_vola(1.0, 120.0, 100.0, 1.0, 0.0, "call"))


@pytest.mark.parametrize("typ", ["call", "put"])
def test_mc_im_konfidenzintervall(typ):
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    exakt = preis(S, K, T, r, sigma, typ)
    mc, se = mc_preis(S, K, T, r, sigma, typ, n_sim=200_000, antithetisch=True, seed=1)
    assert abs(mc - exakt) < 4 * se


def test_antithetisch_reduziert_standardfehler():
    args = (100.0, 100.0, 1.0, 0.05, 0.20, "call", 100_000)
    _, se_naiv = mc_preis(*args, antithetisch=False, seed=7)
    _, se_anti = mc_preis(*args, antithetisch=True, seed=7)
    assert se_anti < se_naiv
