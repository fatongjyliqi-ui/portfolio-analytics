"""
options.py — Werkzeug-Modul fuer Black-Scholes-Optionspreistheorie.

Oeffentliche API:
    Preise:  black_scholes_call, black_scholes_put
    Greeks:  delta, gamma, vega, theta, rho
    IV:      implizite_vola
    MC:      mc_preis
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Interne Hilfsfunktion
# ---------------------------------------------------------------------------

def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    """d1 und d2 der Black-Scholes-Formel."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


# ---------------------------------------------------------------------------
# Preise
# ---------------------------------------------------------------------------

def black_scholes_call(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Black-Scholes Preis eines europaeischen Calls.

    Args:
        S:     Aktueller Aktienkurs
        K:     Strike-Preis
        T:     Restlaufzeit in Jahren
        r:     Risikofreier Zins (dezimal, annualisiert)
        sigma: Volatilitaet (dezimal, annualisiert)

    Returns:
        Call-Preis
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Black-Scholes Preis eines europaeischen Puts.

    Args:
        S, K, T, r, sigma: wie bei black_scholes_call

    Returns:
        Put-Preis
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def delta(
    S: float, K: float, T: float, r: float, sigma: float,
    optionstyp: str = "call",
) -> float:
    """
    Delta: dPreis/dS.

    Args:
        optionstyp: "call" (0 bis 1) oder "put" (-1 bis 0)

    Returns:
        Delta
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if optionstyp == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def gamma(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Gamma: d²Preis/dS² (gleich fuer Call und Put).

    Misst, wie schnell sich Delta mit S aendert. Hoch bei ATM-Optionen
    kurz vor Verfall — der "Albtraum" von Optionsverkaeufern.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Vega: dPreis/dsigma (gleich fuer Call und Put).

    Gibt Sensitivitaet pro 1.0 sigma-Einheit zurueck (fuer pro 1%-Punkt: / 100).
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def theta(
    S: float, K: float, T: float, r: float, sigma: float,
    optionstyp: str = "call",
) -> float:
    """
    Theta: dPreis/dT (pro Jahr).

    Typisch negativ fuer Long-Positionen (Zeitwertverfall).
    Fuer taeglich: theta(...) / 365.

    Args:
        optionstyp: "call" oder "put"
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    zeitverfall = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if optionstyp == "call":
        return zeitverfall - r * K * np.exp(-r * T) * norm.cdf(d2)
    return zeitverfall + r * K * np.exp(-r * T) * norm.cdf(-d2)


def rho(
    S: float, K: float, T: float, r: float, sigma: float,
    optionstyp: str = "call",
) -> float:
    """
    Rho: dPreis/dr.

    Call-Rho positiv (hoeherer Zins = guenstiger fuer Calls),
    Put-Rho negativ.

    Args:
        optionstyp: "call" oder "put"
    """
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if optionstyp == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


# ---------------------------------------------------------------------------
# Implizite Volatilitaet
# ---------------------------------------------------------------------------

def implizite_vola(
    marktpreis: float,
    S: float,
    K: float,
    T: float,
    r: float,
    optionstyp: str = "call",
) -> float:
    """
    Implizite Volatilitaet via Brent-Root-Finding.

    Loest BS_Preis(sigma) = marktpreis nach sigma auf. Da Black-Scholes
    streng monoton in sigma ist, gibt es genau eine Loesung.

    Args:
        marktpreis: Beobachteter Optionspreis (z.B. Bid-Ask-Mitte)
        optionstyp: "call" oder "put"

    Returns:
        Implizite Vola [0.001, 5.0], oder np.nan falls keine Loesung.
    """
    if optionstyp == "call":
        def ziel(sigma: float) -> float:
            return black_scholes_call(S, K, T, r, sigma) - marktpreis
    else:
        def ziel(sigma: float) -> float:
            return black_scholes_put(S, K, T, r, sigma) - marktpreis

    try:
        return brentq(ziel, 0.001, 5.0)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def mc_preis(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    optionstyp: str = "call",
    n_sim: int = 100_000,
    antithetisch: bool = False,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Monte-Carlo-Preis eines europaeischen Calls oder Puts.

    Simuliert Endkurse unter dem risikoneutralen Mass:
        S_T = S * exp((r - 0.5*sigma²)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)

    Args:
        optionstyp:   "call" oder "put"
        n_sim:        Anzahl Simulationen
        antithetisch: True fuer antithetische Variaten (Varianzreduktion
                      ohne zusaetzliche Simulationen — empfohlen)
        seed:         Zufallsseed fuer Reproduzierbarkeit

    Returns:
        (preis, standardfehler) — Standardfehler skaliert mit 1/sqrt(n_sim)
    """
    rng = np.random.default_rng(seed)

    if antithetisch:
        Z_halb = rng.standard_normal(n_sim // 2)
        Z = np.concatenate([Z_halb, -Z_halb])
    else:
        Z = rng.standard_normal(n_sim)

    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if optionstyp == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    diskontiert = np.exp(-r * T) * payoffs
    preis = float(diskontiert.mean())
    se = float(diskontiert.std(ddof=1) / np.sqrt(len(diskontiert)))
    return preis, se
