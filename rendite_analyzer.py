"""
Renditen-Analyzer: Berechnung wichtiger Risiko- und Performance-Kennzahlen
für eine Folge von Tages-Renditen.

Phase 1 / Woche 1 des 5-Monats-Projekts Mathematik x Finanzen.

Annahme: Renditen sind tägliche Dezimalwerte (0.01 = 1%).
"""

import math


# ---------------------------------------------------------------------------
# Statistik-Funktionen
# ---------------------------------------------------------------------------

def mittelwert(liste: list) -> float:
    """Gibt den arithmetischen Mittelwert einer Liste zurueck."""
    return sum(liste) / len(liste)


def standardabweichung(liste: list) -> float:
    """Gibt die Stichproben-Standardabweichung zurueck (Division durch n-1)."""
    m = mittelwert(liste)
    abw_q = [(x - m)**2 for x in liste]
    return math.sqrt(sum(abw_q) / (len(liste) - 1))


def annualisieren_rendite(tagesrendite: float, tage: int = 252) -> float:
    """Annualisiert eine taegliche Rendite (lineare Skalierung)."""
    return tagesrendite * tage


def annualisieren_vol(tagesvol: float, tage: int = 252) -> float:
    """Annualisiert eine taegliche Volatilitaet (Wurzel-T-Regel)."""
    return tagesvol * math.sqrt(tage)


def sharpe(rendite: float, vol: float, rf: float = 0) -> float:
    """
    Berechnet die Sharpe Ratio.

    Parameter:
        rendite: annualisierte Rendite
        vol: annualisierte Volatilitaet
        rf: risikofreier Zinssatz (Default 0)
    """
    return (rendite - rf) / vol


def klassifiziere_volatilitaet(annual_vol: float) -> str:
    """Ordnet eine annualisierte Volatilitaet einer Risikoklasse zu."""
    if annual_vol < 0.10:
        return "NIEDRIG (z.B. Anleihen, Geldmarkt)"
    elif annual_vol < 0.20:
        return "MITTEL (z.B. breit gestreute Aktien)"
    elif annual_vol < 0.35:
        return "HOCH (z.B. Einzelaktien, Schwellenlaender)"
    else:
        return "SEHR HOCH (z.B. Krypto, gehebelte Produkte)"


def drawdown(renditen: list) -> float:
    """
    Berechnet den maximalen Drawdown einer Renditenfolge.

    Returns:
        Max Drawdown als Dezimalwert (0.15 = 15% Verlust vom Hoechststand).
    """
    kum_rendite = [1.0]
    for r in renditen:
        kum_rendite.append(kum_rendite[-1] * (1 + r))

    peak = kum_rendite[0]
    max_dd = 0.0
    for value in kum_rendite:
        if value > peak:
            peak = value
        current_dd = (peak - value) / peak if peak != 0 else 0
        if current_dd > max_dd:
            max_dd = current_dd

    return max_dd


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(renditen: list, asset_name: str = "Asset") -> None:
    """
    Gibt einen vollstaendigen Statistik-Report fuer eine Renditenfolge aus.
    """
    n = len(renditen)
    m = mittelwert(renditen)
    s = standardabweichung(renditen)
    ar = annualisieren_rendite(m)
    av = annualisieren_vol(s)
    sr = sharpe(ar, av)
    klasse = klassifiziere_volatilitaet(av)
    positive = len([r for r in renditen if r > 0])
    negative = len([r for r in renditen if r < 0])
    dd = drawdown(renditen)

    print("=" * 50)
    print(f"  RENDITEN-ANALYSE: {asset_name}")
    print("=" * 50)
    print(f"  Anzahl Tage:               {n}")
    print(f"  Beste Tagesrendite:        {max(renditen)*100:>7.2f}%")
    print(f"  Schlechteste Tagesrendite: {min(renditen)*100:>7.2f}%")
    print("-" * 50)
    print(f"  Mittelwert (taeglich):     {m*100:>7.4f}%")
    print(f"  Std-Abweichung (taegl.):   {s*100:>7.4f}%")
    print("-" * 50)
    print(f"  Annualisierte Rendite:     {ar*100:>7.2f}%")
    print(f"  Annualisierte Volatilitaet:{av*100:>7.2f}%")
    print(f"  Sharpe Ratio:              {sr:>7.2f}")
    print("-" * 50)
    print(f"  Volatilitaetsklasse: {klasse}")
    print(f"  Gewinn-Tage: {positive} | Verlust-Tage: {negative}")
    print(f"  Max Drawdown:              {dd*100:>7.2f}%")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Beispiel-Lauf
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    beispiel_renditen = [
        0.012, -0.005, 0.018, 0.003, -0.022,
        0.015, 0.008, -0.011, 0.025, -0.018,
        0.030, -0.005, 0.012, 0.007, -0.014,
        0.022, 0.005, -0.008, 0.018, 0.011,
        -0.003, 0.014, 0.021, -0.009, 0.007,
        0.013, -0.017, 0.024, 0.006, -0.012,
        0.011, 0.018, -0.020, 0.015, 0.009,
        -0.006, 0.022, 0.013, -0.014, 0.008,
        0.017, 0.005, -0.011, 0.019, 0.010,
        -0.007, 0.014, 0.022, -0.013, 0.006,
    ]
    report(beispiel_renditen, asset_name="Beispiel-Asset")
