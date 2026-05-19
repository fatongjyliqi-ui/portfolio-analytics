"""
Strategievergleich: 4 Asset-Allokationen ueber 2015-2024.

Vergleicht konservative, ausgewogene, aggressive und defensive Portfolios
auf SPY (US-Aktien), TLT (US-Treasuries), GLD (Gold) und EXS1.DE (DAX-ETF).

Alle Strategien werden jaehrlich rebalanciert, gestartet mit 10.000 Euro.

Output:
  - Tabelle der Performance-Kennzahlen auf der Konsole
  - Plot mit Performance- und Drawdown-Verlauf in plots/strategievergleich.png

Aufruf:
  python strategievergleich.py
"""

import os
from backtester import lade_daten, vergleiche_strategien, plotte_vergleich


TICKERS = ["SPY", "TLT", "GLD", "EXS1.DE"]
START   = "2015-01-01"
END     = "2024-12-31"

STRATEGIEN = {
    "Konservativ": {"SPY": 0.10, "TLT": 0.60, "GLD": 0.20, "EXS1.DE": 0.10},
    "Ausgewogen":  {"SPY": 0.40, "TLT": 0.30, "GLD": 0.15, "EXS1.DE": 0.15},
    "Aggressiv":   {"SPY": 0.70, "TLT": 0.10, "GLD": 0.10, "EXS1.DE": 0.10},
    "Defensiv":    {"SPY": 0.20, "TLT": 0.40, "GLD": 0.30, "EXS1.DE": 0.10},
}


def formatiere_tabelle(tabelle):
    """Wandelt die rohe Kennzahlen-Tabelle in eine lesbare Anzeige um."""
    anzeige = tabelle.copy()
    for col in ["gesamt_rendite", "annual_rendite", "annual_vol", "max_drawdown"]:
        anzeige[col] = (anzeige[col] * 100).map("{:+.2f}%".format)
    anzeige["sharpe"] = tabelle["sharpe"].map("{:.2f}".format)
    anzeige.columns = ["Ges.-Rendite", "Ann. Rendite", "Ann. Vol", "Sharpe", "Max DD"]
    return anzeige


def main():
    print("Lade Marktdaten ...")
    preise = lade_daten(TICKERS, START, END)
    print(f"  {len(preise)} Handelstage | {list(preise.columns)}\n")

    print("Simuliere Strategien ...")
    tabelle, serien = vergleiche_strategien(preise, STRATEGIEN, rebal_freq="Y")

    print("\n--- Strategievergleich (2015-2024, jaehrliches Rebalancing) ---")
    print(formatiere_tabelle(tabelle).to_string())
    print()

    # Plot relativ zum Skript-Verzeichnis speichern (funktioniert unabhaengig
    # davon, von wo das Skript aufgerufen wird)
    skript_verzeichnis = os.path.dirname(os.path.abspath(__file__))
    plot_pfad = os.path.join(skript_verzeichnis, "plots", "strategievergleich.png")
    plotte_vergleich(serien, ausgabe_pfad=plot_pfad)


if __name__ == "__main__":
    main()
