"""
ETF-Kennzahlen-Analyse: Performance und Risiko von 6 Welt-Assets seit 2015.

Analysiert die wichtigsten Anlageklassen-ETFs ueber 10 Jahre:
- US-Aktien (SPY)
- US-Technologie (QQQ)
- Welt-Aktien (URTH)
- Deutscher Aktienmarkt (EXS1.DE)
- US-Langlaufanleihen (TLT)
- Gold (GLD)

Output: Konsolen-Tabelle mit Kennzahlen plus Korrelationsmatrix.

Aufruf:
  python etf_kennzahlen.py
"""

import yfinance as yf
from analytics import analyse_dataframe, korrelationsmatrix


TICKER_NAMEN = {
    "SPY":     "US-Aktien (SPY)",
    "QQQ":     "US-Tech (QQQ)",
    "URTH":    "Welt-Aktien (URTH)",
    "EXS1.DE": "DAX (EXS1.DE)",
    "TLT":     "US-Langanleihen (TLT)",
    "GLD":     "Gold (GLD)",
}

START = "2015-01-01"
END   = "2024-12-31"


def formatiere_kennzahlen(tabelle):
    """Wandelt Roh-Kennzahlen in eine lesbare Anzeige um."""
    anzeige = tabelle.copy()
    for col in ["annualisierte_rendite", "annualisierte_vol",
                 "max_drawdown", "gesamtrendite"]:
        anzeige[col] = (anzeige[col] * 100).map("{:+.2f}%".format)
    anzeige["sharpe_ratio"] = tabelle["sharpe_ratio"].map("{:.2f}".format)
    anzeige.columns = ["Ann. Rendite", "Ann. Vol", "Sharpe", "Max DD",
                        "Handelstage", "Gesamt-Rendite"]
    return anzeige


def main():
    tickers = list(TICKER_NAMEN.keys())

    print(f"Lade Daten fuer {len(tickers)} Assets ({START} bis {END}) ...")
    preise = yf.download(tickers, start=START, end=END,
                          auto_adjust=True, progress=False)["Close"].dropna()
    print(f"  {len(preise)} gemeinsame Handelstage\n")

    # Spaltennamen freundlicher machen
    preise = preise.rename(columns=TICKER_NAMEN)

    # Kennzahlen
    print("--- Performance- und Risiko-Kennzahlen ---")
    kennzahlen_tabelle = analyse_dataframe(preise)
    print(formatiere_kennzahlen(kennzahlen_tabelle).to_string())
    print()

    # Korrelationsmatrix
    print("--- Korrelationsmatrix der Tagesrenditen ---")
    korr = korrelationsmatrix(preise)
    print(korr.round(2).to_string())


if __name__ == "__main__":
    main()
