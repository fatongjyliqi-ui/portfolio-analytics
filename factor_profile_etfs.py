"""
factor_profile_etfs.py — Anwendungs-Skript
===========================================
Analysiert Fama-French-3-Faktor-Exposures für ein breites ETF-Universum.
Gibt eine Tabelle aus und speichert die Faktor-Karte (SMB vs. HML).

Ausführen:
    python factor_profile_etfs.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
from factor_analysis import lade_ff_3_faktor_daily, drei_faktor_regression

# --- Konfiguration ---
TICKERS = [
    "VTI",          # US-Aktien gesamt
    "VUG", "QQQ",   # US-Growth
    "VTV", "IWD",   # US-Value
    "IWM", "IJR",   # US-Small Caps
    "SPY", "OEF",   # US-Large Caps
    "VEA", "VWO",   # International
]
START   = "2018-01-01"
END     = "2024-12-31"
AUSGABE = Path(__file__).parent / "plots" / "faktor_karte.png"

# --- Daten laden ---
print("Lade Fama-French-Faktoren ...")
ff = lade_ff_3_faktor_daily()

print("Lade ETF-Preise ...")
preise = yf.download(TICKERS, start=START, end=END,
                     auto_adjust=True, progress=False)["Close"].dropna()
renditen = preise.pct_change().dropna()

# --- Regressionen ---
print("Berechne Faktor-Exposures ...")
ergebnisse = {
    tkr: drei_faktor_regression(renditen[tkr], ff)
    for tkr in TICKERS
}

df = pd.DataFrame(ergebnisse).T[
    ["alpha_annual", "beta_mkt", "beta_smb", "beta_hml", "r_squared", "p_alpha"]
].sort_values("beta_hml", ascending=False)

print("\n=== 3-Faktor-Exposures (sortiert nach HML / Value-Exposure) ===")
print(df.round(4).to_string())

value_etf  = df["beta_hml"].idxmax()
growth_etf = df["beta_hml"].idxmin()
print(f"\n  Stärkste Value-Orientierung:  {value_etf:>4}  "
      f"(beta_hml = {df.loc[value_etf,  'beta_hml']:+.4f})")
print(f"  Stärkste Growth-Orientierung: {growth_etf:>4}  "
      f"(beta_hml = {df.loc[growth_etf, 'beta_hml']:+.4f})")

# --- Faktor-Karte: SMB vs. HML ---
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(df["beta_smb"], df["beta_hml"], s=90, color="steelblue", zorder=3)
for tkr, row in df.iterrows():
    ax.annotate(tkr, xy=(row["beta_smb"], row["beta_hml"]),
                xytext=(6, 4), textcoords="offset points", fontsize=9)

ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Beta SMB  (Small Cap ← → Large Cap)")
ax.set_ylabel("Beta HML  (Growth ← → Value)")
ax.set_title("Faktor-Karte: SMB vs. HML  (Fama-French, 2018–2024)")
ax.grid(True, alpha=0.3)
fig.tight_layout()

AUSGABE.parent.mkdir(exist_ok=True)
fig.savefig(AUSGABE, dpi=150, bbox_inches="tight")
print(f"\nFaktor-Karte gespeichert: {AUSGABE}")
plt.show()
