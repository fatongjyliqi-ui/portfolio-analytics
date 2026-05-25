"""
efficient_frontier_demo.py — Anwendungs-Skript
===============================================
Lädt 5 Assets, optimiert Min-Variance und Tangency,
gibt eine Gewichtstabelle aus und speichert den Frontier-Plot.

Ausführen:
    python efficient_frontier_demo.py
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from optimizer import min_variance_portfolio, tangency_portfolio, efficient_frontier, plot_frontier

# --- Konfiguration ---
TICKERS = ["SPY", "QQQ", "URTH", "TLT", "GLD"]
START   = "2015-01-01"
END     = "2024-12-31"
RF      = 0.04
AUSGABE = Path(__file__).parent / "plots" / "efficient_frontier.png"

# --- Daten laden ---
print("Lade Marktdaten ...")
preise = yf.download(TICKERS, start=START, end=END,
                     auto_adjust=True, progress=False)["Close"].dropna()
renditen = preise.pct_change().dropna()

# --- Optimierungen ---
mvp = min_variance_portfolio(renditen)
tan = tangency_portfolio(renditen, rf=RF)
fr_volas, fr_renditen, _ = efficient_frontier(renditen, n_points=50)

# --- Gewichtstabelle ---
tabelle = pd.DataFrame({
    "Min-Variance": mvp["gewichte"],
    "Tangency":     tan["gewichte"],
}).T * 100

print("\n=== Gewichte (%) ===")
print(tabelle.round(2).to_string())

print(f"\n--- Min-Variance ---")
print(f"  Rendite: {mvp['rendite']*100:.2f}%    Vola: {mvp['vola']*100:.2f}%")

print(f"\n--- Tangency (r_f={RF*100:.1f}%) ---")
print(f"  Rendite: {tan['rendite']*100:.2f}%    Vola: {tan['vola']*100:.2f}%    Sharpe: {tan['sharpe']:.4f}")

print(f"\n--- Efficient Frontier ---")
print(f"  {len(fr_volas)} Punkte  |  "
      f"Vola: {fr_volas.min()*100:.2f}%–{fr_volas.max()*100:.2f}%  |  "
      f"Rendite: {fr_renditen.min()*100:.2f}%–{fr_renditen.max()*100:.2f}%")

# --- Plot ---
print(f"\nSpeichere Plot nach {AUSGABE} ...")
plot_frontier(renditen, rf=RF, ausgabe_pfad=AUSGABE)
