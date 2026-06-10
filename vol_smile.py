"""
Volatility-Smile-Plot — importiert aus options.py.

Variante A: echte AAPL-Daten von yfinance (eine Faelligkeit)
Variante B: synthetischer Smile als Fallback

Speichert in plots/vol_smile.png.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import yfinance as yf

from options import implizite_vola

r = 0.04

# ---------------------------------------------------------------------------
# Variante A: yfinance-Marktdaten
# ---------------------------------------------------------------------------

use_marktdaten = False
df_smile = None
title = ""

try:
    print("Lade AAPL-Optionsdaten...")
    ticker = yf.Ticker("AAPL")
    spot = ticker.history(period="1d")["Close"].iloc[-1]

    faell = ticker.options[3]
    T = (pd.Timestamp(faell) - pd.Timestamp.now()).days / 365

    if T > 0:
        calls = ticker.option_chain(faell).calls.copy()

        # Bereinigung: nur echte Bid/Ask, Volumen > 0
        calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)].copy()
        calls = calls[calls["volume"].fillna(0) > 0].copy()

        # Moneyness 75%–125%
        calls = calls[
            (calls["strike"] >= 0.75 * spot) &
            (calls["strike"] <= 1.25 * spot)
        ].copy()

        calls["mid"] = (calls["bid"] + calls["ask"]) / 2
        calls["iv"] = calls.apply(
            lambda row: implizite_vola(row["mid"], spot, row["strike"], T, r, "call"),
            axis=1,
        )
        calls = calls.dropna(subset=["iv"])
        calls = calls[(calls["iv"] > 0.01) & (calls["iv"] < 2.0)].copy()
        calls["moneyness"] = calls["strike"] / spot

        if len(calls) >= 5:
            df_smile = calls[["moneyness", "iv", "impliedVolatility"]].copy()
            use_marktdaten = True
            title = f"AAPL Volatility Smile  (Faelligkeit {faell},  Spot={spot:.2f})"
            print(f"  {len(df_smile)} liquide Strikes fuer {faell}  (T={T:.3f} Jahre)")

except Exception as exc:
    print(f"yfinance-Fehler: {exc}")

# ---------------------------------------------------------------------------
# Variante B: Synthetischer Smile (AAPL-typischer Skew + Smile)
# ---------------------------------------------------------------------------

if not use_marktdaten:
    print("Fallback auf synthetischen Smile.")
    moneyness = np.linspace(0.75, 1.25, 80)
    # Negativer Skew (Put-Praemie) + leichte Smile-Kruemmung
    iv_syn = 0.23 - 0.12 * (moneyness - 1.0) + 0.20 * (moneyness - 1.0)**2
    df_smile = pd.DataFrame({"moneyness": moneyness, "iv": iv_syn})
    title = "Volatility Smile  (synthetisch, AAPL-typischer Skew + Smile)"

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

os.makedirs("plots", exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    df_smile["moneyness"], df_smile["iv"] * 100,
    "o-", color="steelblue", linewidth=2, markersize=5,
    label="Eigene implizite Vola",
)

# yfinance-Referenz-IV einzeichnen (nur Marktdaten-Modus)
if use_marktdaten and "impliedVolatility" in df_smile.columns:
    ax.plot(
        df_smile["moneyness"], df_smile["impliedVolatility"] * 100,
        "s--", color="tomato", linewidth=1.5, markersize=4, alpha=0.75,
        label="yfinance IV (Referenz)",
    )

ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.6)
ax.text(1.01, ax.get_ylim()[0] + 0.5, "ATM", fontsize=9, color="gray")

ax.set_title(title, fontsize=12, fontweight="bold")
ax.set_xlabel("Moneyness  (K / S)", fontsize=11)
ax.set_ylabel("Implizite Volatilitaet (%)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)

plt.tight_layout()
outpath = os.path.join("plots", "vol_smile.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.show()
print(f"Gespeichert: {outpath}")
