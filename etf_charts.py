"""
ETF-Visualisierung: 4-Panel-Uebersicht ueber 6 Welt-Assets.

Erzeugt vier Charts in einer PNG-Grafik:
1. Indexverlauf (alle Assets normiert auf Start = 100)
2. Renditen-Verteilung des SPY (Histogramm mit Mittel-Linie)
3. Korrelations-Heatmap aller Assets
4. Drawdown-Verlauf der wichtigsten Assets

Aufruf:
  python etf_charts.py

Output:
  plots/etf_charts.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


TICKERS = ["SPY", "QQQ", "URTH", "EXS1.DE", "TLT", "GLD"]
START   = "2018-01-01"
END     = "2024-12-31"


def lade_daten():
    """Laedt die ETF-Preisdaten."""
    daten = yf.download(
        TICKERS, start=START, end=END, auto_adjust=True, progress=False
    )["Close"].dropna()
    return daten


def plot_indexverlauf(ax, preise):
    """Indexverlauf aller Assets, normiert auf Start = 100."""
    indexed = preise / preise.iloc[0] * 100
    for col in indexed.columns:
        ax.plot(indexed.index, indexed[col], label=col, linewidth=1.2)
    ax.set_title("Indexverlauf (Start = 100)")
    ax.set_ylabel("Index-Wert")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_renditen_verteilung(ax, renditen, asset="SPY"):
    """Histogramm der Tagesrenditen eines Assets."""
    werte = renditen[asset]
    ax.hist(werte, bins=80, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(werte.mean(), color="red", linestyle="--",
                label=f"Mittel: {werte.mean()*100:.3f}%")
    ax.axvline(0, color="black", alpha=0.3)
    ax.set_title(f"{asset} - Renditen-Verteilung")
    ax.set_xlabel("Tagesrendite")
    ax.set_ylabel("Haeufigkeit")
    ax.legend()


def plot_korrelations_heatmap(ax, renditen):
    """Korrelationsmatrix als Heatmap mit Werten."""
    corr = renditen.corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    ax.set_title("Korrelationsmatrix")

    # Werte in die Zellen schreiben
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                     ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_drawdowns(ax, renditen, assets=("SPY", "TLT", "GLD")):
    """Drawdown-Verlauf der angegebenen Assets."""
    kum = (1 + renditen).cumprod()
    peak = kum.cummax()
    dd = (kum - peak) / peak

    for asset in assets:
        ax.plot(dd.index, dd[asset] * 100, label=asset, linewidth=1.2)
    ax.set_title("Drawdown-Verlauf (%)")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    print("Lade Daten ...")
    preise = lade_daten()
    renditen = preise.pct_change().dropna()
    print(f"  {len(preise)} Handelstage | {list(preise.columns)}\n")

    print("Erstelle Charts ...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    plot_indexverlauf(axes[0, 0], preise)
    plot_renditen_verteilung(axes[0, 1], renditen, asset="SPY")
    plot_korrelations_heatmap(axes[1, 0], renditen)
    plot_drawdowns(axes[1, 1], renditen)

    plt.tight_layout()

    # Pfad relativ zum Skript, Ordner anlegen falls noetig
    skript_verzeichnis = os.path.dirname(os.path.abspath(__file__))
    ausgabe_pfad = os.path.join(skript_verzeichnis, "plots", "etf_charts.png")
    os.makedirs(os.path.dirname(ausgabe_pfad), exist_ok=True)

    plt.savefig(ausgabe_pfad, dpi=150, bbox_inches="tight")
    print(f"Plot gespeichert: {ausgabe_pfad}")


if __name__ == "__main__":
    main()
