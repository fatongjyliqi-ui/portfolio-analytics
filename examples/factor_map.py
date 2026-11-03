"""Fama-French 3-factor loadings for eleven ETFs and a SMB/HML factor map."""

import matplotlib.pyplot as plt
import pandas as pd
from _common import save

from portfolio_analytics.data import load_ff3_daily, load_prices
from portfolio_analytics.factors import ff3_regression

TICKERS = [
    "VTI",          # US total market
    "VUG", "QQQ",   # US growth
    "VTV", "IWD",   # US value
    "IWM", "IJR",   # US small caps
    "SPY", "OEF",   # US large caps
    "VEA", "VWO",   # international
]
START, END = "2018-01-01", "2024-12-31"


def main() -> None:
    factors = load_ff3_daily()
    returns = load_prices(TICKERS, START, END).pct_change().dropna()

    table = pd.DataFrame({t: ff3_regression(returns[t], factors) for t in TICKERS}).T
    table = table.sort_values("beta_hml", ascending=False)
    shown = table.copy()
    shown["alpha_annual"] = table["alpha_annual"].map("{:+.2%}".format)
    print("Fama-French 3-factor loadings (sorted by HML)\n")
    print(shown.round(3).to_string())

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(table["beta_smb"], table["beta_hml"], s=90, color="steelblue", zorder=3)
    for t, row in table.iterrows():
        ax.annotate(t, (row["beta_smb"], row["beta_hml"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set(xlabel="SMB loading (large cap  <->  small cap)",
           ylabel="HML loading (growth  <->  value)",
           title=f"Factor map, Fama-French 3-factor model, {START[:4]}-{END[:4]}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "factor_map.png")


if __name__ == "__main__":
    main()
