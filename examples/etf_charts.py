"""Four-panel overview of six asset-class ETFs: index paths, SPY return
distribution, correlation heatmap and drawdowns."""

import matplotlib.pyplot as plt
from _common import save

from portfolio_analytics.data import load_prices
from portfolio_analytics.metrics import drawdown

TICKERS = ["SPY", "QQQ", "URTH", "EXS1.DE", "TLT", "GLD"]
START, END = "2018-01-01", "2024-12-31"


def main() -> None:
    prices = load_prices(TICKERS, START, END)
    returns = prices.pct_change().dropna()
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    indexed = prices / prices.iloc[0] * 100
    indexed.plot(ax=ax[0, 0], linewidth=1.2)
    ax[0, 0].set(title="Index paths (start = 100)", ylabel="Index", xlabel="")
    ax[0, 0].legend(fontsize=8, loc="upper left")

    spy = returns["SPY"]
    ax[0, 1].hist(spy, bins=80, color="steelblue", edgecolor="black", alpha=0.7)
    ax[0, 1].axvline(spy.mean(), color="red", linestyle="--", label=f"mean {spy.mean():.3%}")
    ax[0, 1].set(title="SPY daily returns", xlabel="Daily return", ylabel="Frequency")
    ax[0, 1].legend()

    corr = returns.corr()
    im = ax[1, 0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax[1, 0].set_xticks(range(len(corr)), corr.columns, rotation=45)
    ax[1, 0].set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax[1, 0].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax[1, 0].set_title("Correlation of daily returns")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)

    (drawdown(prices[["SPY", "TLT", "GLD"]]) * 100).plot(ax=ax[1, 1], linewidth=1.2)
    ax[1, 1].set(title="Drawdown (%)", ylabel="Drawdown (%)", xlabel="")

    for a in (ax[0, 0], ax[1, 1]):
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "etf_charts.png")


if __name__ == "__main__":
    main()
