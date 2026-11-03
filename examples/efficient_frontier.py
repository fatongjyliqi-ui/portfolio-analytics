"""Minimum-variance and tangency portfolios for five ETFs, plus the frontier plot."""

import pandas as pd
from _common import save

from portfolio_analytics.data import load_prices
from portfolio_analytics.optimizer import min_variance_portfolio, plot_frontier, tangency_portfolio

TICKERS = ["SPY", "QQQ", "URTH", "TLT", "GLD"]
START, END = "2015-01-01", "2024-12-31"
RF = 0.04


def main() -> None:
    returns = load_prices(TICKERS, START, END).pct_change().dropna()
    mvp = min_variance_portfolio(returns)
    tan = tangency_portfolio(returns, rf=RF)

    weights = pd.DataFrame({"Min variance": mvp["weights"], "Tangency": tan["weights"]}).T
    print("Weights (%)")
    print((weights * 100).round(1).to_string(), "\n")
    print(f"Min variance: return {mvp['return']:.2%}, vol {mvp['vol']:.2%}")
    print(f"Tangency (rf {RF:.1%}): return {tan['return']:.2%}, vol {tan['vol']:.2%}, "
          f"Sharpe {tan['sharpe']:.2f}")

    save(plot_frontier(returns, rf=RF), "efficient_frontier.png")


if __name__ == "__main__":
    main()
