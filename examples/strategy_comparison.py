"""Backtest of four static allocations (yearly rebalancing), 2015-2024."""

from _common import pct, save

from portfolio_analytics.backtest import compare_strategies, plot_comparison
from portfolio_analytics.data import load_prices

TICKERS = ["SPY", "TLT", "GLD", "EXS1.DE"]
START, END = "2015-01-01", "2024-12-31"
STRATEGIES = {
    "Conservative": {"SPY": 0.10, "TLT": 0.60, "GLD": 0.20, "EXS1.DE": 0.10},
    "Balanced":     {"SPY": 0.40, "TLT": 0.30, "GLD": 0.15, "EXS1.DE": 0.15},
    "Aggressive":   {"SPY": 0.70, "TLT": 0.10, "GLD": 0.10, "EXS1.DE": 0.10},
    "Defensive":    {"SPY": 0.20, "TLT": 0.40, "GLD": 0.30, "EXS1.DE": 0.10},
}


def main() -> None:
    prices = load_prices(TICKERS, START, END)
    table, series = compare_strategies(prices, STRATEGIES, rebalance="Y")

    shown = table.copy()
    for col in ["total_return", "cagr", "ann_vol", "max_drawdown"]:
        shown[col] = table[col].map(pct)
    shown["sharpe"] = table["sharpe"].map("{:.2f}".format)
    print(f"Yearly rebalancing, {START} to {END}, no transaction costs\n")
    print(shown.to_string())

    save(plot_comparison(series), "strategy_comparison.png")


if __name__ == "__main__":
    main()
