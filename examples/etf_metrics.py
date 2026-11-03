"""Return, risk and correlation table for six asset-class ETFs, 2015-2024."""

from _common import pct

from portfolio_analytics.data import load_prices
from portfolio_analytics.metrics import asset_summary, correlation_matrix

ETFS = {
    "SPY": "US equities",
    "QQQ": "US tech",
    "URTH": "World equities",
    "EXS1.DE": "DAX",
    "TLT": "US long Treasuries",
    "GLD": "Gold",
}
START, END = "2015-01-01", "2024-12-31"


def main() -> None:
    prices = load_prices(list(ETFS), START, END)
    print(f"{len(prices)} common trading days, {START} to {END}\n")

    table = asset_summary(prices)
    shown = table.copy()
    for col in ["ann_return", "ann_vol", "max_drawdown", "total_return"]:
        shown[col] = table[col].map(pct)
    shown["sharpe"] = table["sharpe"].map("{:.2f}".format)
    shown.index = [f"{ETFS[t]} ({t})" for t in shown.index]
    print(shown.to_string(), "\n")

    print("Correlation of daily returns")
    print(correlation_matrix(prices).round(2).to_string())


if __name__ == "__main__":
    main()
