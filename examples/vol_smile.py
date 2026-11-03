"""Implied volatility smile from a live option chain (Yahoo Finance).

Uses out-of-the-money options only: puts below the spot, calls above it.
OTM options are the liquid ones, and deep in-the-money prices consist mostly
of intrinsic value, which makes the implied vol badly conditioned.
Implied vols are computed with this package and compared with Yahoo's.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from _common import save

from portfolio_analytics.options import implied_vol

RATE = 0.04


def otm_chain(ticker: str, expiry_index: int) -> tuple[pd.DataFrame, float, str, float]:
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    expiry = tk.options[expiry_index]
    T = (pd.Timestamp(expiry) - pd.Timestamp.now()).days / 365
    if T <= 0:
        raise RuntimeError(f"expiry {expiry} is not in the future")

    chain = tk.option_chain(expiry)
    puts = chain.puts.assign(kind="put")
    calls = chain.calls.assign(kind="call")
    df = pd.concat([puts[puts["strike"] < spot], calls[calls["strike"] >= spot]])
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["volume"].fillna(0) > 0)]
    df = df[df["strike"].between(0.75 * spot, 1.25 * spot)].copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    return df, spot, expiry, T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--expiry-index", type=int, default=3,
                        help="which listed expiry to use (0 = nearest)")
    args = parser.parse_args()

    df, spot, expiry, T = otm_chain(args.ticker, args.expiry_index)
    df["iv"] = [implied_vol(row.mid, spot, row.strike, T, RATE, row.kind)
                for row in df.itertuples()]
    df = df.dropna(subset=["iv"])
    df = df[df["iv"].between(0.01, 2.0)].sort_values("strike")
    if len(df) < 5:
        raise RuntimeError("too few liquid strikes; try another expiry")
    df["moneyness"] = df["strike"] / spot
    print(f"{args.ticker} {expiry}: {len(df)} liquid OTM strikes, T = {T:.3f}y, spot {spot:.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["moneyness"], df["iv"] * 100, "o-", color="steelblue", linewidth=2,
            label="implied vol (this package)")
    ax.plot(df["moneyness"], df["impliedVolatility"] * 100, "s--", color="tomato",
            alpha=0.75, label="Yahoo Finance implied vol")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.set(title=f"{args.ticker} volatility smile, expiry {expiry}, spot {spot:.2f} "
                 "(OTM puts left, OTM calls right)",
           xlabel="Moneyness K/S", ylabel="Implied volatility (%)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save(fig, "vol_smile.png")


if __name__ == "__main__":
    main()
