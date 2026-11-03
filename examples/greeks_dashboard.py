"""Black-Scholes price and the five Greeks as functions of the spot price."""

import matplotlib.pyplot as plt
import numpy as np
from _common import save

from portfolio_analytics.options import bs_call, delta, gamma, rho, theta, vega

K, T, R, SIGMA = 100.0, 1.0, 0.05, 0.20


def main() -> None:
    S = np.linspace(50, 150, 300)
    args = (S, K, T, R, SIGMA)
    panels = [
        ("Call price", {"call": bs_call(*args)}),
        ("Delta  dV/dS", {"call": delta(*args, "call"), "put": delta(*args, "put")}),
        ("Gamma  d2V/dS2", {"call = put": gamma(*args)}),
        ("Vega  dV/dsigma", {"call = put": vega(*args)}),
        ("Theta  dV/dt (per year)", {"call": theta(*args, "call"), "put": theta(*args, "put")}),
        ("Rho  dV/dr", {"call": rho(*args, "call"), "put": rho(*args, "put")}),
    ]
    style = {"call": ("steelblue", "-"), "put": ("tomato", "--"),
             "call = put": ("mediumseagreen", "-")}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Black-Scholes Greeks (K={K:g}, T={T:g}y, r={R:.0%}, sigma={SIGMA:.0%})",
                 fontsize=14, fontweight="bold")
    for ax, (title, curves) in zip(axes.flat, panels, strict=True):
        for label, y in curves.items():
            color, ls = style[label]
            ax.plot(S, y, label=label, color=color, linestyle=ls, linewidth=2)
        ax.axvline(K, color="gray", linestyle=":", alpha=0.7)
        ax.set(title=title, xlabel="Spot S")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save(fig, "greeks_dashboard.png")


if __name__ == "__main__":
    main()
