"""Monte Carlo pricing of a European call: plain vs antithetic sampling,
convergence to the Black-Scholes price and 1/sqrt(n) error decay."""

import time

import matplotlib.pyplot as plt
import numpy as np
from _common import save

from portfolio_analytics.options import bs_call, mc_price

S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20
N = 100_000
PATH_COUNTS = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]


def main() -> None:
    exact = bs_call(S, K, T, R, SIGMA)

    print(f"European call, S=K={S:g}, T={T:g}, r={R}, sigma={SIGMA}, n={N:,}\n")
    print(f"{'method':<22}{'price':>10}{'std error':>12}{'time (s)':>10}")
    print(f"{'Black-Scholes':<22}{exact:>10.4f}")
    se = {}
    for name, anti in [("plain MC", False), ("antithetic MC", True)]:
        t0 = time.perf_counter()
        price, se[anti] = mc_price(S, K, T, R, SIGMA, "call", N, antithetic=anti, seed=42)
        print(f"{name:<22}{price:>10.4f}{se[anti]:>12.5f}{time.perf_counter() - t0:>10.3f}")
    ratio = se[True] / se[False]
    print(f"\nantithetic: std error {ratio - 1:+.0%}, variance {ratio**2 - 1:+.0%}\n")

    results = {
        anti: np.array([mc_price(S, K, T, R, SIGMA, "call", n, antithetic=anti, seed=42)
                        for n in PATH_COUNTS])
        for anti in (False, True)
    }
    n = np.array(PATH_COUNTS)

    fig, (ax_p, ax_e) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Monte Carlo convergence, European call", fontsize=12, fontweight="bold")
    for anti, (color, fmt, label) in {False: ("steelblue", "o-", "plain"),
                                      True: ("tomato", "s--", "antithetic")}.items():
        price, err = results[anti][:, 0], results[anti][:, 1]
        ax_p.errorbar(n, price, yerr=1.96 * err, fmt=fmt, capsize=4, color=color,
                      label=f"{label} (95% CI)")
        ax_e.plot(n, err, fmt, color=color, label=label)
    ax_p.axhline(exact, color="black", linestyle="--", label=f"Black-Scholes {exact:.4f}")
    ref = results[False][0, 1] * np.sqrt(n[0]) / np.sqrt(n)
    ax_e.plot(n, ref, "k:", alpha=0.5, label="1/sqrt(n) reference")
    ax_p.set(xscale="log", title="Price", xlabel="paths", ylabel="call price")
    ax_e.set(xscale="log", yscale="log", title="Standard error", xlabel="paths")
    for ax in (ax_p, ax_e):
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save(fig, "mc_convergence.png")


if __name__ == "__main__":
    main()
