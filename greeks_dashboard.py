"""
Greeks-Dashboard aus Tag 4 — importiert aus options.py.
Speichert in plots/greeks_dashboard.png.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from options import black_scholes_call, delta, gamma, vega, theta, rho

S_range = np.linspace(50, 150, 300)
K, T, r, sigma = 100, 1.0, 0.05, 0.20

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    "Greeks-Dashboard  (K=100, T=1 Jahr, r=5%, σ=20%)",
    fontsize=14, fontweight="bold",
)

panels = [
    (
        axes[0, 0], "Call-Preis",
        [
            (S_range, black_scholes_call(S_range, K, T, r, sigma), "Call", "steelblue", "-"),
        ],
    ),
    (
        axes[0, 1], "Delta  (dPreis/dS)",
        [
            (S_range, delta(S_range, K, T, r, sigma, "call"), "Call", "steelblue", "-"),
            (S_range, delta(S_range, K, T, r, sigma, "put"),  "Put",  "tomato",    "--"),
        ],
    ),
    (
        axes[0, 2], "Gamma  (d²Preis/dS²)",
        [
            (S_range, gamma(S_range, K, T, r, sigma), "Call = Put", "mediumseagreen", "-"),
        ],
    ),
    (
        axes[1, 0], "Vega  (dPreis/dσ,  pro 1.0 σ)",
        [
            (S_range, vega(S_range, K, T, r, sigma), "Call = Put", "mediumseagreen", "-"),
        ],
    ),
    (
        axes[1, 1], "Theta  (dPreis/dT,  pro Jahr)",
        [
            (S_range, theta(S_range, K, T, r, sigma, "call"), "Call", "steelblue", "-"),
            (S_range, theta(S_range, K, T, r, sigma, "put"),  "Put",  "tomato",    "--"),
        ],
    ),
    (
        axes[1, 2], "Rho  (dPreis/dr)",
        [
            (S_range, rho(S_range, K, T, r, sigma, "call"), "Call", "steelblue", "-"),
            (S_range, rho(S_range, K, T, r, sigma, "put"),  "Put",  "tomato",    "--"),
        ],
    ),
]

for ax, title, lines in panels:
    for x, y, label, color, ls in lines:
        ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=2)
    ax.axvline(K, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Aktienkurs S", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

# ATM-Annotation auf jedem Panel
for ax in axes.flatten():
    ylim = ax.get_ylim()
    ax.text(K + 1, ylim[0] + (ylim[1] - ylim[0]) * 0.04, "ATM",
            fontsize=7, color="gray")

plt.tight_layout()
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(plots_dir, exist_ok=True)
outpath = os.path.join(plots_dir, "greeks_dashboard.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.show()
print(f"Gespeichert: {outpath}")
