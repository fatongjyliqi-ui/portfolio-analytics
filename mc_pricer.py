"""
Monte-Carlo-Pricer fuer europaeische Optionen — importiert aus options.py.

Demonstriert:
  1. Naiver MC vs. antithetische Variaten (Varianzreduktion)
  2. Konvergenzanalyse gegen Black-Scholes (Preis und Standardfehler)
  3. Rechenzeit-Vergleich
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from options import black_scholes_call, black_scholes_put, mc_preis

# ---------------------------------------------------------------------------
# Parameter (ATM-Standardbox)
# ---------------------------------------------------------------------------

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
N = 100_000

# ---------------------------------------------------------------------------
# 1. Vergleichstabelle: BS  vs.  Naiver MC  vs.  Antithetischer MC
# ---------------------------------------------------------------------------

bs_call = black_scholes_call(S, K, T, r, sigma)
bs_put  = black_scholes_put(S, K, T, r, sigma)

t0 = time.perf_counter()
naiv_preis, naiv_se = mc_preis(S, K, T, r, sigma, "call", N, antithetisch=False, seed=42)
t_naiv = time.perf_counter() - t0

t0 = time.perf_counter()
anti_preis, anti_se = mc_preis(S, K, T, r, sigma, "call", N, antithetisch=True, seed=42)
t_anti = time.perf_counter() - t0

se_reduktion = (1.0 - anti_se / naiv_se) * 100.0

print()
print("=" * 65)
print(f"  European Call  (S=K={S}, T={T}, r={r}, σ={sigma})  —  n={N:,}")
print("-" * 65)
print(f"  {'Methode':<24} {'Preis':>8}  {'Std.fehler':>12}  {'Zeit (s)':>9}")
print("-" * 65)
print(f"  {'Black-Scholes (exakt)':<24} {bs_call:>8.4f}  {'—':>12}  {'—':>9}")
print(f"  {'Naiver MC':<24} {naiv_preis:>8.4f}  {naiv_se:>12.6f}  {t_naiv:>9.3f}")
print(f"  {'Antithetisch MC':<24} {anti_preis:>8.4f}  {anti_se:>12.6f}  {t_anti:>9.3f}")
print("-" * 65)
print(f"  Varianzreduktion durch antithetische Variaten: {se_reduktion:.1f}%")
print("=" * 65)
print()

# ---------------------------------------------------------------------------
# 2. Konvergenzanalyse
# ---------------------------------------------------------------------------

simulationszahlen = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

preise_naiv, fehler_naiv = [], []
preise_anti, fehler_anti = [], []

print("Konvergenzanalyse:")
print(f"{'n':>8}   {'Naiv Preis':>11}  {'±SE':>8}   {'Anti Preis':>11}  {'±SE':>8}")
print("-" * 58)

for n in simulationszahlen:
    p_n, se_n = mc_preis(S, K, T, r, sigma, "call", n, antithetisch=False, seed=42)
    p_a, se_a = mc_preis(S, K, T, r, sigma, "call", n, antithetisch=True,  seed=42)
    preise_naiv.append(p_n); fehler_naiv.append(se_n)
    preise_anti.append(p_a); fehler_anti.append(se_a)
    print(f"{n:>8,}   {p_n:>11.4f}  {se_n:>8.5f}   {p_a:>11.4f}  {se_a:>8.5f}")

# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------

plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(plots_dir, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Monte-Carlo-Konvergenz  —  European Call  (S=K={S}, T={T}, r={r}, σ={sigma})",
    fontsize=12, fontweight="bold",
)

n_arr   = np.array(simulationszahlen)
ci_naiv = np.array(fehler_naiv) * 1.96
ci_anti = np.array(fehler_anti) * 1.96

# --- Panel links: Preis-Konvergenz ---
ax = axes[0]
ax.errorbar(n_arr, preise_naiv, yerr=ci_naiv, fmt="o-", capsize=4,
            color="steelblue", linewidth=1.8, label="Naiver MC  ±95% CI")
ax.errorbar(n_arr, preise_anti, yerr=ci_anti, fmt="s--", capsize=4,
            color="tomato", linewidth=1.8, alpha=0.85,
            label="Antithetisch MC  ±95% CI")
ax.axhline(bs_call, color="black", linestyle="--", linewidth=1.5,
           label=f"Black-Scholes: {bs_call:.4f}")
ax.set_xscale("log")
ax.set_title("Preis-Konvergenz")
ax.set_xlabel("Anzahl Simulationen (log-Skala)")
ax.set_ylabel("Call-Preis")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

# --- Panel rechts: Standardfehler (log-log) ---
ax = axes[1]
ax.plot(n_arr, fehler_naiv, "o-", color="steelblue", linewidth=1.8,
        label="Naiver MC")
ax.plot(n_arr, fehler_anti, "s--", color="tomato", linewidth=1.8,
        label="Antithetisch MC")
# Theoretische 1/sqrt(n)-Kurve zur Orientierung
n_ref = np.linspace(n_arr[0], n_arr[-1], 300)
c_ref = fehler_naiv[0] * np.sqrt(n_arr[0])
ax.plot(n_ref, c_ref / np.sqrt(n_ref), "k:", linewidth=1.2, alpha=0.45,
        label="1/√n Referenz")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Standardfehler (log-log-Skala)")
ax.set_xlabel("Anzahl Simulationen")
ax.set_ylabel("Standardfehler")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

plt.tight_layout()
outpath = os.path.join(plots_dir, "mc_konvergenz.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nGespeichert: {outpath}")
