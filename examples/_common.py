"""Shared helpers for the example scripts."""

from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def save(fig, name: str) -> Path:
    """Save a figure to plots/<name> and close it."""
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return path


def pct(x: float) -> str:
    return f"{x * 100:+.2f}%"
