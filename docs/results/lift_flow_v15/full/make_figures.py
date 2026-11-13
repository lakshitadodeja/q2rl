"""Plot actual training-time evaluations supplied by the user; no model evaluation."""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter, FuncFormatter

with (ROOT / "evaluation_metrics.csv").open() as f:
    rows = list(csv.DictReader(f))
epochs = np.array([int(r["epoch"]) for r in rows])
updates = np.array([int(r["gradient_updates"]) for r in rows])
rates = np.array([float(r["success_rate"]) for r in rows])
assert np.array_equal(epochs, np.r_[np.arange(1, 51), np.arange(100, 2001, 50)])
assert np.array_equal(updates, epochs * 100)
assert np.all((rates >= 0) & (rates <= 1))
assert np.allclose(rates * 50, np.round(rates * 50))
assert rates[7] == .54 and rates[-1] == 1

BLUE, TEAL, INK, MUTED = "#397F9C", "#007D78", "#162B40", "#526777"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "svg.fonttype": "none", "savefig.facecolor": "white"})

def plot(early):
    fig = plt.figure(figsize=(10, 5.9), facecolor="white")
    ax = fig.add_axes([.095, .20, .865, .55])
    fig.text(.095, .95, "Flow matching | " + ("early training" if early else "full training"),
             fontsize=20, weight="bold", va="top")
    fig.text(.095, .89, "Lift · 200 demonstrations (100%) · v15 data · robosuite 1.5.1 · training seed 1",
             fontsize=10.5, color=MUTED, va="top")
    select = epochs <= 50 if early else np.ones(len(rows), dtype=bool)
    ax.axhspan(.5, .6, color="#E4F1EB", zorder=0, label="Target: 50–60%")
    ax.plot(updates[select], rates[select], color=BLUE, lw=2, marker="o",
            markersize=4, markerfacecolor="white", markeredgewidth=1.2,
            label="Training-time evaluation (50 rollouts)", zorder=3)
    ax.set_ylim(-.025, 1.045)
    ax.set_yticks(np.arange(0, 1.01, .1))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("Success rate", labelpad=10)
    ax.set_xlabel("BC gradient updates", labelpad=12)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5EBEF", lw=.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color("#BFCBD3")
    ax.tick_params(length=0, pad=7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, ncol=2, fontsize=10,
              loc="lower left", bbox_to_anchor=(-.01, 1.045), borderaxespad=0)
    if early:
        ax.set_xlim(0, 5100)
        ax.set_xticks(np.arange(0, 5001, 500))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.scatter([800], [.54], s=95, color=TEAL, edgecolor="white", zorder=5)
        ax.annotate("Epoch 8 · 800 updates\n54% (27/50)", xy=(800, .54),
                    xytext=(1800, .24), fontsize=11, color=TEAL, weight="bold",
                    arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.2},
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 5})
        foot = "Epochs 1–50; 100 gradient updates per epoch. The shaded band is a target, not a confidence interval."
        stem = "02_early_training_success"
    else:
        ax.set_xlim(0, 204000)
        ax.set_xticks(np.arange(0, 200001, 25000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x/1000:.0f}k"))
        ax.annotate("Epoch 2000 · 200k updates\n100% (50/50)", xy=(200000, 1),
                    xytext=(146000, .75), fontsize=11, color=TEAL, ha="center",
                    arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.2})
        foot = "Evaluated every epoch for epochs 1–50, then every 50 epochs. 100 gradient updates per epoch."
        stem = "01_full_training_success"
    fig.text(.095, .075, foot, fontsize=9, color=MUTED)
    fig.text(.095, .038, "Raw results from one training run; no smoothing. Lines guide the eye between evaluations. Job 8946875.",
             fontsize=9, color=MUTED)
    for suffix in ["png", "svg"]:
        fig.savefig(ROOT / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)

plot(False)
plot(True)
print(f"Validated {len(rows)} evaluations; saved full and early curves as PNG/SVG.")
