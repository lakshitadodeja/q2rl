"""Render the verified CSV data; this script performs no model evaluation."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, help="Defaults to the study's figures directory")
    args = parser.parse_args()
    output = args.output_dir or args.study_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    loss = read_rows(args.study_dir / "data/training_loss.csv")
    success = read_rows(args.study_dir / "data/success_rate.csv")
    verified = read_rows(args.study_dir / "data/verified_checkpoints.csv")
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 17, "axes.labelsize": 13})
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    ax.plot([r["gradient_updates"] for r in loss], [r["loss"] for r in loss], lw=2)
    ax.set(title="Lift Diffusion Policy: Training Loss", xlabel="Gradient Updates", ylabel="Training Loss")
    ax.grid(alpha=0.25)
    fig.savefig(output / "training_loss.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    ax.axhspan(50, 60, alpha=0.12, color="tab:blue", label="Target: 50–60%")
    ax.plot([r["gradient_updates"] for r in success], [100 * r["success_rate"] for r in success],
            marker="o", lw=2, label="Preliminary evaluation (20 rollouts)")
    ax.set(title="Lift Diffusion Policy: Evaluation Success Rate", xlabel="Gradient Updates",
           ylabel="Success Rate (%)", ylim=(0, 100))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)
    fig.savefig(output / "success_rate.png", dpi=180)
    plt.close(fig)
    by_epoch = {int(r["epoch"]): r["success_rate"] for r in success}
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    positions = list(range(len(verified)))
    ax.axhspan(50, 60, alpha=0.10, color="tab:blue", label="Target: 50–60%")
    first = ax.bar([x - 0.2 for x in positions], [100 * by_epoch[int(r["epoch"])] for r in verified],
                   width=0.4, label="Preliminary (20 rollouts)")
    second = ax.bar([x + 0.2 for x in positions], [100 * r["success_rate_50"] for r in verified],
                    width=0.4, label="Independent evaluation (50 rollouts)")
    ax.bar_label(first, fmt="%.0f%%", padding=3, fontsize=11)
    ax.bar_label(second, fmt="%.0f%%", padding=3, fontsize=11)
    ax.set_xticks(positions, [str(int(r["epoch"])) for r in verified])
    ax.set(title="Lift: Candidate Checkpoint Comparison", xlabel="Checkpoint Epoch",
           ylabel="Success Rate (%)", ylim=(0, 100))
    ax.legend(loc="upper left", fontsize=10)
    fig.savefig(output / "checkpoint_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Wrote three figures to {output}")


if __name__ == "__main__":
    main()
