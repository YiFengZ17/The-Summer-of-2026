"""Create the training-curve figure used by the Chinese project report."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "logs" / "cnn-adam.csv"
OUTPUT_DIR = PROJECT_ROOT / "report" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "cnn_training_curves.png"


def main() -> None:
    with LOG_PATH.open(newline="") as file:
        data = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]
    epochs = [row["epoch"] for row in data]
    train_acc = [row["train_acc"] * 100 for row in data]
    eval_acc = [row["val_acc"] * 100 for row in data]
    train_loss = [row["train_loss"] for row in data]
    eval_loss = [row["val_loss"] for row in data]
    best = max(data, key=lambda row: row["val_acc"])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#3f4650",
            "text.color": "#20252b",
            "axes.labelcolor": "#20252b",
            "xtick.color": "#4f5965",
            "ytick.color": "#4f5965",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    blue = "#3568a8"
    orange = "#d07a32"
    grid = "#d9dde3"

    axes[0].plot(
        epochs, train_acc,
        color=blue, marker="o", linewidth=2, label="Train accuracy",
    )
    axes[0].plot(
        epochs, eval_acc,
        color=orange, marker="s", linewidth=2, linestyle="--",
        label="Evaluation accuracy",
    )
    axes[0].scatter(
        [best["epoch"]], [best["val_acc"] * 100],
        color=orange, edgecolor="#20252b", linewidth=0.8, s=70, zorder=4,
    )
    axes[0].annotate(
        f"best: {best['val_acc'] * 100:.2f}% (epoch {int(best['epoch'])})",
        xy=(best["epoch"], best["val_acc"] * 100),
        xytext=(-126, -31), textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#4f5965"},
    )
    axes[0].set_title("CNN accuracy by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(epochs)
    axes[0].set_ylim(35, 80)
    axes[0].grid(axis="y", color=grid, linewidth=0.8)
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].plot(
        epochs, train_loss,
        color=blue, marker="o", linewidth=2, label="Train loss",
    )
    axes[1].plot(
        epochs, eval_loss,
        color=orange, marker="s", linewidth=2, linestyle="--",
        label="Evaluation loss",
    )
    axes[1].axvline(best["epoch"], color="#6b7280", linestyle=":", linewidth=1.2)
    axes[1].set_title("CNN loss by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_xticks(epochs)
    axes[1].grid(axis="y", color=grid, linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper right")

    fig.suptitle(
        "SmallCNN training curves on CIFAR-10",
        fontsize=14, fontweight="semibold",
    )
    fig.text(
        0.5, -0.025,
        "10 epochs, Adam, learning rate 1e-3, weight decay 5e-4; "
        "the current evaluation split is the official CIFAR-10 test split.",
        ha="center", color="#4f5965", fontsize=9,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
