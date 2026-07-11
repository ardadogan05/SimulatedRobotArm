from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_losses(epochs, train_losses, val_losses, output_path, log_scale=False):
    fig, ax = plt.subplots()

    ax.plot(epochs, train_losses, label="Training loss")
    ax.plot(epochs, val_losses, label="Validation loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Combined loss")

    if log_scale:
        ax.set_title("3-link Neural IK Policy Training Loss, Log Scale")
        ax.set_yscale("log")
    else:
        ax.set_title("3-link Neural IK Policy Training Loss")

    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    loss_file = Path("results/phase8/neural_ik_3link_losses.npz")
    results_dir = Path("results/phase8")
    results_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(loss_file)

    train_losses = data["train_losses"]
    val_losses = data["val_losses"]

    epochs = np.arange(1, len(train_losses) + 1)

    normal_path = results_dir / "neural_ik_3link_losses.png"
    log_path = results_dir / "neural_ik_3link_losses_log.png"

    plot_losses(
        epochs,
        train_losses,
        val_losses,
        normal_path,
        log_scale=False,
    )

    plot_losses(
        epochs,
        train_losses,
        val_losses,
        log_path,
        log_scale=True,
    )

    print(f"Saved normal plot to: {normal_path}")
    print(f"Saved log plot to: {log_path}")


if __name__ == "__main__":
    main()
