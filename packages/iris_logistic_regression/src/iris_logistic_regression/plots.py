from pathlib import Path

import matplotlib.pyplot as plt


def plot_train_history(
    history: dict[str, list[float]],
    *,
    save: bool = False,
    save_dir: Path = Path("history.png"),
    show: bool = True,
) -> None:
    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 8),
        constrained_layout=True,
    )

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], label="Train")
    ax.plot(epochs, history["val_loss"], label="Validation")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["acc"], label="Train")
    ax.plot(epochs, history["val_acc"], label="Validation")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Learning Rate
    ax = axes[1, 0]
    ax.plot(epochs, history["lr"])
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)

    # Epoch Time / GPU Memory
    ax = axes[1, 1]

    if history.get("gpu_memory"):
        ax2 = ax.twinx()

        line1 = ax.plot(
            epochs,
            history["epoch_time"],
            label="Epoch Time",
        )

        line2 = ax2.plot(
            epochs,
            history["gpu_memory"],
            linestyle="--",
            label="GPU Memory",
        )

        ax.set_title("Runtime")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Time (s)")
        ax2.set_ylabel("Memory (GB)")
        ax.grid(True, alpha=0.3)

        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels)

    else:
        ax.plot(epochs, history["epoch_time"])
        ax.set_title("Epoch Time")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Time (s)")
        ax.grid(True, alpha=0.3)

    if save:
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_dir,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    plt.close(fig)
