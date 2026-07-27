# MIT License
# Copyright (c) 2026 Brett Daley

import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

GRID = np.linspace(0.0, 1.0, num=21)


def get_best_index(scores: np.ndarray) -> tuple[int, ...]:
    idx = np.unravel_index(np.argmax(scores), scores.shape)
    print("best =", idx)
    return idx


def compute_auc(arr: Any, axis: int = -1) -> Any:
    T = arr.shape[axis] - 1
    area = np.trapezoid(arr, dx=1, axis=axis)
    return area / T


def plot_heatmap(scores: np.ndarray) -> None:
    mean = np.mean(scores, axis=-2)
    scores = aucs = compute_auc(mean)

    print("range = ", scores.min(), scores.max())

    idx = get_best_index(aucs)

    best_x, best_y, best_z = GRID[idx[0]], GRID[idx[1]], GRID[idx[2]]
    print(best_x, best_y, best_z)

    # 3. Extract Slices at the Optimal Point
    # Slice 1: Fix X, vary Y and Z
    slice_x = scores[idx[0], :, :]
    # Slice 2: Fix Y, vary X and Z
    slice_y = scores[:, idx[1], :]
    # Slice 3: Fix Z, vary X and Y
    slice_z = scores[:, :, idx[2]]

    # 4. Plotting the Slices (2D Heatmaps)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot X-Slice (Y vs Z)
    im1 = axes[0].imshow(
        slice_x.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="viridis",
        aspect="auto",
        vmin=0.0,
        vmax=0.25,
    )
    axes[0].set_title(rf"Slice at $\alpha={best_x:.2f}$")
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel(r"$\chi$")
    # Mark the optimal point
    axes[0].scatter(best_y, best_z, c="red", marker="x", s=100)

    # Plot Y-Slice (X vs Z)
    im2 = axes[1].imshow(
        slice_y.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="viridis",
        aspect="auto",
        vmin=0.0,
        vmax=0.25,
    )
    axes[1].set_title(rf"Slice at $\lambda={best_y:.2f}$")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$\chi$")
    axes[1].scatter(best_x, best_z, c="red", marker="x", s=100)

    # Plot Z-Slice (X vs Y)
    im3 = axes[2].imshow(
        slice_z.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="viridis",
        aspect="auto",
        vmin=0.0,
        vmax=0.25,
    )
    axes[2].set_title(rf"Slice at $\chi={best_z:.2f}$")
    axes[2].set_xlabel(r"$\alpha$")
    axes[2].set_ylabel(r"$\lambda$")
    axes[2].scatter(best_x, best_y, c="red", marker="x", s=100)

    os.makedirs("figures", exist_ok=True)
    path = "figures/3D_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"Created {path}")

    plt.close()


def plot_learning_curves(scores: np.ndarray) -> None:
    N = scores.shape[0]
    NUM_RUNS = scores.shape[-2]
    NUM_STEPS: int = scores.shape[-1] - 1
    print(f"{NUM_RUNS=}")

    scores *= 100.0  # Convert to percentage
    mean = np.mean(scores, axis=-2)
    std = np.std(scores, axis=-2, ddof=1)
    t_value = stats.t.ppf(0.975, NUM_RUNS - 1)
    conf95 = t_value * std / np.sqrt(NUM_RUNS)
    aucs = compute_auc(mean)
    print("AUCs =", aucs[:, :, 0])

    print("range =", aucs.min(), aucs.max())
    print("range (gate=1)", aucs[:, :, -1].min(), aucs[:, :, -1].max())
    print("range (gate=0)", aucs[:, :, 0].min(), aucs[:, :, 0].max())

    X = np.arange(NUM_STEPS + 1)

    # Gated
    axes = get_best_index(aucs)
    print(f"{axes=}")
    Y = mean[*axes]
    print("best AUC =", compute_auc(Y))
    C = conf95[*axes]
    plt.fill_between(X, Y - C, Y + C, alpha=0.25, linewidth=0)
    plt.plot(X, Y, label=r"Gated Q($\lambda$)")

    # Peng's (gate=1)
    idx = np.unravel_index(np.argmax(aucs[:, :, -1]), aucs.shape[:2])
    axes = (idx[0], idx[1], -1)
    print(f"{axes=}")
    Y = mean[*axes]
    print("best AUC =", compute_auc(Y))
    C = conf95[*axes]
    plt.fill_between(X, Y - C, Y + C, alpha=0.25, linewidth=0)
    plt.plot(X, Y, label=r"Peng's Q($\lambda$)")

    # Watkins' (gate=0)
    idx = np.unravel_index(np.argmax(aucs[:, :, 0]), aucs.shape[:2])
    axes = (idx[0], idx[1], 0)
    print(f"{axes=}")
    Y = mean[*axes]
    print("best AUC =", compute_auc(Y))
    C = conf95[*axes]
    plt.fill_between(X, Y - C, Y + C, alpha=0.25, linewidth=0)
    plt.plot(X, Y, label=r"Watkins' Q($\lambda$)")

    plt.xlabel("Time Step")
    plt.ylabel("Prediction Accuracy (%)")
    plt.xlim([0, NUM_STEPS])
    plt.ylim([0.0, 50.0])
    plt.legend(loc="upper left")
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    path = "figures/learning_curves.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"Created {path}")

    plt.close()


def main() -> None:
    scores = np.load("data/learning_curves.npy")
    print(scores.shape, flush=True)
    plot_heatmap(scores)
    plot_learning_curves(scores)


if __name__ == "__main__":
    matplotlib.style.use("plot.mplstyle")
    main()
