"""Animaciones de optimizaci?n para la Parte 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from src.experiments.run_part1 import run_single


ROOT_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT_DIR / "figures"

DISPLAY_NAMES = {
    "gd": "Descenso por gradiente",
    "de": "Evoluci?n diferencial",
}


def rosenbrock_2d(x, y):
    """Funci?n de Rosenbrock est?ndar en dos dimensiones."""
    return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2


def sample_history(history, max_frames):
    """Reduce un historial largo conservando inicio y final."""
    if not history:
        raise ValueError("El optimizador no produjo historial.")

    count = min(len(history), max_frames)

    indices = np.unique(
        np.linspace(
            0,
            len(history) - 1,
            count,
            dtype=int,
        )
    )

    return [history[i] for i in indices]


def calculate_limits(points):
    """Calcula l?mites de visualizaci?n alrededor de la trayectoria."""
    x_values = np.append(points[:, 0], 1.0)
    y_values = np.append(points[:, 1], 1.0)

    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))

    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)

    x_padding = 0.15 * x_span
    y_padding = 0.15 * y_span

    x_min = max(-5.0, x_min - x_padding)
    x_max = min(10.0, x_max + x_padding)
    y_min = max(-5.0, y_min - y_padding)
    y_max = min(10.0, y_max + y_padding)

    return x_min, x_max, y_min, y_max


def create_animation(optimizer, seed, max_frames, fps):
    """Ejecuta un optimizador y genera su GIF sobre Rosenbrock 2D."""
    result = run_single(
        function_name="rosenbrock",
        dimension=2,
        optimizer_name=optimizer,
        seed=seed,
    )

    history = sample_history(result["history"], max_frames)

    points = np.asarray(
        [entry["best_x"] for entry in history],
        dtype=float,
    )

    x_min, x_max, y_min, y_max = calculate_limits(points)

    x_grid = np.linspace(x_min, x_max, 250)
    y_grid = np.linspace(y_min, y_max, 250)

    X, Y = np.meshgrid(x_grid, y_grid)
    Z = rosenbrock_2d(X, Y)

    # La transformaci?n logar?tmica permite visualizar mejor
    # el valle estrecho de Rosenbrock.
    Z_display = np.log10(1.0 + Z)

    fig, ax = plt.subplots(figsize=(8, 7))

    contour = ax.contourf(
        X,
        Y,
        Z_display,
        levels=35,
    )

    fig.colorbar(
        contour,
        ax=ax,
        label="log10(1 + f(x))",
    )

    ax.plot(
        [1.0],
        [1.0],
        marker="*",
        markersize=14,
        linestyle="None",
        label="M?nimo global (1, 1)",
    )

    path_line, = ax.plot(
        [],
        [],
        linewidth=2,
        label="Mejor soluci?n encontrada",
    )

    current_point, = ax.plot(
        [],
        [],
        marker="o",
        markersize=8,
        linestyle="None",
    )

    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "alpha": 0.85},
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="lower right")

    display_name = DISPLAY_NAMES[optimizer]

    ax.set_title(
        f"{display_name} - Rosenbrock 2D - seed {seed}"
    )

    def update(frame):
        entry = history[frame]

        x_path = points[: frame + 1, 0]
        y_path = points[: frame + 1, 1]

        path_line.set_data(x_path, y_path)

        current_point.set_data(
            [points[frame, 0]],
            [points[frame, 1]],
        )

        info_text.set_text(
            f"Paso: {entry['step']}\n"
            f"Evaluaciones equivalentes: "
            f"{entry['equivalent_evaluations']}\n"
            f"Mejor f: {entry['best_f']:.6g}"
        )

        return path_line, current_point, info_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=1000 / fps,
        repeat=True,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        FIGURES_DIR
        / f"{optimizer}_rosenbrock_2d.gif"
    )

    animation.save(
        output_path,
        writer=PillowWriter(fps=fps),
    )

    plt.close(fig)

    print(
        f"GIF generado: {output_path} | "
        f"frames={len(history)} | "
        f"best_f={result['best_f']:.8g}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera GIF de GD y DE para Rosenbrock 2D."
    )

    parser.add_argument(
        "--optimizer",
        choices=["gd", "de", "both"],
        default="both",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=12,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_frames < 2:
        raise ValueError("--max-frames debe ser >= 2.")

    if args.fps < 1:
        raise ValueError("--fps debe ser >= 1.")

    optimizers = (
        ["gd", "de"]
        if args.optimizer == "both"
        else [args.optimizer]
    )

    for optimizer in optimizers:
        create_animation(
            optimizer=optimizer,
            seed=args.seed,
            max_frames=args.max_frames,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
