"""Genera las gr?ficas comparativas finales de la Parte 1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "results" / "raw"
SUMMARY_DIR = ROOT_DIR / "results" / "summary"
FIGURES_DIR = ROOT_DIR / "figures"

ALGORITHMS = ["gd", "pso", "de", "ae"]

PROBLEMS = [
    ("rosenbrock", 2),
    ("rosenbrock", 3),
    ("rastrigin", 2),
    ("rastrigin", 3),
]

DISPLAY_NAMES = {
    "gd": "GD",
    "pso": "PSO",
    "de": "DE",
    "ae": "AE",
}


def load_summary():
    path = SUMMARY_DIR / "parte1_comparacion.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    df = pd.read_csv(path)

    if len(df) != 16:
        raise ValueError(
            f"Se esperaban 16 configuraciones y se encontraron {len(df)}."
        )

    return df


def load_raw(function_name, dimension, optimizer):
    path = RAW_DIR / f"{function_name}_{dimension}d_{optimizer}.csv"

    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    df = pd.read_csv(path)

    if len(df) != 30:
        raise ValueError(
            f"{path.name}: se esperaban 30 corridas y hay {len(df)}."
        )

    return df


def plot_success_rate(summary):
    labels = [
        "Rosenbrock 2D",
        "Rosenbrock 3D",
        "Rastrigin 2D",
        "Rastrigin 3D",
    ]

    x = np.arange(len(labels))
    width = 0.19

    fig, ax = plt.subplots(figsize=(12, 6))

    for index, optimizer in enumerate(ALGORITHMS):
        values = []

        for function_name, dimension in PROBLEMS:
            row = summary[
                (summary["function"] == function_name)
                & (summary["dimension"] == dimension)
                & (summary["optimizer"] == optimizer)
            ]

            values.append(float(row.iloc[0]["success_rate"]))

        ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=DISPLAY_NAMES[optimizer],
        )

    ax.set_ylabel("Tasa de ?xito (%)")
    ax.set_xlabel("Problema")
    ax.set_title("Tasa de ?xito por algoritmo")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "success_rate_comparison.png", dpi=200)
    plt.close(fig)


def plot_best_f_distribution():
    epsilon = 1e-16

    for function_name, dimension in PROBLEMS:
        data = []

        for optimizer in ALGORITHMS:
            df = load_raw(function_name, dimension, optimizer)
            values = np.maximum(
                df["best_f"].to_numpy(dtype=float),
                epsilon,
            )
            data.append(values)

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.boxplot(
            data,
            tick_labels=[DISPLAY_NAMES[a] for a in ALGORITHMS],
        )

        ax.set_yscale("log")
        ax.set_ylabel("Best f (escala logar?tmica)")
        ax.set_xlabel("Algoritmo")
        ax.set_title(
            f"Distribuci?n del resultado final - "
            f"{function_name.capitalize()} {dimension}D"
        )
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()

        output = (
            FIGURES_DIR
            / f"best_f_distribution_{function_name}_{dimension}d.png"
        )

        fig.savefig(output, dpi=200)
        plt.close(fig)


def plot_mean_best_f(summary):
    epsilon = 1e-16

    labels = [
        "Rosenbrock 2D",
        "Rosenbrock 3D",
        "Rastrigin 2D",
        "Rastrigin 3D",
    ]

    x = np.arange(len(labels))
    width = 0.19

    fig, ax = plt.subplots(figsize=(12, 6))

    for index, optimizer in enumerate(ALGORITHMS):
        values = []

        for function_name, dimension in PROBLEMS:
            row = summary[
                (summary["function"] == function_name)
                & (summary["dimension"] == dimension)
                & (summary["optimizer"] == optimizer)
            ]

            mean_value = float(row.iloc[0]["mean"])
            values.append(max(mean_value, epsilon))

        ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=DISPLAY_NAMES[optimizer],
        )

    ax.set_yscale("log")
    ax.set_ylabel("Media de best_f (escala logar?tmica)")
    ax.set_xlabel("Problema")
    ax.set_title("Error final medio por algoritmo")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mean_best_f_comparison.png", dpi=200)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_summary()

    plot_success_rate(summary)
    plot_best_f_distribution()
    plot_mean_best_f(summary)

    print("Gr?ficas generadas correctamente:")
    print(" - figures/success_rate_comparison.png")
    print(" - figures/mean_best_f_comparison.png")

    for function_name, dimension in PROBLEMS:
        print(
            f" - figures/best_f_distribution_"
            f"{function_name}_{dimension}d.png"
        )


if __name__ == "__main__":
    main()
