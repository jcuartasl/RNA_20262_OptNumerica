"""Runner de experimentos para la Parte 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from src.functions import get_function
from src.metrics.statistics import calculate_statistics
from src.optimizers.gradient_descent import optimize as optimize_gd
from src.optimizers.PSO import optimize as optimize_pso
from src.optimizers.differential_evolution import optimize as optimize_de


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "part1.yaml"

RAW_RESULTS_DIR = ROOT_DIR / "results" / "raw"
SUMMARY_RESULTS_DIR = ROOT_DIR / "results" / "summary"


def load_config() -> Dict[str, Any]:
    """Carga la configuración oficial de la Parte 1."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_single(
    function_name: str,
    dimension: int,
    optimizer_name: str,
    seed: int,
    learning_rate: float | None = None,
) -> Dict[str, Any]:
    """Ejecuta una única corrida experimental."""

    config = load_config()

    function_name = function_name.lower()
    optimizer_name = optimizer_name.lower()

    # ---------------------------------------------------------
    # Validaciones generales
    # ---------------------------------------------------------

    if function_name not in config["functions"]:
        available = ", ".join(config["functions"].keys())
        raise ValueError(
            f"Función desconocida '{function_name}'. "
            f"Disponibles: {available}"
        )

    if dimension not in config["dimensions"]:
        raise ValueError(
            f"Dimensión {dimension} no configurada. "
            f"Disponibles: {config['dimensions']}"
        )

    if learning_rate is not None:
        if learning_rate <= 0:
            raise ValueError("learning_rate debe ser mayor que 0.")

        if optimizer_name != "gd":
            raise ValueError(
                "--learning-rate solamente puede utilizarse "
                "con el optimizador gd."
            )

    # ---------------------------------------------------------
    # Función objetivo
    # ---------------------------------------------------------

    function = get_function(function_name)
    bounds = config["functions"][function_name]["bounds"]

    # ---------------------------------------------------------
    # Configuración experimental común
    # ---------------------------------------------------------

    budget = int(config["experiment"]["budget_equivalent"])
    tolerance = float(config["experiment"]["tolerance"])

    # ---------------------------------------------------------
    # Selección del optimizador
    # ---------------------------------------------------------

    if optimizer_name == "gd":
        optimizer_config = dict(config["gradient_descent"])
        optimizer_config["tolerance"] = tolerance

        if learning_rate is not None:
            optimizer_config["learning_rate"] = learning_rate

        result = optimize_gd(
            function=function,
            bounds=bounds,
            dimension=dimension,
            seed=seed,
            budget=budget,
            config=optimizer_config,
        )

        result["learning_rate"] = float(
            optimizer_config["learning_rate"]
        )

    elif optimizer_name == "pso":
        optimizer_config = dict(config["pso"])
        optimizer_config["tolerance"] = tolerance

        result = optimize_pso(
            function=function,
            bounds=bounds,
            dimension=dimension,
            seed=seed,
            budget=budget,
            config=optimizer_config,
        )

    elif optimizer_name == "de":
        optimizer_config = dict(config["differential_evolution"])
        optimizer_config["tolerance"] = tolerance

        result = optimize_de(
            function=function,
            bounds=bounds,
            dimension=dimension,
            seed=seed,
            budget=budget,
            config=optimizer_config,
        )

    else:
        raise ValueError(
            f"Optimizador desconocido '{optimizer_name}'. "
            "Disponibles: gd, pso, de"
        )

    # ---------------------------------------------------------
    # Metadatos de la corrida
    # ---------------------------------------------------------

    result["function"] = function_name
    result["dimension"] = dimension
    result["optimizer"] = optimizer_name
    result["seed"] = seed
    result["budget"] = budget
    result["tolerance"] = tolerance

    return result


def result_to_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte el resultado de una corrida en una fila de CSV."""

    row = {
        "function": result["function"],
        "dimension": result["dimension"],
        "optimizer": result["optimizer"],
        "seed": result["seed"],
        "budget": result["budget"],
        "tolerance": result["tolerance"],
        "best_f": result["best_f"],
        "best_x": json.dumps(result["best_x"]),
        "f_evaluations": result["f_evaluations"],
        "gradient_evaluations": result["gradient_evaluations"],
        "equivalent_evaluations": result["equivalent_evaluations"],
        "iterations": result["iterations"],
        "success": result["success"],
    }

    if "learning_rate" in result:
        row["learning_rate"] = result["learning_rate"]

    if "initial_x" in result:
        row["initial_x"] = json.dumps(result["initial_x"])

    return row


def run_batch(
    function_name: str,
    dimension: int,
    optimizer_name: str,
    learning_rate: float | None = None,
) -> None:
    """Ejecuta todas las semillas configuradas y guarda CSV."""

    config = load_config()

    seeds: List[int] = list(config["experiment"]["seeds"])
    runs = int(config["experiment"]["runs"])

    if len(seeds) != runs:
        raise ValueError(
            f"experiment.runs={runs}, pero se configuraron "
            f"{len(seeds)} semillas."
        )

    print()
    print("=" * 60)
    print("PARTE 1 - EJECUCIÓN DE MÚLTIPLES CORRIDAS")
    print("=" * 60)
    print(f"Function:  {function_name}")
    print(f"Dimension: {dimension}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Runs:      {runs}")

    if optimizer_name == "gd":
        effective_lr = (
            learning_rate
            if learning_rate is not None
            else float(config["gradient_descent"]["learning_rate"])
        )
        print(f"Learning rate: {effective_lr}")

    print("-" * 60)

    results: List[Dict[str, Any]] = []

    for index, seed in enumerate(seeds, start=1):
        result = run_single(
            function_name=function_name,
            dimension=dimension,
            optimizer_name=optimizer_name,
            seed=seed,
            learning_rate=learning_rate,
        )

        results.append(result)

        print(
            f"[{index:02d}/{runs:02d}] "
            f"seed={seed:2d} | "
            f"best_f={result['best_f']:.8g} | "
            f"E_eq={result['equivalent_evaluations']} | "
            f"success={result['success']}"
        )

    global_minimum = float(
        config["functions"][function_name]["global_minimum"]
    )
    tolerance = float(config["experiment"]["tolerance"])

    statistics = calculate_statistics(
        results,
        global_minimum=global_minimum,
        tolerance=tolerance,
    )

    print("-" * 60)
    print("RESUMEN ESTADÍSTICO")
    print("-" * 60)
    print(f"Runs:          {statistics['runs']}")
    print(f"Mean:          {statistics['mean']}")
    print(f"Std:           {statistics['std']}")
    print(f"Best:          {statistics['best']}")
    print(f"Worst:         {statistics['worst']}")
    print(f"Successes:     {statistics['successes']}")
    print(f"Success rate:  {statistics['success_rate']:.2f}%")
    print("-" * 60)

    # ---------------------------------------------------------
    # Crear carpetas de resultados
    # ---------------------------------------------------------

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Nombre de archivos
    # ---------------------------------------------------------

    base_name = (
        f"{function_name}_{dimension}d_{optimizer_name}"
    )

    # Solo los pilotos explícitos de GD llevan LR en el nombre.
    if optimizer_name == "gd" and learning_rate is not None:
        lr_text = format(learning_rate, ".10g").replace(".", "p")
        base_name += f"_lr_{lr_text}"

    raw_path = RAW_RESULTS_DIR / f"{base_name}.csv"
    summary_path = SUMMARY_RESULTS_DIR / f"{base_name}_summary.csv"

    # ---------------------------------------------------------
    # Guardar resultados por corrida
    # ---------------------------------------------------------

    raw_rows = [result_to_row(result) for result in results]
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)

    # ---------------------------------------------------------
    # Guardar resumen
    # ---------------------------------------------------------

    summary_row = {
        "function": function_name,
        "dimension": dimension,
        "optimizer": optimizer_name,
        "runs": statistics["runs"],
        "mean": statistics["mean"],
        "std": statistics["std"],
        "best": statistics["best"],
        "worst": statistics["worst"],
        "successes": statistics["successes"],
        "success_rate": statistics["success_rate"],
        "budget": int(config["experiment"]["budget_equivalent"]),
        "tolerance": tolerance,
    }

    if optimizer_name == "gd":
        summary_row["learning_rate"] = (
            learning_rate
            if learning_rate is not None
            else float(config["gradient_descent"]["learning_rate"])
        )

    pd.DataFrame([summary_row]).to_csv(
        summary_path,
        index=False,
    )

    print(f"Raw CSV:       {raw_path}")
    print(f"Summary CSV:   {summary_path}")
    print("=" * 60)


def print_single_result(result: Dict[str, Any]) -> None:
    """Muestra en consola el resultado de una única corrida."""

    print()
    print("=" * 60)
    print("PARTE 1 - EJECUCIÓN INDIVIDUAL")
    print("=" * 60)
    print(f"Function:               {result['function']}")
    print(f"Dimension:              {result['dimension']}")
    print(f"Optimizer:              {result['optimizer']}")
    print(f"Seed:                   {result['seed']}")

    if "learning_rate" in result:
        print(f"Learning rate:          {result['learning_rate']}")

    if "initial_x" in result:
        print(f"Initial x:              {result['initial_x']}")

    print(f"Best x:                 {result['best_x']}")
    print(f"Best f:                 {result['best_f']}")
    print(f"F evaluations:          {result['f_evaluations']}")
    print(
        f"Gradient evaluations:   "
        f"{result['gradient_evaluations']}"
    )
    print(
        f"Equivalent evaluations: "
        f"{result['equivalent_evaluations']}"
    )
    print(f"Iterations:             {result['iterations']}")
    print(f"Success:                {result['success']}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de línea de comandos."""

    parser = argparse.ArgumentParser(
        description="Runner de experimentos de la Parte 1."
    )

    parser.add_argument(
        "--function",
        required=True,
        choices=["rosenbrock", "rastrigin"],
        help="Función objetivo.",
    )

    parser.add_argument(
        "--dim",
        required=True,
        type=int,
        choices=[2, 3],
        help="Dimensión del problema.",
    )

    parser.add_argument(
        "--optimizer",
        required=True,
        choices=["gd", "pso", "de"],
        help="Optimizador: gd, pso o de.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate para Gradient Descent.",
    )

    run_group = parser.add_mutually_exclusive_group(required=True)

    run_group.add_argument(
        "--seed",
        type=int,
        help="Ejecuta una única corrida con esta seed.",
    )

    run_group.add_argument(
        "--all-seeds",
        action="store_true",
        help="Ejecuta todas las semillas configuradas.",
    )

    return parser.parse_args()


def main() -> None:
    """Punto de entrada del runner."""

    args = parse_args()

    if args.learning_rate is not None and args.optimizer != "gd":
        raise ValueError(
            "--learning-rate solamente puede utilizarse "
            "con --optimizer gd."
        )

    if args.all_seeds:
        run_batch(
            function_name=args.function,
            dimension=args.dim,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
        )
    else:
        result = run_single(
            function_name=args.function,
            dimension=args.dim,
            optimizer_name=args.optimizer,
            seed=args.seed,
            learning_rate=args.learning_rate,
        )

        print_single_result(result)


if __name__ == "__main__":
    main()