"""Consolida los summaries experimentales de la Parte 1."""

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
SUMMARY_DIR = ROOT_DIR / "results" / "summary"
OUTPUT_PATH = SUMMARY_DIR / "parte1_comparacion.csv"

OPTIMIZER_ORDER = {
    "gd": 0,
    "pso": 1,
    "de": 2,
    "ae": 3,
}

FUNCTION_ORDER = {
    "rosenbrock": 0,
    "rastrigin": 1,
}


def main():
    files = sorted(
        path
        for path in SUMMARY_DIR.glob("*_summary.csv")
        if path.name != OUTPUT_PATH.name
    )

    rows = []

    for path in files:
        df = pd.read_csv(path)

        if len(df) != 1:
            raise ValueError(
                f"{path.name} debe contener exactamente una fila."
            )

        rows.append(df)

    if not rows:
        raise ValueError("No se encontraron archivos summary.")

    table = pd.concat(rows, ignore_index=True)

    table["_function_order"] = table["function"].map(FUNCTION_ORDER)
    table["_optimizer_order"] = table["optimizer"].map(OPTIMIZER_ORDER)

    if table["_function_order"].isna().any():
        raise ValueError("Se encontr? una funci?n desconocida.")

    if table["_optimizer_order"].isna().any():
        raise ValueError("Se encontr? un optimizador desconocido.")

    table = (
        table
        .sort_values(
            ["_function_order", "dimension", "_optimizer_order"]
        )
        .drop(columns=["_function_order", "_optimizer_order"])
        .reset_index(drop=True)
    )

    if len(table) != 16:
        raise ValueError(
            f"Se esperaban 16 configuraciones, "
            f"pero se encontraron {len(table)}."
        )

    table.to_csv(OUTPUT_PATH, index=False)

    print(f"Configuraciones consolidadas: {len(table)}")
    print(f"Archivo generado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
