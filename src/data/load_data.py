import pandas as pd
from src.config import DATASET_PATH


def load_dataset() -> pd.DataFrame:
    """
    Carga el dataset original desde Excel.
    """
    df = pd.read_excel(DATASET_PATH)
    return df


def main():
    df = load_dataset()

    print("\n=== DATASET CARGADO ===")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    print("\n=== COLUMNAS ===")
    for col in df.columns:
        print(col)

    print("\n=== TIPOS DE DATOS ===")
    print(df.dtypes)

    print("\n=== PRIMERAS 5 FILAS ===")
    print(df.head())

    print("\n=== VALORES NULOS POR COLUMNA ===")
    print(df.isnull().sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()