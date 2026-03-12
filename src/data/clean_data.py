import pandas as pd
from src.data.load_data import load_dataset
from src.config import PROCESSED_DATA_DIR


DROP_COLUMNS = [
    "index",
    "Order ID",
    "SKU",
    "ASIN",
    "Style",
    "promotion-ids",
    "Unnamed: 22",
    "ship-city",
    "ship-state",
    "ship-postal-code",
    "ship-country",
    "currency",
]

CATEGORICAL_COLUMNS = [
    "Status",
    "Fulfilment",
    "Sales Channel",
    "ship-service-level",
    "Category",
    "Size",
    "Courier Status",
    "fulfilled-by",
]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def clean_dataset():
    df = load_dataset()

    # Normalizar nombres de columnas
    df = normalize_column_names(df)

    print("Filas iniciales:", df.shape)
    print("\nColumnas detectadas:")
    print(list(df.columns))

    # Eliminar columnas no útiles
    drop_existing = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=drop_existing, errors="ignore")

    # Eliminar filas sin target
    if "Amount" not in df.columns:
        raise KeyError("No se encontró la columna 'Amount' en el dataset.")

    df = df.dropna(subset=["Amount"]).copy()

    # Procesar fecha
    if "Date" not in df.columns:
        raise KeyError("No se encontró la columna 'Date' en el dataset.")

    df["Date"] = pd.to_datetime(df["Date"], format="%m-%d-%y", errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["weekday"] = df["Date"].dt.weekday
    df = df.drop(columns=["Date"])

    # Procesar B2B
    if "B2B" not in df.columns:
        raise KeyError("No se encontró la columna 'B2B' en el dataset.")

    df["B2B"] = df["B2B"].fillna(False)
    df["B2B"] = df["B2B"].astype(bool).astype(int)

    # Rellenar categóricas solo si existen
    existing_categorical = [col for col in CATEGORICAL_COLUMNS if col in df.columns]

    for col in existing_categorical:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # Casos especiales
    if "fulfilled-by" in df.columns:
        df["fulfilled-by"] = df["fulfilled-by"].replace("", "Unknown")

    if "Courier Status" in df.columns:
        df["Courier Status"] = df["Courier Status"].replace("", "Unknown")

    print("\nFilas después limpieza:", df.shape)

    print("\nNulos restantes por columna:")
    print(df.isnull().sum().sort_values(ascending=False))

    print("\nColumnas finales:")
    print(list(df.columns))

    return df


def main():
    df = clean_dataset()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "amazon_clean.csv"

    df.to_csv(output_path, index=False)

    print("\nDataset limpio guardado en:")
    print(output_path)


if __name__ == "__main__":
    main()