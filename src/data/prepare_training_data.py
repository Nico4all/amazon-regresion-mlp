import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PROCESSED_DATA_DIR, RANDOM_STATE


TARGET_COLUMN = "Amount"

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

NUMERICAL_COLUMNS = [
    "Qty",
    "B2B",
    "month",
    "day",
    "weekday",
]


def load_clean_dataset() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "amazon_clean.csv"
    df = pd.read_csv(path)
    return df


def build_preprocessor():
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
            ("num", numerical_transformer, NUMERICAL_COLUMNS),
        ]
    )

    return preprocessor


def main():
    df = load_clean_dataset()

    print("Dataset limpio cargado:", df.shape)

    X = df[CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    print("\nShape X original:", X.shape)
    print("Shape y original:", y.shape)

    # Split inicial: train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # Split secundario: train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print("\nSplit completado:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Escalado de y
    y_scaler = StandardScaler()

    y_train_scaled = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.to_numpy().reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.to_numpy().reshape(-1, 1))

    # Nombres de columnas transformadas
    feature_names = preprocessor.get_feature_names_out().tolist()

    print("\nShapes procesados:")
    print("X_train_processed:", X_train_processed.shape)
    print("X_val_processed:", X_val_processed.shape)
    print("X_test_processed:", X_test_processed.shape)
    print("Número de features finales:", len(feature_names))

    # Guardar datasets procesados
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train_processed, columns=feature_names).to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_val_processed, columns=feature_names).to_csv(PROCESSED_DATA_DIR / "X_val.csv", index=False)
    pd.DataFrame(X_test_processed, columns=feature_names).to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)

    pd.DataFrame(y_train_scaled, columns=["Amount"]).to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    pd.DataFrame(y_val_scaled, columns=["Amount"]).to_csv(PROCESSED_DATA_DIR / "y_val.csv", index=False)
    pd.DataFrame(y_test_scaled, columns=["Amount"]).to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    # Guardar target sin escalar para evaluación final
    pd.DataFrame(y_train, columns=["Amount"]).to_csv(PROCESSED_DATA_DIR / "y_train_raw.csv", index=False)
    pd.DataFrame(y_val, columns=["Amount"]).to_csv(PROCESSED_DATA_DIR / "y_val_raw.csv", index=False)
    pd.DataFrame(y_test, columns=["Amount"]).to_csv(PROCESSED_DATA_DIR / "y_test_raw.csv", index=False)

    # Guardar nombres de features
    with open(PROCESSED_DATA_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    # Guardar parámetros del scaler de y
    scaler_params = {
        "target_mean": float(y_scaler.mean_[0]),
        "target_scale": float(y_scaler.scale_[0]),
        "target_var": float(y_scaler.var_[0]),
    }

    with open(PROCESSED_DATA_DIR / "target_scaler.json", "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, ensure_ascii=False, indent=2)

    print("\nArchivos generados en data/processed:")
    print("- X_train.csv")
    print("- X_val.csv")
    print("- X_test.csv")
    print("- y_train.csv")
    print("- y_val.csv")
    print("- y_test.csv")
    print("- y_train_raw.csv")
    print("- y_val_raw.csv")
    print("- y_test_raw.csv")
    print("- feature_columns.json")
    print("- target_scaler.json")


if __name__ == "__main__":
    main()