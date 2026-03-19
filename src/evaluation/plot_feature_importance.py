import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error


FIGURES_DIR = Path("results/figures")
TABLES_DIR = Path("results/tables")
PROCESSED_DIR = Path("data/processed")


def inverse_scale(y_scaled: np.ndarray, scaler: dict) -> np.ndarray:
    return y_scaled * scaler["target_scale"] + scaler["target_mean"]


def predict_real(model, X: np.ndarray, scaler: dict) -> np.ndarray:
    y_pred_scaled = model.predict(X, verbose=0).flatten()
    return inverse_scale(y_pred_scaled, scaler)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    X_test_df = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test_raw.csv")["Amount"].values

    with open(PROCESSED_DIR / "target_scaler.json", "r", encoding="utf-8") as f:
        scaler = json.load(f)

    # Cargar modelo
    model = tf.keras.models.load_model("models/A2_Adam.keras")

    # Predicción base
    X_test = X_test_df.values.astype(np.float32)
    y_pred_base = predict_real(model, X_test, scaler)
    baseline_mae = mean_absolute_error(y_test, y_pred_base)

    print(f"Baseline MAE: {baseline_mae:.4f}")

    # Permutation importance
    rng = np.random.default_rng(42)
    importances = []

    for col in X_test_df.columns:
        X_permuted = X_test_df.copy()
        X_permuted[col] = rng.permutation(X_permuted[col].values)

        y_pred_perm = predict_real(model, X_permuted.values.astype(np.float32), scaler)
        perm_mae = mean_absolute_error(y_test, y_pred_perm)

        importance = perm_mae - baseline_mae

        importances.append({
            "feature": col,
            "baseline_mae": baseline_mae,
            "permuted_mae": perm_mae,
            "importance_mae_increase": importance
        })

        print(f"{col}: +{importance:.6f}")

    # Ordenar de mayor a menor importancia
    imp_df = pd.DataFrame(importances).sort_values(
        by="importance_mae_increase",
        ascending=False
    )

    # Guardar tabla completa
    imp_df.to_csv(TABLES_DIR / "feature_importance.csv", index=False)

    # Graficar top 15
    top_n = 15
    top_df = imp_df.head(top_n).copy()
    top_df = top_df.iloc[::-1]  # para que la más importante quede arriba visualmente

    plt.figure(figsize=(10, 7))
    plt.barh(top_df["feature"], top_df["importance_mae_increase"])

    plt.xlabel("Incremento del MAE al permutar la variable")
    plt.ylabel("Variable")
    plt.title("Importancia aproximada de variables (Permutation Importance)")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
    print("Gráfica guardada en results/figures/feature_importance.png")
    print("Tabla guardada en results/tables/feature_importance.csv")

    plt.show()


if __name__ == "__main__":
    main()