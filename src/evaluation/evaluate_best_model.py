import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras


def load_data():

    base = Path("data/processed")

    X_test = pd.read_csv(base / "X_test.csv").values.astype("float32")

    y_test_scaled = pd.read_csv(base / "y_test.csv").values.flatten()
    y_test_raw = pd.read_csv(base / "y_test_raw.csv").values.flatten()

    with open(base / "target_scaler.json") as f:
        scaler = json.load(f)

    return X_test, y_test_scaled, y_test_raw, scaler


def inverse_scale(y_scaled, scaler):

    mean = scaler["target_mean"]
    scale = scaler["target_scale"]

    return y_scaled * scale + mean


def main():

    model_path = Path("models/A2_Adam.keras")

    model = keras.models.load_model(model_path)

    X_test, y_test_scaled, y_test_raw, scaler = load_data()

    y_pred_scaled = model.predict(X_test).flatten()

    y_pred = inverse_scale(y_pred_scaled, scaler)

    mae = mean_absolute_error(y_test_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
    r2 = r2_score(y_test_raw, y_pred)

    print("\nEvaluación del mejor modelo (escala real):")

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    results = {
        "model": "A2_Adam",
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    results_path = Path("results/tables/final_metrics.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nResultados guardados en:")
    print(results_path)


if __name__ == "__main__":
    main()