import json
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras


def relu(x):
    return np.maximum(0, x)


def load_weights():
    data = np.load("models/best_model_weights.npz")
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]
    W3 = data["W3"]
    b3 = data["b3"]
    return W1, b1, W2, b2, W3, b3


def numpy_forward(X, W1, b1, W2, b2, W3, b3):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, W3) + b3
    y = z3  # salida lineal

    return y


def inverse_scale(y_scaled, scaler):
    mean = scaler["target_mean"]
    scale = scaler["target_scale"]
    return y_scaled * scale + mean


def main():
    X_test = pd.read_csv("data/processed/X_test.csv").values.astype(np.float32)
    y_test_raw = pd.read_csv("data/processed/y_test_raw.csv").values.flatten()

    with open("data/processed/target_scaler.json", "r", encoding="utf-8") as f:
        scaler = json.load(f)

    W1, b1, W2, b2, W3, b3 = load_weights()

    # Emulación NumPy
    y_pred_scaled_numpy = numpy_forward(X_test, W1, b1, W2, b2, W3, b3).flatten()
    y_pred_numpy = inverse_scale(y_pred_scaled_numpy, scaler)

    # Predicción TensorFlow
    model = keras.models.load_model("models/A2_Adam.keras")
    y_pred_scaled_tf = model.predict(X_test, verbose=0).flatten()
    y_pred_tf = inverse_scale(y_pred_scaled_tf, scaler)

    diff_abs = np.abs(y_pred_tf - y_pred_numpy)

    print("Comparación TensorFlow vs Emulador NumPy")
    print(f"Diferencia absoluta máxima : {diff_abs.max():.10f}")
    print(f"Diferencia absoluta media  : {diff_abs.mean():.10f}")

    print("\nPrimeras 10 predicciones:")
    for i in range(10):
        print(
            f"[{i}] TF={y_pred_tf[i]:.6f} | NumPy={y_pred_numpy[i]:.6f} | Diff={diff_abs[i]:.10f}"
        )


if __name__ == "__main__":
    main()