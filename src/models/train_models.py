import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import MODELS_DIR, RESULTS_DIR, RANDOM_STATE


INPUT_DIM = 47
EPOCHS = 30
BATCH_SIZE = 64


def set_seed(seed: int = RANDOM_STATE):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_training_data():
    base = Path("data/processed")

    X_train = pd.read_csv(base / "X_train.csv").values.astype("float32")
    X_val = pd.read_csv(base / "X_val.csv").values.astype("float32")
    X_test = pd.read_csv(base / "X_test.csv").values.astype("float32")

    y_train = pd.read_csv(base / "y_train.csv").values.astype("float32")
    y_val = pd.read_csv(base / "y_val.csv").values.astype("float32")
    y_test = pd.read_csv(base / "y_test.csv").values.astype("float32")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(architecture_name: str, optimizer_name: str, input_dim: int):
    model = keras.Sequential(name=f"{architecture_name}_{optimizer_name}")
    model.add(layers.Input(shape=(input_dim,)))

    if architecture_name == "A1":
        model.add(layers.Dense(16, activation="relu"))
    elif architecture_name == "A2":
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(16, activation="relu"))
    elif architecture_name == "A3":
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(16, activation="relu"))
    else:
        raise ValueError(f"Arquitectura no válida: {architecture_name}")

    model.add(layers.Dense(1, activation="linear"))

    if optimizer_name == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=0.01)
    elif optimizer_name == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer_name == "RMSprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
    else:
        raise ValueError(f"Optimizador no válido: {optimizer_name}")

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model


def train_one_model(architecture_name, optimizer_name, X_train, X_val, y_train, y_val):
    model = build_model(architecture_name, optimizer_name, INPUT_DIM)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )

    return model, history


def main():
    set_seed()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_training_data()

    input_dim = X_train.shape[1]
    print(f"Input dim detectado: {input_dim}")

    architectures = ["A1", "A2", "A3"]
    optimizers = ["SGD", "Adam", "RMSprop"]

    all_results = []

    for arch in architectures:
        for opt in optimizers:
            print(f"\nEntrenando modelo: {arch} + {opt}")

            model, history = train_one_model(
                arch, opt,
                X_train, X_val,
                y_train, y_val
            )

            train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
            val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
            test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

            model_filename = f"{arch}_{opt}.keras"
            history_filename = f"{arch}_{opt}_history.json"

            model.save(MODELS_DIR / model_filename)

            with open(MODELS_DIR / history_filename, "w", encoding="utf-8") as f:
                json.dump(history.history, f, ensure_ascii=False, indent=2)

            all_results.append({
                "architecture": arch,
                "optimizer": opt,
                "train_loss": float(train_loss),
                "train_mae": float(train_mae),
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
                "test_loss": float(test_loss),
                "test_mae": float(test_mae),
                "epochs_trained": len(history.history["loss"])
            })

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="val_loss", ascending=True)

    results_path = RESULTS_DIR / "tables" / "metrics_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\nResumen de modelos:")
    print(results_df)

    best_row = results_df.iloc[0]
    print("\nMejor modelo encontrado:")
    print(best_row)

    best_model_name = f"{best_row['architecture']}_{best_row['optimizer']}.keras"
    print(f"\nArchivo del mejor modelo: {best_model_name}")
    print(f"Métricas guardadas en: {results_path}")


if __name__ == "__main__":
    main()