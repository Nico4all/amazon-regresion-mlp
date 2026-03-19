import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

Path("results/figures").mkdir(parents=True, exist_ok=True)

# cargar modelo
model = tf.keras.models.load_model("models/A2_Adam.keras")

# cargar datos
X_test = pd.read_csv("data/processed/X_test.csv").values
y_test = pd.read_csv("data/processed/y_test_raw.csv").values.flatten()

# predecir
y_pred = model.predict(X_test).flatten()

# error absoluto
error = np.abs(y_test - y_pred)

plt.figure(figsize=(8,5))

plt.hist(error, bins=50)

plt.xlabel("Error absoluto")
plt.ylabel("Frecuencia")
plt.title("Distribución del error de predicción")

plt.tight_layout()

plt.savefig("results/figures/error_distribution.png", dpi=300)

print("Gráfica guardada en results/figures/error_distribution.png")

plt.show()