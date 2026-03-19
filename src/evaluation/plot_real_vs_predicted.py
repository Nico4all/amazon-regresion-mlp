import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score

# crear carpeta de figuras
Path("results/figures").mkdir(parents=True, exist_ok=True)

# cargar modelo
model = tf.keras.models.load_model("models/A2_Adam.keras")

# cargar datos de test
X_test = pd.read_csv("data/processed/X_test.csv").values
y_test = pd.read_csv("data/processed/y_test_raw.csv").values.flatten()

# predecir
y_pred = model.predict(X_test).flatten()

# calcular R2
r2 = r2_score(y_test, y_pred)

# gráfica
plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred, alpha=0.3)

plt.xlabel("Valor real")
plt.ylabel("Predicción del modelo")
plt.title(f"Real vs Predicted (R² = {r2:.3f})")

# línea ideal
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.tight_layout()

plt.savefig("results/figures/real_vs_predicted.png", dpi=300)

print("Gráfica guardada en results/figures/real_vs_predicted.png")

plt.show()