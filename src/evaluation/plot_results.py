import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Crear carpeta si no existe
Path("results/figures").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("results/tables/metrics_comparison.csv")

# ordenar por mejor MAE
df = df.sort_values("test_mae")

plt.figure(figsize=(10,6))
plt.bar(df["architecture"] + "_" + df["optimizer"], df["test_mae"])

plt.xticks(rotation=45)
plt.ylabel("Test MAE")
plt.title("Comparación de modelos (ordenados por desempeño)")
plt.tight_layout()

plt.savefig("results/figures/model_comparison.png", dpi=300)
plt.show()