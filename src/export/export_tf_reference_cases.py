import json
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras


def inverse_scale(y_scaled, scaler):
    return y_scaled * scaler["target_scale"] + scaler["target_mean"]


def main():
    X_test = pd.read_csv("data/processed/X_test.csv").values.astype(np.float32)

    with open("data/processed/target_scaler.json", "r", encoding="utf-8") as f:
        scaler = json.load(f)

    model = keras.models.load_model("models/A2_Adam.keras")
    y_pred_scaled = model.predict(X_test[:5], verbose=0).flatten()
    y_pred = inverse_scale(y_pred_scaled, scaler)

    lines = []
    lines.append("#ifndef TF_REFERENCE_CASES_H")
    lines.append("#define TF_REFERENCE_CASES_H\n")

    values = ", ".join(f"{float(v):.8f}f" for v in y_pred)
    lines.append("const float TF_REFERENCE_OUTPUTS[5] = {" + values + "};\n")

    lines.append("#endif")

    output_path = Path("arduino/amazon_mlp/tf_reference_cases.h")
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print("Archivo generado:")
    print("- arduino/amazon_mlp/tf_reference_cases.h")


if __name__ == "__main__":
    main()