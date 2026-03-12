import json
import numpy as np
import pandas as pd
from pathlib import Path


ARDUINO_DIR = Path("arduino/amazon_mlp")
WEIGHTS_PATH = Path("models/best_model_weights.npz")
SCALER_PATH = Path("data/processed/target_scaler.json")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TEST_RAW_PATH = Path("data/processed/y_test_raw.csv")


def format_array_1d(name: str, arr: np.ndarray, dtype: str = "float") -> str:
    values = ", ".join(f"{float(x):.8f}f" for x in arr.flatten())
    return f"const {dtype} {name}[{len(arr)}] = {{{values}}};\n"


def format_array_2d(name: str, arr: np.ndarray, dtype: str = "float") -> str:
    rows, cols = arr.shape
    lines = [f"const {dtype} {name}[{rows}][{cols}] = {{"]
    for row in arr:
        row_values = ", ".join(f"{float(x):.8f}f" for x in row)
        lines.append(f"    {{{row_values}}},")
    lines.append("};\n")
    return "\n".join(lines)


def generate_model_data(weights_data: dict) -> str:
    W1 = weights_data["W1"]
    b1 = weights_data["b1"]
    W2 = weights_data["W2"]
    b2 = weights_data["b2"]
    W3 = weights_data["W3"]
    b3 = weights_data["b3"]

    content = []
    content.append("#ifndef MODEL_DATA_H")
    content.append("#define MODEL_DATA_H\n")

    content.append("// Arquitectura: 47 -> 32 -> 16 -> 1")
    content.append("const int INPUT_SIZE = 47;")
    content.append("const int HIDDEN1_SIZE = 32;")
    content.append("const int HIDDEN2_SIZE = 16;")
    content.append("const int OUTPUT_SIZE = 1;\n")

    content.append(format_array_2d("W1", W1))
    content.append(format_array_1d("b1", b1))

    content.append(format_array_2d("W2", W2))
    content.append(format_array_1d("b2", b2))

    content.append(format_array_2d("W3", W3))
    content.append(format_array_1d("b3", b3))

    content.append("#endif")
    return "\n".join(content)


def generate_scaler_data(target_mean: float, target_scale: float) -> str:
    return f"""#ifndef SCALER_DATA_H
#define SCALER_DATA_H

const float TARGET_MEAN = {target_mean:.8f}f;
const float TARGET_SCALE = {target_scale:.8f}f;

#endif
"""


def generate_test_cases(X_test: np.ndarray, y_test_raw: np.ndarray, n_samples: int = 5) -> str:
    X_samples = X_test[:n_samples]
    y_samples = y_test_raw[:n_samples]

    lines = []
    lines.append("#ifndef TEST_CASES_H")
    lines.append("#define TEST_CASES_H\n")
    lines.append(f"const int TEST_SAMPLES = {n_samples};\n")

    rows, cols = X_samples.shape
    lines.append(f"const float TEST_INPUTS[{rows}][{cols}] = {{")
    for row in X_samples:
        row_values = ", ".join(f"{float(x):.8f}f" for x in row)
        lines.append(f"    {{{row_values}}},")
    lines.append("};\n")

    y_values = ", ".join(f"{float(y):.8f}f" for y in y_samples)
    lines.append(f"const float TEST_EXPECTED[{n_samples}] = {{{y_values}}};\n")

    lines.append("#endif")
    return "\n".join(lines)


def main():
    ARDUINO_DIR.mkdir(parents=True, exist_ok=True)

    weights_npz = np.load(WEIGHTS_PATH)
    weights_data = {k: weights_npz[k] for k in weights_npz.files}

    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        scaler = json.load(f)

    X_test = pd.read_csv(X_TEST_PATH).values.astype(np.float32)
    y_test_raw = pd.read_csv(Y_TEST_RAW_PATH).values.flatten().astype(np.float32)

    model_data = generate_model_data(weights_data)
    scaler_data = generate_scaler_data(
        scaler["target_mean"],
        scaler["target_scale"]
    )
    test_cases = generate_test_cases(X_test, y_test_raw, n_samples=5)

    (ARDUINO_DIR / "model_data.h").write_text(model_data, encoding="utf-8")
    (ARDUINO_DIR / "scaler_data.h").write_text(scaler_data, encoding="utf-8")
    (ARDUINO_DIR / "test_cases.h").write_text(test_cases, encoding="utf-8")

    print("Archivos generados:")
    print("- arduino/amazon_mlp/model_data.h")
    print("- arduino/amazon_mlp/scaler_data.h")
    print("- arduino/amazon_mlp/test_cases.h")


if __name__ == "__main__":
    main()