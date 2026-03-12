import json
import numpy as np
from pathlib import Path
from tensorflow import keras


def main():
    model_path = Path("models/A2_Adam.keras")
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path)

    weights = model.get_weights()

    export_data = {
        "architecture": "A2_Adam",
        "layers": []
    }

    npz_data = {}

    layer_index = 0
    dense_count = 0

    for layer in model.layers:
        if hasattr(layer, "get_weights"):
            layer_weights = layer.get_weights()
            if len(layer_weights) == 2:
                w, b = layer_weights
                dense_count += 1

                npz_data[f"W{dense_count}"] = w
                npz_data[f"b{dense_count}"] = b

                export_data["layers"].append({
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "weights_shape": list(w.shape),
                    "bias_shape": list(b.shape)
                })

                print(f"Capa {dense_count}:")
                print(f"  Pesos: {w.shape}")
                print(f"  Bias:  {b.shape}")

    np.savez(output_dir / "best_model_weights.npz", **npz_data)

    with open(output_dir / "best_model_structure.json", "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print("\nArchivos generados:")
    print("- models/best_model_weights.npz")
    print("- models/best_model_structure.json")


if __name__ == "__main__":
    main()