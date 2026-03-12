# Guía de instalación y ejecución del proyecto

Este documento describe el proceso completo para ejecutar el proyecto de **regresión con redes neuronales multicapa**, desde la preparación del entorno hasta la **emulación del modelo en C++ equivalente a Arduino**.

El flujo completo incluye:

1. Creación del entorno de trabajo  
2. Preparación del dataset  
3. Entrenamiento del modelo  
4. Evaluación del modelo  
5. Exportación del modelo  
6. Emulación del modelo en C++  

---

# 1. Requisitos previos

Antes de comenzar, el sistema debe contar con:

- **Python 3.11**
- **pip**
- **g++** (para compilar la emulación en C++)
- **Windows PowerShell**
- Dataset: `Amazon Sale Report.xlsx`

Estructura esperada del proyecto:

```
amazon-regresion-mlp/
│
├── data/
│   ├── raw/
│   │   └── Amazon Sale Report.xlsx
│   └── processed/
│
├── src/
├── models/
├── results/
└── arduino/
    └── amazon_mlp/
```

---

# 2. Crear el entorno virtual

## Para qué sirve

El entorno virtual permite instalar las dependencias del proyecto sin afectar otras instalaciones de Python del sistema.

## Comando

```powershell
py -3.11 -m venv .venv-tf311
```

Esto crea la carpeta:

```
.venv-tf311/
```

---

# 3. Activar el entorno virtual

## Para qué sirve

Activa el entorno virtual para que todos los paquetes instalados se utilicen dentro del proyecto.

## Comando

```powershell
.\.venv-tf311\Scripts\Activate.ps1
```

Cuando esté activo aparecerá algo así:

```
(.venv-tf311) PS C:\ruta\del\proyecto>
```

---

# 4. Instalar dependencias

## Para qué sirve

Instala las librerías necesarias para:

- procesamiento de datos
- entrenamiento del modelo
- evaluación
- exportación del modelo

## Comandos

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install tensorflow pandas numpy scikit-learn matplotlib openpyxl jupyter notebook tensorboard
```

---

# 5. Verificar el entorno

## Para qué sirve

Comprueba que todas las librerías fueron instaladas correctamente.

## Comando

```powershell
python check_env.py
```

## Qué muestra

El script imprime:

- versión de Python
- versión de TensorFlow
- versión de pandas
- versión de NumPy
- versión de scikit-learn

---

# 6. Cargar el dataset

## Para qué sirve

Verifica que el archivo Excel pueda ser leído correctamente.

También muestra información básica del dataset.

## Comando

```powershell
python -m src.data.load_data
```

## Información que muestra

- número de filas
- número de columnas
- nombres de columnas
- tipos de datos
- valores nulos

---

# 7. Limpieza del dataset

## Para qué sirve

Realiza el preprocesamiento inicial del dataset:

- eliminación de columnas innecesarias
- manejo de valores faltantes
- transformación de variables
- generación de variables temporales

## Comando

```powershell
python -m src.data.clean_data
```

## Archivo generado

```
data/processed/amazon_clean.csv
```

Este archivo contiene el dataset listo para modelado.

---

# 8. Preparar datos para entrenamiento

## Para qué sirve

Convierte los datos al formato necesario para entrenar el modelo.

Este paso realiza:

- codificación One-Hot de variables categóricas
- escalado de variables
- división en train / validation / test

## Comando

```powershell
python -m src.data.prepare_training_data
```

## Archivos generados

```
data/processed/X_train.csv
data/processed/X_val.csv
data/processed/X_test.csv

data/processed/y_train.csv
data/processed/y_val.csv
data/processed/y_test.csv

data/processed/y_train_raw.csv
data/processed/y_val_raw.csv
data/processed/y_test_raw.csv

data/processed/feature_columns.json
data/processed/target_scaler.json
```

---

# 9. Entrenar los modelos

## Para qué sirve

Entrena múltiples configuraciones de redes neuronales para encontrar el mejor modelo.

Se prueban:

- diferentes arquitecturas
- diferentes optimizadores

## Comando

```powershell
python -m src.models.train_models
```

## Qué genera

Modelos entrenados en:

```
models/
```

Y la tabla de comparación:

```
results/tables/metrics_comparison.csv
```

---

# 10. Evaluar el mejor modelo

## Para qué sirve

Evalúa el mejor modelo utilizando el conjunto de prueba.

Calcula métricas de desempeño.

## Comando

```powershell
python -m src.evaluation.evaluate_best_model
```

## Métricas calculadas

- MAE  
- RMSE  
- R²  

## Archivo generado

```
results/tables/final_metrics.json
```

---

# 11. Exportar el mejor modelo

## Para qué sirve

Extrae los parámetros del modelo entrenado:

- pesos
- sesgos
- estructura de la red

## Comando

```powershell
python -m src.export.export_best_model
```

## Archivos generados

```
models/best_model_weights.npz
models/best_model_structure.json
```

Estos archivos permiten reconstruir la red fuera de TensorFlow.

---

# 12. Emulación del modelo en NumPy

## Para qué sirve

Reconstruye la red neuronal utilizando solo **NumPy** para comprobar que el modelo puede ejecutarse sin TensorFlow.

## Comando

```powershell
python -m src.emulation.mlp_numpy_emulator
```

## Qué valida

Compara:

- salida de TensorFlow
- salida del modelo reconstruido

La diferencia debe ser muy pequeña.

---

# 13. Exportar el modelo para Arduino

## Para qué sirve

Convierte el modelo a formato compatible con C++/Arduino.

## Comandos

```powershell
python -m src.export.export_arduino_files
python -m src.export.export_tf_reference_cases
```

## Archivos generados

```
arduino/amazon_mlp/model_data.h
arduino/amazon_mlp/scaler_data.h
arduino/amazon_mlp/test_cases.h
arduino/amazon_mlp/tf_reference_cases.h
```

Estos archivos contienen:

- pesos del modelo
- sesgos
- parámetros de escalado
- casos de prueba

---

# 14. Compilar la emulación en C++

## Para qué sirve

Simula la ejecución del modelo en un entorno equivalente a Arduino.

## Ir a la carpeta

```powershell
cd .\arduino\amazon_mlp
```

## Compilar

```powershell
g++ amazon_mlp_sim.cpp -O2 -o mlp_sim.exe
```

Esto genera:

```
mlp_sim.exe
```

---

# 15. Ejecutar la emulación

## Para qué sirve

Ejecuta el modelo reconstruido en C++ y compara:

- predicción del modelo
- valor real
- salida original de TensorFlow

## Comando

```powershell
.\mlp_sim.exe
```

## Resultado esperado

La consola mostrará algo similar a:

```
Caso 1
Valor real dataset: ...
Referencia TensorFlow: ...
Prediccion Arduino: ...
Error vs valor real: ...
Error vs TensorFlow: ...
```

La diferencia con TensorFlow debe ser **muy cercana a cero**.

---

# Flujo completo resumido

```powershell
.\.venv-tf311\Scripts\Activate.ps1

python -m src.data.load_data
python -m src.data.clean_data
python -m src.data.prepare_training_data

python -m src.models.train_models
python -m src.evaluation.evaluate_best_model

python -m src.export.export_best_model
python -m src.emulation.mlp_numpy_emulator

python -m src.export.export_arduino_files
python -m src.export.export_tf_reference_cases

cd .\arduino\amazon_mlp
g++ amazon_mlp_sim.cpp -O2 -o mlp_sim.exe
.\mlp_sim.exe
```