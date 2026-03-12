# Amazon Regression MLP

Miniproyecto de regresión con TensorFlow/Keras usando el dataset Amazon Sale Report.

## Objetivo
Desarrollar modelos de regresión con redes neuronales multicapa superficiales para predecir una variable numérica del dataset y posteriormente emular la red en Arduino.

## Entorno
- Python 3.11
- TensorFlow 2.21.0
- CPU

## Estructura
- `data/raw/`: datos originales
- `data/processed/`: datos limpios y transformados
- `src/`: código fuente
- `models/`: modelos entrenados
- `results/`: métricas, tablas y gráficas
- `arduino/`: exportación para emulación

## Flujo
1. Carga y exploración de datos
2. Limpieza y preprocesamiento
3. Entrenamiento de 3 arquitecturas con 3 optimizadores
4. Evaluación
5. Exportación de pesos
6. Emulación en Arduino



## Variables finales del modelo
Qty
month
day
weekday
Status
Fulfilment
Sales Channel
ship-service-level
Category
Size
Courier Status
B2B
fulfilled-by

# Target
Amount