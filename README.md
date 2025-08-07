# Red_Keras_Clasificación

Este proyecto implementa una red neuronal secuencial utilizando TensorFlow y Keras para resolver un problema de clasificación binaria sobre el dataset de cáncer de mama de sklearn.

## Objetivo

Predecir si un tumor es maligno o benigno a partir de sus características físicas, utilizando un modelo de red neuronal con múltiples capas densas.

## Dataset

Se utiliza el dataset `breast_cancer` incluido en `sklearn.datasets`, que contiene 30 características numéricas de tumores y una variable objetivo binaria.

## Arquitectura del modelo

- Red neuronal secuencial (`tf.keras.Sequential`)
- Capas densas (`Dense`) con funciones de activación `selu`
- Capa de salida con activación `sigmoid` (para clasificación binaria)
- Semilla fija (`random.set_seed(42)`) para resultados reproducibles

## Capas del modelo

1. Capa de entrada (30 neuronas)
2. Capas ocultas: [2, 5, 10, 15, 20, 15, 10, 5] neuronas con activación `selu`
3. Capa de salida: 1 neurona con activación `sigmoid`

## Compilación y entrenamiento

- Función de pérdida: `mean_squared_error`
- Optimizador: `sgd`
- Métrica: `accuracy`
- Número de épocas: 100

## Resultados

Después del entrenamiento, el modelo alcanzó una precisión del **91,4%** (`accuracy: 0.9140`) y una pérdida de **0.0623**.

## Herramientas utilizadas

- Python
- TensorFlow / Keras
- Scikit-learn

## Código base

El archivo `Red_Keras_Clasificación.py` incluye todo el proceso: definición del modelo, partición de datos, entrenamiento y evaluación.