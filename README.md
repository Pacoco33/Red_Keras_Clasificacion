# Red_Keras_Clasificacion
Red neuronal con Keras

Creación de un pequeño modelo que intenta solucionar un problema de clasificación.

Estamos utilizando el dataset load_breast_cancer que contiene información sobre tumores de mama, y el objetivo es clasificar si un tumor es:

0 = maligno

1 = benigno

Es un problema de clasificación binaria ya que solo hay dos clases posibles (0 o 1).

La capa de salida tiene una sola neurona con activación "sigmoid" ya que esto es típico para problemas binarios (clasificación).

Se entrena con "X_train" y "y_train" donde "y_train" contiene valores discretos (0 o 1).

