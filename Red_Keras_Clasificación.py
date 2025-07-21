from sklearn.datasets import load_breast_cancer #Importamos el dataset

cancer = load_breast_cancer() #Volcamos ese dataset a una variable

tf.random.set_seed(42) #Establecemos una semilla, esta es como un punto de partida para los números aleatorios.
#Tenemos la misma secuencia de números aleatorios cada vez que ejecutes tu código (42)

cancer1 = tf.keras.Sequential() #crea un modelo donde las capas están organizadas de forma secuencial
#Las capas se apilan una tras otra, y cada capa recibe la salida de la capa anterior


cancer1.add(tf.keras.layers.Input(shape=(30,))) #Para agregar las características del dataset (en este caso 30, a la capa de entrada)
#Normalmente esto se hace en la misma línea donde configuras las neuronas de la capa de entrada, pero lo he tenido que hacer por separado por el siguiente error:
"""/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/dense.py:87:
  UserWarning: Do not pass an input_shape/input_dim argument to a layer.
  When using Sequential models, prefer using an Input(shape) object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)"""

cancer1.add(tf.keras.layers.Dense(2, activation='selu'))#Capa de entrada, consta de dos neuronas, y función de activación "selu"
cancer1.add(tf.keras.layers.Dense(5, activation='selu')) #Capa escondida, de cinco neuronas, y función de activación "selu"
cancer1.add(tf.keras.layers.Dense(10, activation='selu'))#Capa escondida, de diez neuronas, y función de activación "selu"
cancer1.add(tf.keras.layers.Dense(15, activation='selu'))#Capa escondida, de quince neuronas, y función de activación "selu"
cancer1.add(tf.keras.layers.Dense(20, activation='selu'))#Capa escondida, de veinte neuronas, y función de activación "selu"
cancer1.add(tf.keras.layers.Dense(15, activation='selu'))#Capa escondida, de quince neuronas, y función de activación "selu"
cancer1.add(tf.keras.layers.Dense(10, activation='selu'))#Capa escondida, de diez neuronas, y función de activación "selu"
cancer1.add(tf.keras.layers.Dense(5, activation='selu'))#Capa escondida, de dos neuronas, y función de activación "selu"

cancer1.add(tf.keras.layers.Dense(1, activation="sigmoid"))#capa de salida, de una neurona, ya que es un problema binario (solo dos clases, maligno o benigno)
#La función "sigmoid" transforma el valor de la neurona de salida en un valor entre 0 y 1, que puede interpretarse como una probabilidad

cancer1.summary() #Mostrar un resumen del modelo (número de capas y neuronas)
tf.keras.utils.plot_model(cancer1, show_shapes = True)

#Compilación del modelo
cancer1.compile(loss="mean_squared_error", #Función de pérdida, este calcula qué tan bien o mal lo está haciendo el modelo al hacer predicciones
              optimizer="sgd", #Optimizador, ajusta los parámetros (pesos) del modelo para hacerlo cada vez más preciso
              metrics=["accuracy"]) #Métricas, son herramientas que nos ayudan a evaluar si el modelo está funcionando bien
                                    #Accuracy, mide el porcentaje de veces que el modelo acertó con su predicción

"Particionamiento y Entrenamiento"

#X_train/X_test, son los datos (características) para entrenar y probar el modelo.
#y_train/y_test, son las etiquetas (resultados) para entrenar y probar el modelo.
#train_test_split, función divide los datos en dos partes: una para entrenar el modelo y otra para probarlo.
#.data, características del dataset.
#.target, los resultados.
#test_size =0.33, el 33% de los datos se usará para probar el modelo, y el 67% restante se usará para entrenarlo.
#stratify = .target, que los resultados estén equilibrados en ambos conjuntos (entrenamiento y prueba).
#random_state, para establecer la semilla (42).

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,test_size=0.33,stratify=cancer.target,random_state=42)

#.fit, para entrenar el modelo.
#X_train, características del dataset.
#y_train, resultados que queremos predecir.
#epochs, los ciclos de entrenos que queremos que haga el modelo.

history3 = cancer1.fit(X_train, y_train, epochs=100)

#accuracy: 0.9140 - loss: 0.0623 - El modelo predijo con un 91,4% de aciertos