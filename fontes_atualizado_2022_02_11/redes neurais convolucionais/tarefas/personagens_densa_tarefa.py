import pandas as pd
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense # atualizado: tensorflow==2.0.0-beta1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('personagens.csv')
previsores = base.iloc[:, 0:6].values
classe = base.iloc[:, 6].values
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential([
                tf.keras.layers.Dense(units = 4, activation = 'relu', input_dim = 6),
                tf.keras.layers.Dense(units = 4, activation = 'relu'),
                tf.keras.layers.Dense(units = 1, activation = 'sigmoid')])
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                      metrics = ['binary_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, 
                  epochs = 2000)
resultado = classificador.evaluate(previsores_teste, classe_teste)