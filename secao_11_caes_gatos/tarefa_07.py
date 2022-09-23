# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:29:51 2022

@author: Gabriel
"""

#%% imports
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense, Flatten, Dropout # atualizado: tensorflow==2.0.0-beta1
from tensorflow.python.keras.utils import np_utils # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Conv2D, MaxPooling2D # atualizado: tensorflow==2.0.0-beta1
from tensorflow.python.keras.layers.normalization import BatchNormalization # atualizado: tensorflow==2.0.0-beta1
#%% preprocessamento
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
plt.imshow(X_treinamento[0], cmap = 'gray')
plt.title('Classe ' + str(y_treinamento[0]))

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                               28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)
#%% estrutura da CNN
classificador = Sequential()
classificador.add(Conv2D(32, (3,3), 
                         input_shape=(28, 28, 1),
                         activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
#classificador.add(Flatten())

classificador.add(Conv2D(32, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, 
                        activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam', metrics = ['accuracy'])
#%%treinamento da CNN
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 128, epochs = 2,
                  validation_data = (previsores_teste, classe_teste))

resultado = classificador.evaluate(previsores_teste, classe_teste)

#%%classificacao de uma entrada
#vamos pegar uma imagem quennão foi usada no treinamento, i.e., dentro da variável previsores_teste
imagem = previsores_teste[0]

#visualização
plt.imshow(imagem, cmap = 'inferno') #é um sete

entrada = imagem.reshape((1,28,28,1))

saida = classificador.predict(entrada)

print(f'o número desenhado é o {saida.argmax()}') #7
