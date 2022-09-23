#%% imports
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import np_utils # atualizado: tensorflow==2.0.0-beta1
from tensorflow.python.keras.layers.normalization import BatchNormalization # atualizado: tensorflow==2.0.0-beta1
#%%carregamento
def carregar(file):
    with open(file,'rb') as fi:
        dicionario = pickle.load(fi, encoding = 'bytes')
        return dicionario
    
#primeiro dado de treinamento
file = os.path.join('cifar10','data_batch_1')
dicionario_cifar10 = carregar(file)
previsores_treinamento = dicionario_cifar10[b'data']
classe_treinamento = dicionario_cifar10[b'labels']

#demais dados de treinamento
for batch in range(2,6):
    file = os.path.join('cifar10',f'data_batch_{batch}')
    dicionario_cifar10 = carregar(file)
    previsores_treinamento = np.concatenate((previsores_treinamento,dicionario_cifar10[b'data']),  axis = 0)
    classe_treinamento     = np.concatenate((classe_treinamento,    dicionario_cifar10[b'labels']),axis = 0)

del batch

#dados de teste
file = os.path.join('cifar10','test_batch')
dicionario_cifar10 = carregar(file)
previsores_teste = dicionario_cifar10[b'data']
classe_teste = dicionario_cifar10[b'labels']

del dicionario_cifar10
del file

#%%preprocessamento
previsores_treinamento = previsores_treinamento.reshape(previsores_treinamento.shape[0], 32 , 32, 3) #n,x,y,rgb
previsores_teste = previsores_teste.reshape(previsores_teste.shape[0], 32 , 32, 3) 

previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(classe_treinamento,10)
classe_teste = np_utils.to_categorical(classe_teste,10)

#%%estrutura da CNN
classificador = Sequential()

#parte convolucional
classificador.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (32,32,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

#flatten
classificador.add(Flatten())

#parte densa
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.1))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.1))

#saida
classificador.add(Dense(units = 10, activation = 'softmax'))

#compilacao
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#%%treinamento
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 128, epochs = 4,
                  validation_data = (previsores_teste, classe_teste))

resultado = classificador.evaluate(previsores_teste, classe_teste)




