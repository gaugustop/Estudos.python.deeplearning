# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:22:46 2021

@author: Gabriel
"""
import pandas as pd
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import  GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

#===== colocar os resultador do teste feito em tunning ====
classificador = Sequential()
#camada de entrada (input_dim) + 1 camada oculta
classificador.add(Dense(units = 8, #round((dim_entrada + dim_saida)/2)
                        activation = 'relu', #costuma ser bom para deep_learning
                        kernel_initializer = 'normal',#inicializador dos pesos
                        input_dim = 30)) #elementos na camada de entrada

#dropout na primeira camada para evitar overfitting
classificador.add(Dropout(0.2))

#adicionando uma camada oculta
classificador.add(Dense(units = 8, #round((dim_entrada + dim_saida)/2)
        activation = 'relu', #costuma ser bom para deep_learning
        kernel_initializer = 'normal',#inicializador dos pesos
        )) 

#dropout na camada oculta
classificador.add(Dropout(0.2))
#camada de saida
classificador.add(Dense(units = 1,
        activation = 'sigmoid')) #retorna um valor entre 0 e 1

#compilar a rede
classificador.compile(optimizer = 'adam', #ajuste dos pesos: maneira como é feita a descida do gradiente 
                      loss = 'binary_crossentropy', #funcao erro boa para classificação binária, crossentropy usa regressao logistica
                      metrics = ['binary_accuracy']) #metricas de performance

classificador.fit(previsores, classe, batch_size = 10, epochs = 100)

#novo = np.array([[15.8, ...]])
novo = np.array(previsores.iloc[50])
novo = np.array([novo])

previsao = classificador.predict(novo)
