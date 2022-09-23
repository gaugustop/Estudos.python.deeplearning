# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 16:09:07 2021

@author: Gabriel
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

#%%criarRede

def criarRede():
    classificador = Sequential()
    #camada de entrada (input_dim) + 1 camada oculta
    classificador.add(Dense(units = round((30+1)/2), #round((dim_entrada + dim_saida)/2)
                            activation = 'relu', #costuma ser bom para deep_learning
                            kernel_initializer = 'random_uniform',#inicializador dos pesos
                            input_dim = 30)) #elementos na camada de entrada
    #dropout na primeira camada para evitar overfitting
    classificador.add(Dropout(0.2))
    
    #adicionando uma camada oculta
    classificador.add(Dense(units = round((30+1)/2), #round((dim_entrada + dim_saida)/2)
                            activation = 'relu', #costuma ser bom para deep_learning
                            kernel_initializer = 'random_uniform',#inicializador dos pesos
                            )) 
    #dropout na camada oculta
    classificador.add(Dropout(0.2))
    #camada de saida
    classificador.add(Dense(units = 1,
                            activation = 'sigmoid')) #retorna um valor entre 0 e 1
    
    otimizador = keras.optimizers.Adam(lr = 0.001, #taxa de aprendizagem
                                       decay = 0.0001, #decaimento da taxa de aprendizagem
                                       clipvalue = 0.5) #min e max da taxa de aprendizagem
                                       
    
    #compilar a rede
    classificador.compile(optimizer = otimizador, #ajuste dos pesos: maneira como é feita a descida do gradiente 
                          loss = 'binary_crossentropy', #funcao erro boa para classificação binária, crossentropy usa regressao logistica
                          metrics = ['binary_accuracy']) #metricas de performance
    return classificador
#%%treinamento com cross validation
classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 100,
                                batch_size = 10)

resultados = cross_val_score(estimator = classificador,
                             X = previsores, #entrada
                             y = classe, #saida
                             cv = 10, #para validacao cruzada
                             scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()

