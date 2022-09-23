# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:07:09 2021

@author: Gabriel
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import  GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

#%%criarRede

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    #camada de entrada (input_dim) + 1 camada oculta
    classificador.add(Dense(units = neurons, #round((dim_entrada + dim_saida)/2)
                            activation = activation, #costuma ser bom para deep_learning
                            kernel_initializer = kernel_initializer,#inicializador dos pesos
                            input_dim = 30)) #elementos na camada de entrada
    
    #dropout na primeira camada para evitar overfitting
    classificador.add(Dropout(0.2))
    
    #adicionando uma camada oculta
    classificador.add(Dense(units = neurons, #round((dim_entrada + dim_saida)/2)
                            activation = activation, #costuma ser bom para deep_learning
                            kernel_initializer = kernel_initializer,#inicializador dos pesos
                            )) 
    
    #dropout na camada oculta
    classificador.add(Dropout(0.2))
    #camada de saida
    classificador.add(Dense(units = 1,
                            activation = 'sigmoid')) #retorna um valor entre 0 e 1
        
    #compilar a rede
    classificador.compile(optimizer = optimizer, #ajuste dos pesos: maneira como é feita a descida do gradiente 
                          loss = loss, #funcao erro boa para classificação binária, crossentropy usa regressao logistica
                          metrics = ['binary_accuracy']) #metricas de performance
    return classificador

classificador = KerasClassifier(build_fn=criarRede)
#%%
parametros = {'batch_size': [10,30],
              'epochs': [50,100],
              'optimizer': ['adam', 'sgd'],
              'loss':['bynary_crossentropy', 'hinge'],
              'kernel_initializer':['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons':[16,8]}

grid_search = GridSearchCV(estimator = classificador,
                            param_grid = parametros, 
                            scoring= 'accuracy',
                            cv = 5) #numero de folds para executar a crossvalidation

grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_decisao = grid_search.best_score_