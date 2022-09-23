# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:06:04 2021

@author: Gabriel
"""
import pandas as pd
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense

from keras_visualizer import visualizer 

#%%
classificador = Sequential()
#camada de entrada (input_dim) + 1 camada oculta
classificador.add(Dense(units = round((30+1)/2), #round((dim_entrada + dim_saida)/2)
                        activation = 'relu', #costuma ser bom para deep_learning
                        kernel_initializer = 'random_uniform',#inicializador dos pesos
                        input_dim = 30)) #elementos na camada de entrada

#adicionando uma camada oculta
classificador.add(Dense(units = round((30+1)/2), #round((dim_entrada + dim_saida)/2)
                        activation = 'relu', #costuma ser bom para deep_learning
                        kernel_initializer = 'random_uniform',#inicializador dos pesos
                        )) 

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
#%%
#treinamento
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 10, #para ajuste dos pesos 
                  epochs = 100) #quantas epochs (leitura completa dos dados) para o treinamento

pesos0 = classificador.layers[0].get_weights()
print(len(pesos0)) #0: pesos, 1:bias
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

#avaliacao pelo sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz_confusao = confusion_matrix(classe_teste,previsoes)

#avaliacao pelo keras
resultado = classificador.evaluate(previsores_teste, classe_teste)


#%% una mierda! 
import tensorflow as tf
tf.keras.utils.plot_model(
    classificador,
    to_file="model.png",
    show_shapes=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)
