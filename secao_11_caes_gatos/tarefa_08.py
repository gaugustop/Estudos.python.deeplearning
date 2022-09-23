# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:43:44 2022

@author: Gabriel
"""
#%%imports
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#%%leitura dos dados e divisao entre treinamento e teste
personagens = pd.read_csv('personagens.csv')
previsores = personagens.iloc[:,0:6].values
classes = personagens.iloc[:,6].values
labelencoder = LabelEncoder()
classes = labelencoder.fit_transform(classes) #bart:0 | homer:1
previsores_treinamento, previsores_teste, classes_treinamento, classes_teste = train_test_split(previsores, classes, test_size = 0.25)

#%%arquitetura da rede neural
classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 6))
classificador.add(Dropout(0.1))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dropout(0.1))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

#%%treinamento
classificador.fit(previsores_treinamento, classes_treinamento,
                  epochs = 200, batch_size = 10, 
                  validation_data = (previsores_teste,classes_teste))

#val_loss: 0.1482 - val_binary_accuracy: 0.9324