# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:54:32 2022

@author: Gabriel
"""
import pandas as pd
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense, Dropout, Activation, Input # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Model # atualizado: tensorflow==2.0.0-beta1

#%%preprocessamento
base = pd.read_csv('games.csv')
base = base.drop(['NA_Sales','EU_Sales','JP_Sales','Other_Sales'], axis = 1)
base = base.drop('Developer', axis = 1)

base = base.dropna(axis = 0)
base = base.loc[base['Global_Sales'] > 1]

base['Name'].value_counts()
nome_jogos = base.Name
base = base.drop('Name', axis = 1)

previsores = base.iloc[:, [0,1,2,3,5,6,7,8,9]].values
venda_global = base.iloc[:, 4].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

#%%construcao da rede
camada_entrada = Input(shape=(99,))
camada_oculta1 = Dense(units = 50, activation = 'sigmoid')(camada_entrada)
camada_oculta2 = Dense(units = 50, activation = 'sigmoid')(camada_oculta1)
camada_saida1 = Dense(units = 1, activation = 'linear')(camada_oculta2)


regressor = Model(inputs = camada_entrada,
                  outputs = [camada_saida1])
regressor.compile(optimizer = 'adam',
                  loss = 'mse')

#%%fit
regressor.fit(previsores, [venda_global],
              epochs = 5000, batch_size = 100)

#%%previsao
previsao_global = regressor.predict(previsores)