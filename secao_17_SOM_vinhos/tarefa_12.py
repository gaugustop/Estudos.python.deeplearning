# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 12:06:01 2022

@author: Gabriel
"""
#%%imports
from minisom import MiniSom
import pandas as pd
import numpy as np
#%% load data
X = pd.read_csv('entradas_breast.csv').values
y = pd.read_csv('saidas_breast.csv').values
y = y.reshape(y.shape[0])

#%%preprocessing
from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

#%%construção da rede
dim_som = int(np.sqrt(5*np.sqrt(X.shape[0]))) #10
som = MiniSom(x = dim_som, y = dim_som, 
              input_len = X.shape[1], 
              sigma = 2.5, 
              learning_rate = 0.25, 
              random_seed = 1)

#%%inicialização e treinamento
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 2000)

weights = som._weights
# som._activation_map
# q = som.activation_response(X)
#%%plots
from matplotlib.pylab import pcolor, colorbar, plot
pcolor(som.distance_map())
# MID - mean inter neuron distance
colorbar()

# w = som.winner(X[2])
markers = ['o', 'D']
color = ['g', 'r']


for i, x in enumerate(X):
    #print(i)
    #print(x)
    w = som.winner(x)
    #print(w)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 5,
         markeredgecolor = color[y[i]], markeredgewidth = 2)

