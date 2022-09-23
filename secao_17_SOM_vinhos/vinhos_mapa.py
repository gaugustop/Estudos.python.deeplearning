from minisom import MiniSom
import pandas as pd
import numpy as np
#%% load data
base = pd.read_csv('wines.csv')
X = base.iloc[:,1:14].values
y = base.iloc[:,0].values
y -= 1

#%%preprocessing
from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

#%%construção da rede
som = MiniSom(x = 8, y = 8, input_len = 13, sigma = 1.0, learning_rate = 0.5, random_seed = 2)

#%%inicialização e treinamento
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 10000)

# som._weights
# som._activation_map
# q = som.activation_response(X)
#%%plots
from matplotlib.pylab import pcolor, colorbar, plot
pcolor(som.distance_map())
# MID - mean inter neuron distance
colorbar()

# w = som.winner(X[2])
markers = ['o', 's', 'D']
color = ['r', 'g', 'b']


for i, x in enumerate(X):
    #print(i)
    #print(x)
    w = som.winner(x)
    #print(w)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = color[y[i]], markeredgewidth = 2)
