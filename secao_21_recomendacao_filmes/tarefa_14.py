# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 08:53:41 2022

@author: Gabriel
"""
#%%imports
import numpy as np
from rbm import RBM

#%% base de dados (excluindo o Leonardo)
base = np.array([[0,1,1,1,0,1],
                 [1,1,0,1,1,1],
                 [0,1,0,1,0,1],
                 [0,1,1,1,0,1], 
                 [1,1,0,1,0,1],
                 [1,1,0,1,1,1]])
filmes = ["Freddy x Jason", "O Ultimato Bourne", "Star Trek", 
          "Exterminador do Futuro", "Norbit", "Star Wars"]

#%%criação e treinamento da rede
rbm = RBM(num_visible = 6, num_hidden = 3)

rbm.train(base, max_epochs = 5000)

#%%recomendação
leonardo = np.array([[0,1,0,1,0,0]]) 
camada_escondida = rbm.run_visible(leonardo)
recomendacao = rbm.run_hidden(camada_escondida)
for i in range(len(leonardo[0])):
    if leonardo[0, i] == 0 and recomendacao[0, i] == 1:
        print(filmes[i])
        
#filme recomendado: Star Wars
