import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout


base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

#Criação da rede
classificador = Sequential()
classificador.add(Dense(units = 8, kernel_initializer='random_uniform',
                        activation = 'sigmoid', input_dim = 4))
classificador.add(Dropout(0.1))
classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer = 'adam', 
                      loss = 'categorical_crossentropy',
                      metrics = ['accuracy'])

#treinamento
classificador.fit(previsores, classe_dummy,
                  epochs = 2000, batch_size = 20)

#%%salvamento
open('classificador_iris.json', 'w').write(classificador.to_json())
classificador.save_weights('classificador_iris.h5')

#carregamento
arquivo = open('classificador_iris.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_iris.h5')

#nova previsão
novo = np.array([[5,3,1,1]])
previsao = classificador.predict(novo)
previsao = previsao > 0.5
