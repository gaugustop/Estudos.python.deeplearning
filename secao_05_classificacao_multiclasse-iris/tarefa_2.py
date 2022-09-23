import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede(optimizer, kernel_initializer, activation_func, 
               neurons, dropout):
    
    classificador = Sequential()
    classificador.add(Dense(units = neurons, kernel_initializer = kernel_initializer, 
                            activation = activation_func, input_dim = 4))
    classificador.add(Dropout(dropout))
    # classificador.add(Dense(units = neurons, kernel_initializer = kernel_initializer, 
    #                         activation = activation_func))
    # classificador.add(Dropout(dropout))
   
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = optimizer, 
                          loss = 'categorical_crossentropy',
                          metrics = ['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criar_rede,
                                epochs = 2000,
                                batch_size = 20)

#%%Procura dos melhores parametros (demora!!!)
parametros = {'optimizer': ['adam', 'sgd'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation_func': ['relu', 'sigmoid'],
              'neurons': [6,8],
              'dropout':[0.1,0.3]}


grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           cv = 5)
grid_search = grid_search.fit(previsores, classe_dummy)

#%% resultado

# melhores_parametros = grid_search.best_params_

melhores_parametros = {
                      'activation_func': 'sigmoid',
                      'dropout': 0.1,
                      'kernel_initializer': 'random_uniform',
                      'neurons': 8,
                      'optimizer': 'adam'
                      }
 
melhor_precisao = grid_search.best_score_
 # 0.9333333253860474


