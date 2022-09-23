import pandas as pd
import pandas as pd
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense, Dropout # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier # atualizado: tensorflow==2.0.0-beta1
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import backend as k

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

def criar_rede(optimizer, kernel_initializer, activation, neurons, dropout):
    k.clear_session()
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer, input_dim = 4))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = optimizer, 
                          loss = 'sparse_categorical_crossentropy', 
                          metrics = ['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criar_rede)
parametros = {'batch_size': [10, 30],
              'epochs': [2000, 3000],
              'dropout': [0.2, 0.3],
              'optimizer': ['adam', 'sgd'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh', 'sigmoid'],
              'neurons': [4, 8, 16]}
grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,                           
                           cv = 2)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_