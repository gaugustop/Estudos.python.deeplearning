#%% imports
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense, Flatten, Dropout # atualizado: tensorflow==2.0.0-beta1
from tensorflow.python.keras.utils import np_utils # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Conv2D, MaxPooling2D # atualizado: tensorflow==2.0.0-beta1
from tensorflow.python.keras.layers.normalization import BatchNormalization # atualizado: tensorflow==2.0.0-beta1
#%% preprocessamento
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
plt.imshow(X_treinamento[0], cmap = 'gray')
plt.title('Classe ' + str(y_treinamento[0]))

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                               28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)
#%% estrutura da CNN
classificador = Sequential()
classificador.add(Conv2D(32, (3,3), 
                         input_shape=(28, 28, 1),
                         activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
#classificador.add(Flatten())

classificador.add(Conv2D(32, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, 
                        activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam', metrics = ['accuracy'])
#%%treinamento da CNN
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 128, epochs = 2,
                  validation_data = (previsores_teste, classe_teste))

resultado = classificador.evaluate(previsores_teste, classe_teste)

#%% ********** Previs??o de somente uma imagem **********

# Nesse exemplo escolhi a primeira imagem da base de teste e abaixo voc??
# pode visualizar que trata-se do n??mero 7
plt.imshow(X_teste[0], cmap = 'gray')
plt.title('Classe ' + str(y_teste[0]))

# Criamos uma ??nica vari??vel que armazenar?? a imagem a ser classificada e
# tamb??m fazemos a transforma????o na dimens??o para o tensorflow processar
imagem_teste = X_teste[0].reshape(1, 28, 28, 1)

# Convertermos para float para em seguida podermos aplicar a normaliza????o
imagem_teste = imagem_teste.astype('float32')
imagem_teste /= 255

# Fazemos a previs??o, passando como par??metro a imagem
# Como temos um problema multiclasse e a fun????o de ativa????o softmax, ser??
# gerada uma probabilidade para cada uma das classes. A vari??vel previs??o
# ter?? a dimens??o 1, 10 (uma linha e dez colunas), sendo que em cada coluna
# estar?? o valor de probabilidade de cada classe
previsoes = classificador.predict(imagem_teste)

# Como cada ??ndice do vetor representa um n??mero entre 0 e 9, basta agora
# buscarmos qual ?? o maior ??ndice e o retornarmos. Executando o c??digo abaixo
# voc?? ter?? o ??ndice 7 que representa a classe 7
import numpy as np
resultado = np.argmax(previsoes)

# Caso voc?? esteja trabalhando com a base CIFAR-10, voc?? precisar?? fazer
# um comando if para indicar cada uma das classes

