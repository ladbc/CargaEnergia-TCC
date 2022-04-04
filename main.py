import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

df_2019 = pd.read_csv("BaseDeDados/CARGA_ENERGIA_2019.csv", header=[0], sep=";")
df_2019_byRegion = df_2019.groupby(df_2019.loc[:,'id_subsistema'])
df_2019_set = df_2019_byRegion.get_group('SE').loc[:, 'din_instante':'val_cargaenergiamwmed']

df_2019_set.loc[:, 'din_instante'] = pd.to_datetime(df_2019_set.loc[:, 'din_instante']).astype(int)/ 10**9

sc = MinMaxScaler(feature_range=(0, 1))
training_set = sc.fit_transform(df_2019_set)
print(training_set)
print(training_set.shape)

x_train = []
y_train = []
for i in range(60, 365):
    x_train.append(training_set[i-60:i,1])
    y_train.append(training_set[i,1])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

rnn = Sequential()

rnn.add(LSTM(units= 50, return_sequences=True, input_shape=(x_train.shape[1],1)))
rnn.add(Dropout(0.3))
rnn.add(LSTM(units=50, return_sequences=True))
rnn.add(Dropout(0.3))
rnn.add(LSTM(units=50, return_sequences=True))
rnn.add(Dropout(0.3))
rnn.add(LSTM(units=50, return_sequences=True))
rnn.add(Dropout(0.3))
rnn.add(LSTM(units=50))
rnn.add(Dropout(0.2))
rnn.add(Dense(units=1))
rnn.compile(optimizer='adam', loss='mean_squared_error')

rnn.fit(x_train, y_train, epochs=100, batch_size=32)


df_2020 = pd.read_csv("BaseDeDados/CARGA_ENERGIA_2020.csv", header=[0], sep=";")
df_2020_byRegion = df_2020.groupby(df_2020.loc[:,'id_subsistema'])
df_2020_set = df_2020_byRegion.get_group('SE').loc[:, 'din_instante':'val_cargaenergiamwmed']