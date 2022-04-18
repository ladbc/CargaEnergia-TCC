import pandas as pd
from keras.layers import Dense, TimeDistributed, Bidirectional
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def lstm(trainData, validateData):
    x_train, y_train = lstm_formatting(trainData)
    x_validate, y_validate = lstm_formatting(validateData)
    prediction = train_lstm(x_train, y_train, x_validate, y_validate)
    plot_result(prediction, x_validate, y_validate)


def lstm_formatting(groups):
    group = groups[1]
    group = pd.DataFrame(group)
    print(group.head())
    group.loc[:, 'Data'] = pd.to_datetime(group.loc[:, 'Data']).astype(int) / 10 ** 9
    group = group.drop(columns=['ID', 'Subsistema'])

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set = sc.fit_transform(group)
    print(training_set)

    x_train = training_set
    y_train = training_set[:, 1]
    """for i in range(60, len(group.index)):
        x_train.append(training_set[i - 60:i])
        y_train.append(training_set[i])

        #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print(x_train)"""
    return x_train, y_train


def train_lstm(x_train, y_train, x_validate, y_validate):
    rnn = Sequential()
    rnn.add(LSTM(units=50, return_sequences=True, input_shape=(5, 1)))
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
    rnn.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['MeanSquaredError', 'MeanAbsoluteError'])

    """rnn = Sequential()
    rnn.add(input(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1))))
    rnn.add(Dropout(0.3))
    rnn.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    rnn.add(Dropout(0.3))
    rnn.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    rnn.add(Dropout(0.3))
    rnn.add(TimeDistributed(Dense(units=1)))
    rnn.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['MeanSquaredError', 'MeanAbsoluteError'])"""

    history = rnn.fit(x_train, y_train, epochs=500, batch_size=32)
    predictTrain = rnn.predict(x_train)
    predictTest = rnn.predict(x_validate)

    print("")
    print('-------------------- Model Summary --------------------')
    rnn.summary()

    print('-------------------- Evaluation on Training Data --------------------')
    for item in history.history:
        print("Final", item, ":", history.history[item][-1])
    print("")

    print('-------------------- Evaluation on Test Data --------------------')
    results = rnn.evaluate(x_validate, y_validate)

    return predictTest


def plot_result(predictTest, x_validate, y_validate):
    plt.plot(x_validate[:, 0], predictTest, label="Predicao")
    plt.plot(x_validate[:, 0], y_validate, label="Reais")
    plt.legend()
    plt.show()
