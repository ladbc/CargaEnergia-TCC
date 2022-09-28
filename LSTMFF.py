import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from keras.layers import Dense, Bidirectional
from keras.layers import Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def lstmff(trainDataGroups, validateDataGroups, testDataGroups):
    future_range = 7
    past_range = 28
    sns.color_palette("mako", as_cmap=True)
    sns.set()

    for i in range(0, 4):
        trainData = trainDataGroups[i]
        validateData = validateDataGroups[i]
        testData = testDataGroups[i]

        x_train, y_train = lstm_formatting(trainData, future_range, past_range)

        x_validate, y_validate = lstm_formatting(validateData, future_range, past_range)
        x_test, y_test = lstm_formatting(testData, future_range, past_range)
        scale = MinMaxScaler(feature_range=(0, 1))

        train_lstm(
            x_train,
            y_train,
            x_validate,
            y_validate,
            future_range,
            scale,
            y_test,
            x_test)

    return


def lstm_formatting(group, future_range, past_range):
    group = pd.DataFrame(group)
    print(group.head())
    group.loc[:, 'Data'] = pd.to_datetime(group.loc[:, 'Data']).astype(int) / 10 ** 9
    group = group.drop(columns=['ID', 'Subsistema', 'Dia', 'Ano', 'Data'])
    nrows = group.shape[0]
    ncols = group.shape[1]

    X_Scaler = MinMaxScaler(feature_range=(0, 1))
    Y_Scaler = MinMaxScaler(feature_range=(0, 1))
    training_set = X_Scaler.fit_transform(group)
    training_set_y = Y_Scaler.fit_transform(group[['Carga_WMed']])

    x_train, y_train = [], []
    for i in range(past_range, nrows - future_range):
        x_train.append(training_set[i - past_range:i, 0:ncols])
        y_train.append(training_set_y[i + 1:i + future_range + 1])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print('x_train = {}'.format(x_train.shape))
    print('y_train = {}'.format(y_train.shape))

    return x_train, y_train


def train_lstm(x_train, y_train, x_validate, y_validate, future_range, y_scale, y_test, x_test):
    batch_size = 256
    buffer_size = 150
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
    val_data = tf.data.Dataset.from_tensor_slices((x_validate, y_validate))
    val_data = val_data.batch(batch_size).repeat()
    lstm_model = tf.keras.models.Sequential([
        LSTM(200, return_sequences=True, input_shape=x_train.shape[-2:]),
        LSTM(200),
        Dense(40, activation='tanh'),
        Dense(20, activation='tanh'),
        Dropout(0.25),
        Dense(units=future_range),
    ])
    lstm_model.compile(optimizer='Adam', loss='mse')
    lstm_model.summary()

    model_path = 'Models/FeedFoward_LSTM_Multivariate.h5'
    early_stopings = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1,
        mode='min')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=0)

    callbacks = [early_stopings, checkpoint]

    history = lstm_model.fit(
        train_data,
        epochs=150,
        steps_per_epoch=80,
        validation_data=val_data,
        validation_steps=30,
        verbose=1,
        callbacks=callbacks)

    plt.figure(figsize=(16, 9))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Erro modelo')
    plt.ylabel('Erro')
    plt.xlabel('Época')
    plt.legend(['Erro treino', 'Erro validacao'])
    plt.show()

    pred = lstm_model.predict(x_test)
    print(pred)
    print(y_test)

    pred_inverse = y_scale.inverse_transform(pred)
    y_test = np.squeeze(y_test, axis=2)
    test_result = y_scale.inverse_transform(y_test)

    timeseries_evaluation_metrics_func(y_test, pred)
    return test_result


def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    f = open("ResultsTXT/resultados.txt", "a")
    f.write('Metricas de Avaliacao:-')
    f.write(f'MSE de : {tf.metrics.mean_squared_error(y_true, y_pred)}')
    f.write('\n')
    f.write(f'MSE final de : {np.mean(tf.metrics.mean_squared_error(y_true, y_pred))}')
    f.write('\n')
    f.write(f'MAE de : {tf.metrics.mean_absolute_error(y_true, y_pred)}')
    f.write('\n')
    f.write(f'MAE final de : {np.mean(tf.metrics.mean_absolute_error(y_true, y_pred))}')
    f.write('\n')
    f.write(f'RMSE de : {np.sqrt(tf.metrics.mean_squared_error(y_true, y_pred))}')
    f.write('\n')
    f.write(f'RMSE final de : {np.mean(np.sqrt(tf.metrics.mean_squared_error(y_true, y_pred)))}')
    f.write('\n')
    f.write(f'MAPE de : {mean_absolute_percentage_error(y_true, y_pred)}')
    f.write('\n')
    f.close()
