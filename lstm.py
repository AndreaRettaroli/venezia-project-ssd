import numpy as np
from keras import Sequential, Input, Model
from keras.callbacks import LearningRateScheduler
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.losses import Huber, mean_absolute_error
from keras.optimizer_v1 import Adam
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def split_series(series, n_past, n_future):
    #
    # n_past ==> no of past observations
    #
    # n_future ==> no of future observations
    #
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end], series[past_end:future_end]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)


def lstm_forecast(df_floor, look_back):
    train_df, test_df = df_floor.totals[:-look_back], df_floor.totals[look_back:]

    train = train_df
    scalers = {}

    scaler = MinMaxScaler(feature_range=(-1, 1))
    s_s = scaler.fit_transform(train_df.values.reshape(-1, 1))
    s_s = np.reshape(s_s, len(s_s))
    scalers['scaler_'] = scaler
    train = s_s

    test = test_df
    scaler = scalers['scaler_']
    s_s = scaler.transform(test_df.values.reshape(-1, 1))
    s_s = np.reshape(s_s, len(s_s))
    scalers['scaler_'] = scaler
    test = s_s

    n_past = look_back
    n_future = look_back
    n_features = 1
    X_train, y_train = split_series(train, n_past, n_future)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
    X_test, y_test = split_series(test, n_past, n_future)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))

    """ Encoders """
    encoder_inputs = Input(shape=(n_past, n_features))
    encoder_l1 = LSTM(100, return_sequences=True, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]
    encoder_l2 = LSTM(100, return_state=True)
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]

    decoder_inputs = RepeatVector(n_future)(encoder_outputs2[0])

    """ Decoders """
    decoder_l1 = LSTM(100, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_l2 = LSTM(100, return_sequences=True)(decoder_l1, initial_state=encoder_states2)
    decoder_outputs2 = TimeDistributed(Dense(n_features))(decoder_l2)

    model = Model(encoder_inputs, decoder_outputs2)

    model.compile(optimizer="adam", loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), batch_size=32)

    predictions = model.predict(X_test)

    scaler = scalers['scaler_']
    prediction_1 = scaler.inverse_transform(predictions[:, :])
    # y_train[:, :] = scaler.inverse_transform(y_train[:, :])
    # y_test[:, :] = scaler.inverse_transform(y_test[:, :])
    plt.plot(df_floor.totals, label="Data")
    plt.plot(train, color="blue", label="Expdata")
    plt.plot([None for _ in train] + [x for x in prediction_1],color="green", label='Forecasting')
    plt.legend()

    plt.show()