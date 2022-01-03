import math

import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.losses import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from functions import create_dataset_2


def lstm_window(df_floor, window_size):
    np.random.seed(7)

    df = df_floor.set_index("date")
    dataset = df.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.80)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset_2(train, window_size)
    testX, testY = create_dataset_2(test, window_size)
    # reshape input to be [samples, time steps, features]

    scaler = MinMaxScaler()
    scaler.fit_transform(trainX.reshape(-1, 1))
    scaled_train_data = scaler.transform(trainX.reshape(-1, 1))
    print(scaled_train_data.shape)
    scaled_test_data = scaler.transform(testX.reshape(-1, 1))
    print(scaled_test_data.shape)
    from keras.preprocessing.sequence import TimeseriesGenerator
    n_input = window_size;
    n_features = 1
    generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input,
                                    batch_size=1)
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM


    model = Sequential()
    model.add(LSTM(10, input_shape=(n_input, 1 )))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=1, batch_size=1, verbose=1)
    model.summary()
    model.save('lstm_model')



    losses_lstm = model.history.history['loss']
    plt.xticks(np.arange(0, 21, 1))  # convergence trace
    plt.plot(range(len(losses_lstm)), losses_lstm);
    plt.show()
    # window
    lstm_predictions_scaled = list()
    batch = scaled_train_data[-window_size:]
    curbatch = batch.reshape((1, window_size, 1))  # 1 dim more
    for i in range(len(test) * 10):
        lstm_pred = model.predict(curbatch)[0]
        lstm_predictions_scaled.append(lstm_pred)
        curbatch = np.append(curbatch[:, 1:, :], [[lstm_pred]], axis=1)
    lstm_forecast = scaler.inverse_transform(lstm_predictions_scaled)
    yfore = np.transpose(lstm_forecast).squeeze()



    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset), color="blue", label="data")
    plt.plot(yfore, color="pink", label="forecasting_pred")
    plt.legend()
    plt.show()