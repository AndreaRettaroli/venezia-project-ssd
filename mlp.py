import numpy as np
from keras import Sequential
from keras.layers import Dense
import math

from keras.models import load_model

from functions import *


def mlp_forecast(df_floor, look_back):
    # split into train and test sets
    train_size = int(len(df_floor.totals) - look_back)
    test_size = len(df_floor.totals) - train_size
    train, test = df_floor.totals[0:train_size], df_floor.totals[train_size:len(df_floor.totals)]

    testdata = np.concatenate((train[-look_back:], test))
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(testdata, look_back)

    print(testX.shape)
    # Define model
    # model = Sequential()
    # model.add(Dense(128, input_dim=look_back, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(1))  # 1 output neuron
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(trainX, trainY, epochs=500, verbose=2)

    model = load_model("models/mlp.h5")
    # Save model
    # model.save('models/mlp.h5')

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(trainScore,
                                                               math.sqrt(trainScore)))

    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(testScore, math.sqrt(testScore)))

    # generate predictions for training and forecast for plotting
    trainPredict = model.predict(trainX)
    testForecast = model.predict(testX)

    plt.plot(df_floor.totals, color="blue", label="data")
    plt.plot(np.concatenate((np.full(look_back - 1, np.nan), trainPredict[:, 0])), color="yellow", label="Prediction")
    plt.plot(np.concatenate((np.full(len(train) - 1, np.nan), testForecast[:, 0])), color="green", label="Forecasting")

    plt.legend()
    plt.show()

    return testForecast
