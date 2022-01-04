import numpy as np
from keras import Sequential
from keras.layers import Dense
import math

from functions import *


def mlp_forecast_window(df_floor, window_size):
    # split into train and test sets
    train_size = int(len(df_floor.totals) - window_size)
    test_size = len(df_floor.totals) - train_size
    train, test = df_floor.totals[0:train_size], df_floor.totals[train_size:len(df_floor.totals)]
    val=df_floor.totals[:]

    testdata = np.concatenate((train[-window_size:], test))
    trainX, trainY = create_dataset(train, window_size)
    testX, testY = create_dataset(testdata, window_size)
    valX,valY=create_dataset(val,window_size)

    print(testX.shape)
    # Define model
    model = Sequential()
    model.add(Dense(128, input_dim=window_size, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # 1 output neuron
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, verbose=2)

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(trainScore,
                                                               math.sqrt(trainScore)))

    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(testScore, math.sqrt(testScore)))

    # generate predictions for training and forecast for plotting
    trainPredict = model.predict(trainX)
    testForecast = model.predict(testX)

    mlp_predictions_scaled = list()
    mlp_predictions_scaled_1 = list()
    mlp_predictions_scaled_diag = list()
    batch = valX[-window_size*2:-window_size]
    curbatch = batch.reshape((window_size,window_size)) # 1 dim more

    # v1=curbatch
    for i in range(3024):

        #mlp_pred = model.predict(curbatch)[0]
        mlp_pred_p = model.predict(curbatch)
        mlp_predictions_scaled.append(mlp_pred_p[0])
        mlp_predictions_scaled_diag.append(mlp_pred_p[i % 1008])
        mlp_predictions_scaled_1.append(mlp_pred_p)
        curbatch = np.append(curbatch[:, 1:], mlp_pred_p, axis=1)




    yfore = np.transpose(mlp_predictions_scaled).squeeze()
    yfore1 = np.transpose(mlp_predictions_scaled_1).squeeze()
    ydiag = np.transpose(mlp_predictions_scaled_diag).squeeze()

    # Y1=mlp_predictions_scaled.squeeze()
    # Y2=np.transpose(mlp_predictions_scaled)

    #forcast_data = testForecast.transpose()
    #generated_forcast_data_X = np.tile(forcast_data, (window_size, 1))
    #test_forecast_on_generated = model.predict(generated_forcast_data_X)

    plt.plot(df_floor.totals, color="blue", label="data")
    plt.plot(np.concatenate((np.full(window_size - 1, np.nan), trainPredict[:, 0])), color="yellow", label="Prediction")

    plt.plot(np.concatenate((np.full(len(train) - 1, np.nan), testForecast[:, 0])), color="green", label="Forecasting")
    #plt.plot(np.concatenate((np.full(len(train) - 1, np.nan),  yfore[:])), color="red", label="Forecasting-window")

    #plt.plot(np.concatenate((np.full(len(train) - 1, np.nan), yfore1[0,:])), color="black", label="Forecasting-window")
    #diagonal = np.diagonal(yfore1)
    #plt.plot(np.concatenate((np.full(len(train) - 1, np.nan), diagonal[:])), color="purple", label="Forecasting-window")
    plt.plot(np.concatenate((np.full(len(train) - 1, np.nan), ydiag[:])), color="pink", label="Forecasting-window")
    # plt.plot(np.concatenate((np.full(len(train) - 1, np.nan), Y2[:, 0])), color="purple", label="Forecasting-window")
    #forecast_1 = np.concatenate((np.full(len(train) - 1, np.nan), testForecast[:, 0]))
    #plt.plot(np.concatenate((np.full(len(forecast_1) - 1, np.nan), test_forecast_on_generated[:, 0])), color="red",
    #         label="ForecastingF")
    plt.legend()
    plt.show()
