from functions import *

def mlp_forecast(df_floor):
    # look_back = 1008 #1week
    look_back = 2016  # 2week
    # look_back = 4320 #month

    # split into train and test sets
    train_size = int(len(df_floor.totals) - look_back)
    test_size = len(df_floor.totals) - train_size
    train, test = df_floor.totals[0:train_size], df_floor.totals[train_size:len(df_floor.totals)]

    testdata = np.concatenate((train[-look_back:], test))
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(testdata, look_back)

    # Define model
    model = Sequential()
    model.add(Dense(128, input_dim=look_back, activation='relu'))  # 8 hidden neurons
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # 1 output neuron
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=500, verbose=2)

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(trainScore,
                                                               math.sqrt(trainScore)))

    testScore = model.evaluate(testX, testY, verbose=0)

    print('Test Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(testScore, math.sqrt(testScore)))
    # generate predictions for training and forecast for plotting
    trainPredict = model.predict(trainX)
    testForecast = model.predict(testX)

    plt.plot(df_floor.totals, label="data")
    plt.plot(np.concatenate((np.full(look_back - 1, np.nan), trainPredict[:, 0])), label="Prediction")
    plt.plot(np.concatenate((np.full(len(train) - 1, np.nan), testForecast[:, 0])), label="Forecasting")
    plt.legend()
    plt.show()
