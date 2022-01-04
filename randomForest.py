from functions import *


def random_forest(df_floor, look_back):
    values = df_floor.totals
    # transform the time series data into supervised learning
    train = series_to_supervised(values, n_in=look_back, n_out=1000)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = RandomForestRegressor(n_estimators=10)
    model.fit(trainX, trainy)
    # construct an input for a new prediction
    row = values[-look_back:]
    # make a one-step prediction
    yhat = model.predict(asarray([row]))
    print('Input: %s, Predicted: %.3f' % (row, yhat[0]))
