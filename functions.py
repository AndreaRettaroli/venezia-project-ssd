from datetime import datetime

import numpy as np
import pandas as pd
from keras.losses import mean_absolute_error
from matplotlib import pyplot as plt
from numpy.ma import asarray
from pandas import DataFrame, concat
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

from config import CONNECTION_STRING


# from series of values to windows matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])

    return np.array(dataX), np.array(dataY)

def get_range_by_date(data, start, finish):
    data["date"] = pd.to_datetime(data.date)
    mask = (data.date > start) & (data.date <= finish)
    return data.loc[mask]


def plot_original_data(date, data, rows, columns, i, color, label):
    plt.subplot(rows, columns, i)
    plt.plot(date, data, color=color, label=label)
    plt.legend()
    plt.show()


def autocorrelation(data):
    plot_acf(data, lags=40, alpha=0.05)
    plot_acf(data.diff().dropna(), lags=40, alpha=0.05)
    plt.show()


def compute_mean_and_variance(data):
    X = data
    split = round(len(X) / 2)
    X1, X2 = X[0:split], X[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()
    print('mean1=%f, mean2=%f' % (mean1, mean2))
    print('variance1=%f, variance2=%f' % (var1, var2))


def function_kpss(data):
    t_stat, p_value, _, critical_values = kpss(data, nlags='auto')

    print(f'ADF Statistic: {t_stat:.2f}')
    for key, value in critical_values.items():
        print('Critial Values:')
        print(f'{key}, {value:.2f}')

    print(f'\np-value: {p_value:.2f}')
    print("Stationary") if p_value > 0.05 else print("Non-Stationary")


def adf_fuller(data):
    t_stat, p_value, _, _, critical_values, _ = adfuller(data, autolag='AIC')
    print(f'ADF Statistic: {t_stat:.2f}')
    for key, value in critical_values.items():
        print('Critial Values:')
        print(f'{key}, {value:.2f}')

    print(f'\np-value: {p_value:.2f}')
    print("Non-Stationary") if p_value > 0.05 else print("Stationary")


def connect_to_db():
    return MongoClient(CONNECTION_STRING)


def convert_timestamp(df):
    date = []

    for t in df.timestamp:
        t = int(t)
        ts = int(t / 1000)
        time_value = datetime.fromtimestamp(ts)
        date.append(time_value)

    return date

    # transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[0]
        df = DataFrame(data)
        cols = list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
        # put it all together
        agg = concat(cols, axis=1)
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg.values

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
        return data[:-n_test], data[-n_test:]

# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        trainX, trainy = train[:, :-1], train[:, -1]
        # fit model
        model = RandomForestRegressor(n_estimators=1000)
        model.fit(trainX, trainy)
        # make a one-step prediction
        yhat = model.predict([testX])
        return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
        predictions = list()
        # split dataset
        train, test = train_test_split(data, n_test)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)):
            # split test row into input and output columns
            testX, testy = test[i, :-1], test[i, -1]
            # fit model on history and make a prediction
            yhat = random_forest_forecast(history, testX)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
            # summarize progress
            print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
        # estimate prediction error
        error = mean_absolute_error(test[:, -1], predictions)
        return error, test[:, -1], predictions

