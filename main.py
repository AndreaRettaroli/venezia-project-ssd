import math
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pymongo import MongoClient
from statsmodels.graphics.tsaplots import plot_acf
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from keras.layers import Dense  # pip install tensorflow (as administrator)
from keras.models import Sequential  # pip install keras

from config import CONNECTION_STRING


# from series of values to windows matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

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


if __name__ == '__main__':
    """ Connect to DB """
    mongo_client = connect_to_db()

    """ Get database and collection """
    db_name = mongo_client.get_database("venezia-project")
    total_per_area_collection = db_name["totals-per-area"]
    db_sorted = mongo_client.get_database("venezia-sorted")
    total_per_area_sorted_collection = db_sorted["totals-per-area-sorted"]

    """ Find by area code"""
    cursors_floor_1 = total_per_area_collection.find({"area_code": "floor_1"})
    # cursors_floor_2 = total_per_area_collection.find({"area_code": "floor_2"})
    # cursors_floor_3 = total_per_area_collection.find({"area_code": "floor_3"})

    """ Convert to dataframe """
    df_floor1 = pd.DataFrame(list(cursors_floor_1))
    df_floor1["totals"] = df_floor1["totals"].astype("int32")

    # df_floor2 = pd.DataFrame(list(cursors_floor_2))
    # df_floor2["totals"] = df_floor2["totals"].astype("int32")
    # df_floor3 = pd.DataFrame(list(cursors_floor_3))
    # df_floor3["totals"] = df_floor3["totals"].astype("int32")

    """ Sorting by timestamp and getting each N data to compact time series """
    df_floor1 = df_floor1.sort_values(by="timestamp", ascending=True)
    df_floor1 = df_floor1.iloc[1::300, :]  # Every 10 minutes

    # df_floor2 = df_floor2.sort_values(by="timestamp", ascending=True)
    # # df_floor2 = df_floor2.iloc[1::300, :]

    # df_floor3 = df_floor3.sort_values(by="timestamp", ascending=True)
    # # df_floor3 = df_floor3.iloc[1::300, :]

    """ Convert timestamp to date and add replace 'date' to dataframe """
    date_val_floor_1 = convert_timestamp(df_floor1)
    df_floor1["date"] = date_val_floor_1
    df_floor1.drop(['_id', 'timestamp', 'area_code', 'alarms'], axis=1, inplace=True)  # Remove useless columns

    # date_val_floor_2 = convert_timestamp(df_floor2)
    # df_floor2["date"] = date_val_floor_2
    # df_floor2.drop(['_id', 'timestamp', 'area_code', 'alarms'], axis=1, inplace=True)  # Remove useless columns

    # date_val_floor_3 = convert_timestamp(df_floor3)
    # df_floor3["date"] = date_val_floor_3
    # df_floor3.drop(['_id', 'timestamp', 'area_code', 'alarms'], axis=1, inplace=True)  # Remove useless columns

    """ Get months range """
    # df_floor1 = get_range_by_date(df_floor1.totals, '2021-9-1', '2021-10-1')
    # df_floor2 = get_range_by_date(df_floor2.totals, '2021-9-1', '2021-10-1')
    # df_floor3 = get_range_by_date(df_floor3.totals, '2021-9-1', '2021-10-1')

    """ Redefining index """
    df_floor1.index = np.arange(0, len(df_floor1))
    # df_floor2.index = np.arange(0, len(df_floor2))
    # df_floor3.index = np.arange(0, len(df_floor3))

    """ Plot original data """
    figure(figsize=(15, 8))
    plot_original_data(df_floor1.date, df_floor1.totals, 2, 2, 1, "green", "Floor 1")

    # plot_original_data(df_floor2.date, df_floor2.totals, 2, 2, 2, "red", "Floor 2")
    # plot_original_data(df_floor3.date, df_floor3.totals, 2, 2, 3, "blue", "Floor 3")

    # plt.subplot(2, 2, 4)
    # plt.plot(df_floor1.date, df_floor1.totals, label="Floor 1")
    # plt.legend()
    #
    # plt.plot(df_floor2.date, df_floor2.totals, label="Floor 2")
    # plt.legend()
    #
    # plt.plot(df_floor3.date, df_floor3.totals, label="Floor 3")
    # plt.legend()

    """ Autocorrelation """
    autocorrelation(df_floor1.totals)
    # autocorrelation(df_floor2.totals)
    # autocorrelation(df_floor3.totals)

    """ ADF """
    adf_fuller(df_floor1.totals)
    # adf_fuller(df_floor2.totals)
    # adf_fuller(df_floor3.totals)

    """ KPPS """
    function_kpss(df_floor1.totals)
    # kpps(df_floor2.totals)
    # kpps(df_floor3.totals)

    """ Compute meaning and variance """
    compute_mean_and_variance(df_floor1.totals)
    # compute_mean_and_variance(df_floor2.totals)
    # compute_mean_and_variance(df_floor3.totals)

    """ Plot seasonal decompose of each area """
    # ds_floor_1 = df_floor1[df_floor1.columns[0]]
    # result = seasonal_decompose(ds_floor_1, model='additive', period=1)
    # result.plot()
    #
    # plt.show()
    #
    # # ds_floor_2 = ds_floor_2[ds_floor_2.columns[0]]
    # # result = seasonal_decompose(ds_floor_2, model='additive', period=1)
    # # plt.subplot(3, 1, 2)
    # # plt.plot(result, label="Seasonal decompose floor 2")
    # # plt.legend()
    # #
    # # ds_floor_3 = ds_floor_3[ds_floor_3.columns[0]]
    # # result = seasonal_decompose(ds_floor_3, model='additive', period=1)
    # # plt.subplot(3, 1, 3)
    # # plt.plot(result, label="Seasonal decompose floor 3")
    # # plt.legend()

    """ Forecasting with Statistical model """
    # model = pm.auto_arima(df_floor1.totals.values, test='adf', start_p=1, start_q=1,
    #                       max_p=3, max_q=3, m=1, d=None, D=None,
    #                       seasonal=False, stationary=True, start_P=0, trace=True, stepwise=True)
    #
    # print(model.summary())
    # fitted = model.fit(df_floor1.totals)
    # yfore = fitted.predict(n_periods=1000)  # forecast
    # ypred = fitted.predict_in_sample()
    # plt.plot(df_floor1.totals, color="blue", label="Data")
    # plt.plot(ypred, color="green", label="Prediction")
    # plt.plot([None for i in ypred] + [x for x in yfore], color="red", label="Forecasting")
    # plt.legend()
    # plt.show()
    #
    # model.plot_diagnostics(figsize=(7, 5))
    # plt.show()

    """ Forecasting with Machine Learning model """

    """ Forecasting with Neural model """
    look_back = 1008
    train_size = int(len(df_floor1.totals) - look_back)
    test_size = len(df_floor1.totals) - train_size
    train, test = df_floor1.totals[0:train_size, :], df_floor1.totals[train_size:len(df_floor1.totals), :]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Define model
    model = Sequential()
    model.add(Dense(32, input_dim=look_back, activation='relu'))  # 8 hidden neurons
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
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

    plt.plot(df_floor1.totals, label="data")
    plt.plot(np.concatenate((np.full(look_back - 1, np.nan), trainPredict[:, 0])), label="Prediction")
    plt.plot(np.concatenate((np.full(len(train) - 1, np.nan), testForecast[:, 0])), label="Forecasting")
    plt.legend()
    plt.show()