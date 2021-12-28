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

