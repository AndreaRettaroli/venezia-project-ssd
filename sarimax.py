import numpy as np
import pmdarima as pm
from matplotlib import pyplot as plt
from statsmodels.sandbox.regression.sympy_diff import df
from statsmodels.tsa.statespace.sarimax import SARIMAX


def auto_arima_for_sarimax(df_floor, forecast_to):# auto arima
    model = pm.auto_arima(df_floor.totals, start_p=1, start_q=1,
                            test='adf', max_p=2, max_q=2, m=12,
                            start_P=0, seasonal=True,
                            d=None, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True) # False full grid
    print(model.summary())
    morder = model.order
    print("Sarimax order {0}".format(morder))
    mseasorder = model.seasonal_order
    print("Sarimax seasonal order {0}".format(mseasorder))
    print("call sarimax")
    sarimax(df_floor,morder,mseasorder,forecast_to)


def sarimax(df_floor,morder,mseasorder, forecast_to):
    sarima_model = SARIMAX(df_floor.totals , order=morder, seasonal_order=mseasorder)
    sfit = sarima_model.fit()
    sfit.save('sarima_model.pkl')# save model
    sfit.plot_diagnostics()
    plt.show()
    ypred = sfit.predict(start=0, end=len(df_floor))
    yfore = sfit.forecast(steps=forecast_to)
    plt.plot(df_floor.totals, color="blue")
    plt.plot(ypred, color="yellow")
    plt.plot(yfore, color="green", label='Forecasting')
    plt.legend()
    plt.show()

