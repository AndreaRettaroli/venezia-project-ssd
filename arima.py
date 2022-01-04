import pickle

import pmdarima as pm

from functions import *
from statsmodels.tsa.arima_model import ARIMAResults

def auto_arima(df_floor, periods):
    model = pm.auto_arima(df_floor.totals.values, test='adf', start_p=1, start_q=1,
                           max_p=3, max_q=3, m=1, d=None, D=None,
                           seasonal=False, stationary=True, start_P=0, trace=True, stepwise=True)
    print(model.summary())
    fitted = model.fit(df_floor.totals)

    # Load model
    # with open('models/arima.pkl', 'rb') as pkl:
    #     fitted = pickle.load(pkl)

    # Serialize with Pickle
    with open('models/arima.pkl', 'wb') as pkl:
        pickle.dump(fitted, pkl)

    yfore = fitted.predict(n_periods=periods)  # forecast
    ypred = fitted.predict_in_sample()
    plt.plot(df_floor.totals, color="blue", label="Data")
    plt.plot(ypred, color="green", label="Prediction")
    plt.plot([None for i in ypred] + [x for x in yfore], color="red", label="Forecasting")
    plt.legend()
    plt.show()

    model.plot_diagnostics(figsize=(7, 5))
    plt.show()