from functions import *
def auto_arima(df_floor):
    model = pm.auto_arima(df_floor.totals.values, test='adf', start_p=1, start_q=1,
                           max_p=3, max_q=3, m=1, d=None, D=None,
                           seasonal=False, stationary=True, start_P=0, trace=True, stepwise=True)
    print(model.summary())
    fitted = model.fit(df_floor.totals)
    yfore = fitted.predict(n_periods=1000)  # forecast
    ypred = fitted.predict_in_sample()
    plt.plot(df_floor.totals, color="blue", label="Data")
    plt.plot(ypred, color="green", label="Prediction")
    plt.plot([None for i in ypred] + [x for x in yfore], color="red", label="Forecasting")
    plt.legend()
    plt.show()

    model.plot_diagnostics(figsize=(7, 5))
    plt.show()