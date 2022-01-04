from matplotlib.pyplot import figure
from statsmodels.tsa.seasonal import seasonal_decompose

from diebold_mariano import *
from arima import auto_arima
from constants import *
from functions import *
from lstm_window import lstm_window
from mlp import mlp_forecast
from mlp_window import mlp_forecast_window
from sarimax import auto_arima_for_sarimax
from lstm import lstm_forecast

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
    df_floor1 = pd.DataFrame(list(cursors_floor_1))  # Timestamp every 2 seconds
    df_floor1["totals"] = df_floor1["totals"].astype("int32")

    # df_floor2 = pd.DataFrame(list(cursors_floor_2))  # Timestamp every 2 seconds
    # df_floor2["totals"] = df_floor2["totals"].astype("int32")

    # df_floor3 = pd.DataFrame(list(cursors_floor_3))  # Timestamp every 2 seconds
    # df_floor3["totals"] = df_floor3["totals"].astype("int32")

    """ Sorting by timestamp and getting each N data to compact time series """
    df_floor1 = df_floor1.sort_values(by="timestamp", ascending=True)
    df_floor1.index = np.arange(0, len(df_floor1))
    df_floor1 = df_floor1.iloc[1::300, :]  # Every 10 minutes

    # df_floor2 = df_floor2.sort_values(by="timestamp", ascending=True)
    # df_floor2.index = np.arange(0, len(df_floor2))
    # df_floor2 = df_floor2.iloc[1::300, :]  # Every 10 minutes

    # df_floor3 = df_floor3.sort_values(by="timestamp", ascending=True)
    # df_floor3.index = np.arange(0, len(df_floor3))
    # df_floor3 = df_floor3.iloc[1::300, :]

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
    # figure(figsize=(20, 13))
    # plot_original_data(df_floor1.date, df_floor1.totals, 2, 2, 1, "green", "Floor 1")
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
    # autocorrelation(df_floor1.totals)
    # autocorrelation(df_floor2.totals)
    # autocorrelation(df_floor3.totals)

    """ ADF """
    # adf_fuller(df_floor1.totals)
    # adf_fuller(df_floor2.totals)
    # adf_fuller(df_floor3.totals)

    """ KPPS """
    # function_kpss(df_floor1.totals)
    # function_kpss(df_floor2.totals)
    # function_kpss(df_floor3.totals)

    """ Compute meaning and variance """
    # compute_mean_and_variance(df_floor1.totals)
    # compute_mean_and_variance(df_floor2.totals)
    # compute_mean_and_variance(df_floor3.totals)

    """ Plot seasonal decompose of each area """
    # ds_floor_1 = df_floor1[df_floor1.columns[0]]
    # result = seasonal_decompose(ds_floor_1, model='additive', period=1)
    # result.plot()
    #
    # plt.show()

    # ds_floor_2 = df_floor2[df_floor2.columns[0]]
    # result = seasonal_decompose(ds_floor_2, model='additive', period=1)
    # result.plot()
    #
    # plt.show()
    #
    # ds_floor_3 = df_floor3[df_floor3.columns[0]]
    # result = seasonal_decompose(ds_floor_3, model='additive', period=1)
    # result.plot()
    #
    # plt.show()

    """ Forecasting with Statistical model """
    # auto_arima(df_floor1, MONTH_LOOK_BACK)
    # auto_arima(df_floor2, MONTH_LOOK_BACK)
    #auto_arima(df_floor3, MONTH_LOOK_BACK)

    # auto_arima_for_sarimax(df_floor1, WEEK_LOOK_BACK)
    # auto_arima_for_sarimax(df_floor2, WEEK_LOOK_BACK)
    # auto_arima_for_sarimax(df_floor3, WEEK_LOOK_BACK)
    """ Forecasting with Machine Learning model """
    # random_forest(df_floor1, WEEK_LOOK_BACK)

    """ Forecasting with Neural model """
    # MLP = mlp_forecast(df_floor1, WEEK_LOOK_BACK)
    # MLP = mlp_forecast(df_floor2, WEEK_LOOK_BACK)
    # MLP = mlp_forecast(df_floor3, WEEK_LOOK_BACK)
    # mlp_forecast_window(df_floor1, WEEK_LOOK_BACK)
    # mlp_forecast_window(df_floor3, WEEK_LOOK_BACK)

    # LSTM = lstm_forecast(df_floor1, WEEK_LOOK_BACK)
    # LSTM = lstm_forecast(df_floor2, WEEK_LOOK_BACK)
    # LSTM = lstm_forecast(df_floor3, WEEK_LOOK_BACK)

    lstm_window(df_floor1, WEEK_LOOK_BACK)


    #diebold mariano
    # actual=get_actual(df_floor1,WEEK_LOOK_BACK)

    # # Supponiamo che la differenza tra il primo elenco di previsione e i valori effettivi sia e1
    # # e il secondo elenco di previsione e il valore effettivo sia e2. La lunghezza delle serie temporali è T.
    # # Allora d può essere definito in base a un criterio diverso ( crit ).
    # # MSE : d = (e1)^2 - (e2)^2
    # # MAD : d = abs(e1) - abs(e2)
    # # MAPPE: d = abs((e1 - effettivo)/(effettivo))
    # # Poly: d = (e1)^potenza - (e2)^potenza
    # # L'ipotesi nulla è E[d] = 0.
    # # Le statistiche del test seguono la distribuzione T studente con grado di libertà (T - 1).

    # dm_test(actual, MLP, LSTM)# default crit = mse
