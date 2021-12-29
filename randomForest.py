from matplotlib import pyplot
from matplotlib import pyplot as plt
from functions import  *


def random_forest(df_floor,look_back):

    values = df_floor.totals
    # transform the time series data into supervised learning
    data = series_to_supervised(values, n_in=look_back,n_out=look_back)
    # evaluate
    mae, y, yhat = walk_forward_validation(data, 12)
    print('MAE: %.3f' % mae)
    # plot expected vs predicted
    pyplot.plot(y, label='Expected')
    pyplot.plot(yhat, label='Predicted')
    pyplot.legend()
    pyplot.show()
    # print("MSE={}".format(mse))
    # plt.plot(df_floor.totals, label="data")
    # plt.plot([None for _ in x_train] + [x for x in pred], color="green", label='Forecasting')
    # plt.legend()
    # plt.show()