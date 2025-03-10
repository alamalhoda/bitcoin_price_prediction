from data_handler import load_data_from_csv
from plotting import plot_close_price, plot_volume, plot_moving_averages, plot_candlestick, plot_daily_returns
# from linear_regression import prepare_and_predict
# from random_forest import predict_with_random_forest
# from lstm_model import predict_with_lstm
# from gru_model import predict_with_gru
# from arima_model import predict_with_arima
from tcn_model import predict_with_tcn



def main():
    data = load_data_from_csv()
    if data is None:
        return

    # print(data.head())

    # رسم نمودارها
    plot_close_price(data)
    # plot_volume(data)
    # plot_moving_averages(data)
    # plot_candlestick(data)
    # plot_daily_returns(data)

    # اجرای مدل‌ها
    # prepare_and_predict(data)
    # predict_with_random_forest(data)
    # predict_with_lstm(data)
    # predict_with_gru(data)
    # predict_with_arima(data)
    predict_with_tcn(data)

if __name__ == "__main__":
    main()