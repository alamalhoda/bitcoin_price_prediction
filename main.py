import pandas as pd
from data_handler import fetch_cryptocompare_data, fetch_data, load_data_from_csv, save_data_to_csv
from plotting import plot_close_price, plot_volume, plot_moving_averages, plot_candlestick, plot_daily_returns
from linear_regression import predict_future_with_linear
from random_forest import predict_with_random_forest
from lstm_model import predict_with_lstm
from gru_model import predict_with_gru
from arima_model import predict_with_arima
from prophet_model import predict_with_prophet
from tcn_model import predict_with_tcn

def main():
    data = load_data_from_csv()
    if data is None:
        return

    # print(data.head())

    # رسم نمودارها
    # plot_close_price(data)
    # plot_volume(data)
    # plot_moving_averages(data)
    # plot_candlestick(data)
    # plot_daily_returns(data)

    # اجرای مدل‌ها
    # predict_future_with_linear(data)
    # predict_with_random_forest(data)
    # predict_with_lstm(data)
    # predict_with_gru(data)
    # predict_with_arima(data)
    # predict_with_tcn(data)

    # لود دیتا از فایل با تنظیم skiprows
    # data = pd.read_csv('/Users/alamalhoda/Projects/bitcoin_price_prediction/bitcoin_data.csv', skiprows=2)  # رد کردن دو خط اول
    # predict_with_prophet(data)

if __name__ == "__main__":
    main()