import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def predict_with_arima(data):
    # فرض کنید داده‌های شما در یک DataFrame به نام 'df' با ستون 'Close' قرار دارند.
    # مثال: df = pd.read_csv('bitcoin_data.csv', index_col='Date', parse_dates=True)
    df = data.copy()
    # 1. ایستایی داده ها
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(df['Close'])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

    # 2. تفاضل گیری در صورت نیاز
    # مثال: df['Close_diff'] = df['Close'].diff().dropna()

    # 3. تقسیم داده ها
    train_size = int(len(df) * 0.8)
    train, test = df['Close'][:train_size], df['Close'][train_size:]

    # 4. انتخاب پارامترها (p, d, q)
    # مثال: با استفاده از Auto-ARIMA
    import pmdarima as pm
    model_autoARIMA = pm.auto_arima(train, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=5, max_q=5, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0,
                        D=0,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)
    print(model_autoARIMA.summary())
    p, d, q = model_autoARIMA.order

    # 5. آموزش مدل ARIMA
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()

    # 6. پیش‌بینی
    predictions = model_fit.predict(start=len(train), end=len(df) - 1, typ='levels')

    # 7. ارزیابی
    mse = mean_squared_error(test, predictions)
    print(f'Test MSE: {mse}')

    # 8. نمایش نتایج
    plt.figure(figsize=(12, 6))
    plt.plot(test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.show()