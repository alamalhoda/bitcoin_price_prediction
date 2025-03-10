import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

def prepare_prophet_data(data):
    # دیباگ: نمایش چند خط اول دیتا
    print("First few rows of data:")
    print(data.head())
    
    # حذف هدرهای اضافی با skiprows (فرض می‌کنیم دو خط اول هدر اضافی هستن)
    data = data.drop(data.index[:2])  # حذف دو خط اول
    data.columns = data.iloc[0]  # ستون‌ها رو از خط سوم می‌گیریم
    data = data[1:].reset_index(drop=True)  # حذف خط سوم به عنوان هدر و ریست ایندکس
    
    # تبدیل ستون Date به ds و Close به y
    prophet_df = data.rename(columns={'Date': 'ds', 'Close': 'y'})
    return prophet_df[['ds', 'y']].astype({'ds': 'datetime64[ns]', 'y': 'float64'})

def predict_with_prophet(data, future_days=30):
    # آماده‌سازی دیتا برای Prophet
    prophet_df = prepare_prophet_data(data)
    
    # ساخت و آموزش مدل
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, changepoint_prior_scale=0.05)
    model.fit(prophet_df)
    
    # پیش‌بینی برای روزهای آینده
    future = model.make_future_dataframe(periods=future_days)
    forecast = model.predict(future)
    
    # نمایش نتایج
    print(f"Train MSE not applicable for Prophet (uses different evaluation)")
    print(f"Forecast available for {future_days} days")
    
    # رسم نمودار با Prophet
    fig = model.plot(forecast)
    plt.title('Bitcoin Price Prediction with Prophet')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend(['Actual/Trend', 'Uncertainty Interval'])
    plt.show()
    
    # نمودار تعاملی با Plotly
    from prophet.plot import plot_plotly
    plot_plotly(model, forecast)
    plt.show()
    
    return forecast
