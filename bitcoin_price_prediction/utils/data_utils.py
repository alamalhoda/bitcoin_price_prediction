"""
توابع کمکی برای کار با داده‌ها در پروژه پیش‌بینی قیمت بیت‌کوین
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import yfinance as yf

def download_bitcoin_data(start_date="2015-01-01", end_date=None, save_path="data/bitcoin_data.csv"):
    """
    دانلود داده‌های قیمت بیت‌کوین از Yahoo Finance
    
    Args:
        start_date (str): تاریخ شروع داده‌ها
        end_date (str): تاریخ پایان داده‌ها (پیش‌فرض: امروز)
        save_path (str): مسیر ذخیره‌سازی فایل CSV
        
    Returns:
        pd.DataFrame: داده‌های بیت‌کوین
    """
    # اگر تاریخ پایان مشخص نشده باشد، از امروز استفاده کن
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    print(f"دریافت داده‌های بیت‌کوین از {start_date} تا {end_date}...")
    
    # دانلود داده‌ها با استفاده از yfinance
    btc_data = yf.download("BTC-USD", start=start_date, end=end_date)
    
    # اطمینان از وجود دایرکتوری خروجی
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # ذخیره داده‌ها در فایل CSV
    btc_data.to_csv(save_path)
    
    print(f"داده‌های بیت‌کوین با موفقیت در مسیر {save_path} ذخیره شدند.")
    print(f"تعداد رکوردها: {len(btc_data)}")
    
    return btc_data

def load_data(data_path):
    """
    بارگذاری داده‌های بیت‌کوین از فایل CSV
    
    Args:
        data_path (str): مسیر فایل CSV
        
    Returns:
        pd.DataFrame: داده‌های بیت‌کوین
    """
    # بررسی وجود فایل
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"فایل داده در مسیر {data_path} یافت نشد!")
    
    # بارگذاری داده‌ها
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # چک کردن ستون‌های مورد نیاز
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"فایل داده باید شامل ستون‌های {required_columns} باشد!")
    
    return data

def prepare_data(data, target_column='Close', lag_days=30, train_split_ratio=0.8, 
                 exponential_weighting=False, weight_decay=0.5):
    """
    آماده‌سازی داده‌ها برای آموزش مدل LSTM
    
    Args:
        data (pd.DataFrame): داده‌های بیت‌کوین
        target_column (str): ستونی که باید پیش‌بینی شود
        lag_days (int): تعداد روزهای گذشته برای استفاده در پیش‌بینی
        train_split_ratio (float): نسبت داده‌های آموزش به کل داده‌ها
        exponential_weighting (bool): استفاده از وزن‌دهی نمایی
        weight_decay (float): ضریب کاهش وزن
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler)
    """
    # استخراج ستون هدف
    data_target = data[target_column].values.reshape(-1, 1)
    
    # نرمال‌سازی داده‌ها
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_target)
    
    # ایجاد داده‌های آموزش و تست
    X, y = [], []
    
    for i in range(lag_days, len(scaled_data)):
        X.append(scaled_data[i-lag_days:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # تغییر شکل داده‌ها برای LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # محاسبه اندیس جداکننده داده‌های آموزش و تست
    train_size = int(len(X) * train_split_ratio)
    
    # تقسیم داده‌ها به مجموعه‌های آموزش و تست
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # اعمال وزن‌دهی نمایی به داده‌های آموزش اگر درخواست شده باشد
    sample_weights = None
    if exponential_weighting:
        # ایجاد وزن‌های نمایی برای داده‌های قدیمی‌تر
        sample_weights = np.array([weight_decay ** i for i in range(len(y_train)-1, -1, -1)])
        # نرمال‌سازی وزن‌ها
        sample_weights = sample_weights / sample_weights.sum()
    
    return X_train, y_train, X_test, y_test, scaler, sample_weights

def prepare_future_data(data, target_column='Close', lag_days=30, future_days=30):
    """
    آماده‌سازی داده‌ها برای پیش‌بینی قیمت روزهای آینده
    
    Args:
        data (pd.DataFrame): داده‌های بیت‌کوین
        target_column (str): ستونی که باید پیش‌بینی شود
        lag_days (int): تعداد روزهای گذشته برای استفاده در پیش‌بینی
        future_days (int): تعداد روزهای آینده برای پیش‌بینی
        
    Returns:
        tuple: (future_data, future_dates, scaler)
    """
    # استخراج ستون هدف
    data_target = data[target_column].values.reshape(-1, 1)
    
    # نرمال‌سازی داده‌ها
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_target)
    
    # آماده‌سازی داده‌های ورودی برای پیش‌بینی روزهای آینده
    # استفاده از آخرین lag_days روز موجود
    future_input = scaled_data[-lag_days:].reshape(1, lag_days, 1)
    
    # ایجاد تاریخ‌های آینده برای پیش‌بینی
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
    
    return future_input, future_dates, scaler

def inverse_transform_predictions(predictions, scaler):
    """
    برگرداندن مقیاس پیش‌بینی‌ها به مقیاس اصلی
    
    Args:
        predictions (numpy.array): پیش‌بینی‌های انجام شده
        scaler (MinMaxScaler): مقیاس‌کننده استفاده شده
        
    Returns:
        numpy.array: پیش‌بینی‌ها در مقیاس اصلی
    """
    # تغییر شکل پیش‌بینی‌ها به دو بعدی برای برگرداندن مقیاس
    predictions = predictions.reshape(-1, 1)
    
    # برگرداندن مقیاس پیش‌بینی‌ها
    return scaler.inverse_transform(predictions) 