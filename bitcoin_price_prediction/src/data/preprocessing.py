"""
ماژول پیش‌پردازش داده‌ها برای مدل‌های پیش‌بینی قیمت
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from bitcoin_price_prediction.src.data.loader import calculate_rsi
from bitcoin_price_prediction.config.base_config import LAG_DAYS, TRAIN_SPLIT_RATIO

def add_technical_indicators(data):
    """
    اضافه کردن شاخص‌های تکنیکال به داده‌ها
    
    Args:
        data: DataFrame داده‌های اصلی
    
    Returns:
        DataFrame: داده‌ها با شاخص‌های تکنیکال
    """
    df = data.copy()
    
    # محاسبه میانگین‌های متحرک
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # محاسبه تغییرات روزانه
    df['Daily_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # محاسبه RSI
    df['RSI'] = calculate_rsi(df)
    
    # محاسبه نوسانات
    df['Volatility'] = df['Close'].rolling(window=14).std()
    
    return df

def prepare_data_for_model(data, lag_days=LAG_DAYS, train_split_ratio=TRAIN_SPLIT_RATIO):
    """
    آماده‌سازی داده‌ها برای ورود به مدل
    
    Args:
        data: DataFrame داده‌های اصلی
        lag_days: تعداد روزهای گذشته برای استفاده در پیش‌بینی
        train_split_ratio: نسبت داده‌های آموزش
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler_X, scaler_y
    """
    # ایجاد ستون هدف (قیمت روز بعد)
    data['Target'] = data['Close'].shift(-1)
    
    # ایجاد ویژگی‌های تاخیری (lag features)
    for i in range(1, lag_days + 1):
        data[f'Close_Lag_{i}'] = data['Close'].shift(i)
    
    # افزودن شاخص‌های تکنیکال
    data = add_technical_indicators(data)
    
    # حذف ردیف‌های با مقدار NaN
    data = data.dropna()
    
    # محاسبه اندازه داده‌های آموزش
    train_size = int(len(data) * train_split_ratio)
    
    # نمایش تاریخ‌های شروع و پایان داده‌های آموزش و آزمون
    print(f"تاریخ شروع داده‌های آموزش: {data.index[0]}")
    print(f"تاریخ پایان داده‌های آموزش: {data.index[train_size-1]}")
    print(f"تاریخ شروع داده‌های آزمون: {data.index[train_size]}")
    print(f"تاریخ پایان داده‌های آزمون: {data.index[-1]}")
    
    # آماده‌سازی ویژگی‌های ورودی و متغیر هدف
    features = [f'Close_Lag_{i}' for i in range(1, lag_days + 1)] + ['Volume', 'SMA_50', 'SMA_200', 'Daily_Change', 'RSI', 'Volatility']
    X = data[features]
    y = data['Target']
    
    # تقسیم داده‌ها به مجموعه‌های آموزش و آزمون
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # نرمال‌سازی داده‌ها
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

def prepare_sequences_for_lstm(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, lag_days=LAG_DAYS):
    """
    آماده‌سازی توالی‌های ورودی برای مدل LSTM
    
    Args:
        X_train_scaled: داده‌های ورودی آموزش نرمال‌سازی شده
        X_test_scaled: داده‌های ورودی آزمون نرمال‌سازی شده
        y_train_scaled: داده‌های هدف آموزش نرمال‌سازی شده
        y_test_scaled: داده‌های هدف آزمون نرمال‌سازی شده
        lag_days: تعداد روزهای گذشته برای استفاده در پیش‌بینی
    
    Returns:
        tuple: X_train_seq, X_test_seq, y_train_seq, y_test_seq
    """
    # آماده‌سازی داده‌های آموزش
    X_train_seq = []
    y_train_seq = []
    for i in range(lag_days, len(X_train_scaled)):
        X_train_seq.append(X_train_scaled[i-lag_days:i])
        y_train_seq.append(y_train_scaled[i])
    
    # آماده‌سازی داده‌های آزمون
    X_test_seq = []
    y_test_seq = []
    for i in range(lag_days, len(X_test_scaled)):
        X_test_seq.append(X_test_scaled[i-lag_days:i])
        y_test_seq.append(y_test_scaled[i])
    
    # تبدیل به آرایه‌های numpy
    X_train_seq, y_train_seq = np.array(X_train_seq), np.array(y_train_seq)
    X_test_seq, y_test_seq = np.array(X_test_seq), np.array(y_test_seq)
    
    print(f"شکل داده‌های آموزش: X: {X_train_seq.shape}, y: {y_train_seq.shape}")
    print(f"شکل داده‌های آزمون: X: {X_test_seq.shape}, y: {y_test_seq.shape}")
    
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq

def create_exponential_weights(length, decay_factor=0.5):
    """
    ایجاد وزن‌های نمایی برای داده‌های اخیر
    
    Args:
        length: طول آرایه وزن‌ها
        decay_factor: ضریب کاهش وزن
    
    Returns:
        array: آرایه وزن‌ها
    """
    weights = np.exp(np.linspace(0, decay_factor, length))
    weights = weights / weights.mean()  # نرمال‌سازی وزن‌ها
    return weights 