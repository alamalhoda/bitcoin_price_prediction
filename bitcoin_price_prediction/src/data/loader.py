"""
ماژول بارگذاری داده‌های بیت‌کوین
"""

import pandas as pd
import numpy as np

def load_data(file_path):
    """
    بارگذاری و پیش‌پردازش داده‌ها
    
    Args:
        file_path: مسیر فایل CSV
        
    Returns:
        DataFrame: داده‌های پیش‌پردازش شده
    """
    # خواندن فایل CSV با رد کردن دو سطر دوم و سوم (تیکر و ستون تاریخ خالی)
    df = pd.read_csv(file_path, skiprows=[1, 2])
    
    # تغییر نام ستون Price به Date
    df = df.rename(columns={'Price': 'Date'})
    
    # تبدیل ستون Date به datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # تنظیم Date به عنوان ایندکس
    df.set_index('Date', inplace=True)
    
    # تبدیل ستون‌ها به عددی
    numeric_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'Daily Return']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"داده‌ها با موفقیت بارگذاری شدند. شکل داده‌ها: {df.shape}")
    return df

def calculate_rsi(data, periods=14):
    """
    محاسبه شاخص قدرت نسبی (RSI)
    
    Args:
        data: DataFrame شامل قیمت‌های Close
        periods: دوره محاسبه RSI (پیش‌فرض: ۱۴)
    
    Returns:
        Series: مقادیر RSI محاسبه شده
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi 