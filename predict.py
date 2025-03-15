"""
اسکریپت پیش‌بینی با استفاده از مدل ذخیره شده
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_saved_model(model_dir):
    """بارگذاری مدل و اسکالرها از مسیر ذخیره شده"""
    print(f"\nبارگذاری مدل از مسیر {model_dir}...")
    model = load_model(f"{model_dir}/lstm_model.h5")
    scaler_X = joblib.load(f"{model_dir}/scaler_X.pkl")
    scaler_y = joblib.load(f"{model_dir}/scaler_y.pkl")
    return model, scaler_X, scaler_y

def prepare_last_sequence(data, scaler_X, lag_days=24):
    """آماده‌سازی آخرین دنباله داده برای پیش‌بینی"""
    # انتخاب آخرین lag_days ردیف
    last_sequence = data.iloc[-lag_days:].copy()
    
    # اضافه کردن ویژگی‌های زمانی
    last_sequence['Hour'] = last_sequence.index.hour
    last_sequence['Day_of_Week'] = last_sequence.index.dayofweek
    last_sequence['Month'] = last_sequence.index.month
    last_sequence['Is_Weekend'] = last_sequence.index.dayofweek.isin([5, 6]).astype(int)
    
    # محاسبه ویژگی‌های قیمت
    last_sequence['Hourly_Return'] = last_sequence['Close'].pct_change()
    last_sequence['Daily_Return'] = last_sequence['Close'].pct_change(periods=24)
    
    # محاسبه میانگین‌های متحرک
    last_sequence['SMA_24'] = last_sequence['Close'].rolling(window=24).mean()
    last_sequence['SMA_50'] = last_sequence['Close'].rolling(window=50).mean()
    last_sequence['SMA_200'] = last_sequence['Close'].rolling(window=200).mean()
    
    # محاسبه RSI
    delta = last_sequence['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    last_sequence['RSI'] = 100 - (100 / (1 + rs))
    
    # محاسبه نوسانات
    last_sequence['Volatility_12h'] = last_sequence['Hourly_Return'].rolling(window=12).std()
    last_sequence['Volatility_24h'] = last_sequence['Hourly_Return'].rolling(window=24).std()
    
    # محاسبه شکاف قیمت
    last_sequence['Price_Gap'] = (last_sequence['High'] - last_sequence['Low']) / last_sequence['Close']
    
    # محاسبه نسبت حجم
    last_sequence['Volume_Ratio'] = last_sequence['Volume'] / last_sequence['Volume'].rolling(window=24).mean()
    
    # محاسبه محدوده قیمت
    last_sequence['Price_Range'] = (last_sequence['High'] - last_sequence['Low']) / last_sequence['Close']
    
    # حذف ستون‌های اضافی
    columns_to_drop = ['Open', 'High', 'Low', 'Volume']
    last_sequence = last_sequence.drop(columns=columns_to_drop, errors='ignore')
    
    # نرمال‌سازی داده‌ها
    last_sequence_scaled = scaler_X.transform(last_sequence)
    return last_sequence_scaled

def predict_next_hours(model, last_sequence, scaler_X, scaler_y, hours=24):
    """پیش‌بینی قیمت‌ها برای ساعت‌های آینده"""
    print(f"\nپیش‌بینی قیمت‌ها برای {hours} ساعت آینده...")
    
    # تبدیل معکوس آخرین دنباله به مقیاس اصلی
    last_sequence_original = scaler_X.inverse_transform(last_sequence)
    
    predictions = []
    dates = []
    current_sequence = last_sequence.copy()
    
    # پیش‌بینی برای هر ساعت
    for i in range(hours):
        # پیش‌بینی قیمت بعدی
        next_price = model.predict(current_sequence.reshape(1, 24, -1))
        next_price_original = scaler_y.inverse_transform(next_price)
        
        # به‌روزرسانی دنباله با داده‌های جدید
        new_row = current_sequence[-1].copy()
        new_row[0] = next_price[0][0]  # قیمت پیش‌بینی شده
        
        # به‌روزرسانی ویژگی‌ها
        new_row_original = scaler_X.inverse_transform(new_row.reshape(1, -1))
        new_row_original = new_row_original[0]
        
        # محاسبه ویژگی‌های جدید
        new_row[1] = (new_row_original[0] - last_sequence_original[-1][0]) / last_sequence_original[-1][0]  # Hourly_Return
        new_row[2] = (new_row_original[0] - last_sequence_original[-24][0]) / last_sequence_original[-24][0]  # Daily_Return
        
        # به‌روزرسانی میانگین‌های متحرک
        new_row[3] = np.mean([last_sequence_original[-23:, 0], new_row_original[0]])  # SMA_24
        new_row[4] = np.mean([last_sequence_original[-49:, 0], new_row_original[0]])  # SMA_50
        new_row[5] = np.mean([last_sequence_original[-199:, 0], new_row_original[0]])  # SMA_200
        
        # به‌روزرسانی RSI
        delta = new_row_original[0] - last_sequence_original[-1][0]
        gain = np.mean([last_sequence_original[-13:, 1], max(delta, 0)])
        loss = np.mean([last_sequence_original[-13:, 2], max(-delta, 0)])
        rs = gain / loss if loss != 0 else 0
        new_row[6] = 100 - (100 / (1 + rs))
        
        # به‌روزرسانی نوسانات
        new_row[7] = np.std([last_sequence_original[-11:, 1], new_row[1]])  # Volatility_12h
        new_row[8] = np.std([last_sequence_original[-23:, 1], new_row[1]])  # Volatility_24h
        
        # به‌روزرسانی شکاف قیمت
        new_row[9] = (new_row_original[0] * 1.001 - new_row_original[0] * 0.999) / new_row_original[0]  # Price_Gap
        
        # به‌روزرسانی نسبت حجم
        new_row[10] = 1.0  # Volume_Ratio (فرضی)
        
        # به‌روزرسانی محدوده قیمت
        new_row[11] = (new_row_original[0] * 1.001 - new_row_original[0] * 0.999) / new_row_original[0]  # Price_Range
        
        # اضافه کردن ردیف جدید به دنباله
        current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # ذخیره پیش‌بینی
        predictions.append(next_price_original[0][0])
        dates.append(last_sequence_original[-1][0] + timedelta(hours=i+1))
    
    return predictions, dates

def plot_predictions(predictions, dates):
    """رسم نمودار پیش‌بینی‌ها"""
    plt.figure(figsize=(15, 7))
    plt.plot(dates, predictions, 'r-', label='پیش‌بینی‌ها')
    plt.title('پیش‌بینی قیمت بیت‌کوین برای 24 ساعت آینده')
    plt.xlabel('تاریخ')
    plt.ylabel('قیمت (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

def main():
    # بارگذاری مدل و اسکالرها
    model_dir = 'saved_models/run_20250313_111519_FEzsKLIJ'
    model, scaler_X, scaler_y = load_saved_model(model_dir)
    
    # بارگذاری داده‌های اصلی
    print("\nبارگذاری داده‌های اصلی...")
    data = pd.read_csv('bitcoin_20170101_20250310_1h.csv', skiprows=[1, 2])
    data['Price'] = pd.to_datetime(data['Price'])
    data.set_index('Price', inplace=True)
    
    # آماده‌سازی آخرین دنباله
    last_sequence = prepare_last_sequence(data, scaler_X)
    
    # پیش‌بینی قیمت‌ها
    predictions, dates = predict_next_hours(model, last_sequence, scaler_X, scaler_y)
    
    # نمایش نتایج
    print("\nپیش‌بینی قیمت‌ها برای 24 ساعت آینده:")
    for date, price in zip(dates, predictions):
        print(f"{date}: ${price:,.2f}")
    
    # رسم نمودار
    plot_predictions(predictions, dates)
    print("\nنمودار پیش‌بینی‌ها در فایل predictions.png ذخیره شد.")

if __name__ == "__main__":
    main() 