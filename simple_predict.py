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

def prepare_features(data):
    """آماده‌سازی ویژگی‌های مورد نیاز برای پیش‌بینی"""
    df = data.copy()
    
    # ویژگی‌های زمانی
    df['Hour'] = df.index.hour
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Is_Weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    # ویژگی‌های قیمت
    df['Hourly_Return'] = df['Close'].pct_change()
    df['Daily_Return'] = df['Close'].pct_change(periods=24)
    
    # میانگین‌های متحرک
    df['SMA_24'] = df['Close'].rolling(window=24).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # نوسانات
    df['Volatility_12h'] = df['Hourly_Return'].rolling(window=12).std()
    df['Volatility_24h'] = df['Hourly_Return'].rolling(window=24).std()
    
    # شکاف قیمت و محدوده
    df['Price_Gap'] = (df['High'] - df['Low']) / df['Close']
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=24).mean()
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    
    # مرتب‌سازی ستون‌ها به ترتیب مورد نیاز مدل
    columns_order = [
        'High', 'Low', 'Open', 'Volume',
        'Hour', 'Day_of_Week', 'Month', 'Is_Weekend',
        'Hourly_Return', 'Daily_Return',
        'SMA_24', 'SMA_50', 'SMA_200',
        'RSI', 'Volatility_12h', 'Volatility_24h',
        'Price_Gap', 'Volume_Ratio', 'Price_Range'
    ]
    
    return df[columns_order]

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

def predict_next_hours(model, last_sequence, scaler_X, scaler_y, hours=24):
    """پیش‌بینی قیمت‌ها برای ساعت‌های آینده"""
    predictions = []
    dates = []
    current_sequence = last_sequence.copy()
    
    # محاسبه تاریخ شروع از آخرین تاریخ موجود در داده‌ها
    last_date = pd.Timestamp('2025-03-09 20:00:00')
    
    # تبدیل آخرین قیمت به مقیاس اصلی
    last_price = scaler_y.inverse_transform(current_sequence[-1:, 0:1])[0][0]
    
    for i in range(hours):
        try:
            # پیش‌بینی قیمت بعدی
            scaled_prediction = model.predict(current_sequence.reshape(1, 24, 19))
            next_price = scaler_y.inverse_transform(scaled_prediction)[0][0]
            
            # اعمال محدودیت برای جلوگیری از نوسانات شدید
            max_change = 0.02  # حداکثر 2% تغییر در هر ساعت
            next_price = np.clip(next_price, 
                               last_price * (1 - max_change),
                               last_price * (1 + max_change))
            
            # به‌روزرسانی توالی با قیمت جدید
            new_row = np.zeros(19)  # تعداد ویژگی‌ها
            
            # تبدیل قیمت جدید به مقیاس نرمال
            scaled_price = scaler_y.transform([[next_price]])[0][0]
            
            new_row[0] = scaled_price  # High
            new_row[1] = scaled_price * 0.999  # Low
            new_row[2] = scaled_price * 1.001  # Open
            new_row[3] = 1.0  # Volume (normalized)
            
            # ویژگی‌های زمانی
            next_date = last_date + pd.Timedelta(hours=i+1)
            new_row[4] = next_date.hour / 24.0  # Hour
            new_row[5] = next_date.dayofweek / 7.0  # Day_of_Week
            new_row[6] = next_date.month / 12.0  # Month
            new_row[7] = 1.0 if next_date.dayofweek >= 5 else 0.0  # Is_Weekend
            
            # ویژگی‌های قیمت
            hourly_return = (next_price - last_price) / last_price
            new_row[8] = hourly_return  # Hourly_Return
            new_row[9] = hourly_return  # Daily_Return
            
            # میانگین‌های متحرک (نرمال شده)
            new_row[10] = scaled_price  # SMA_24
            new_row[11] = scaled_price  # SMA_50
            new_row[12] = scaled_price  # SMA_200
            
            # شاخص‌های تکنیکال
            new_row[13] = 50.0  # RSI (مقدار متوسط)
            new_row[14] = 0.01  # Volatility_12h
            new_row[15] = 0.01  # Volatility_24h
            new_row[16] = hourly_return  # Price_Gap
            new_row[17] = 1.0  # Volume_Ratio
            new_row[18] = 0.002  # Price_Range
            
            # اضافه کردن ردیف جدید به توالی
            current_sequence = np.vstack([current_sequence[1:], new_row])
            
            predictions.append(next_price)
            dates.append(next_date)
            
            last_price = next_price
            
        except Exception as e:
            print(f"خطا در پیش‌بینی ساعت {i+1}: {str(e)}")
            break
    
    return predictions, dates

def calculate_rsi(prices, period=14):
    """محاسبه شاخص قدرت نسبی (RSI)"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def main():
    # بارگذاری مدل و اسکالرها
    model_dir = 'saved_models/run_20250313_111519_FEzsKLIJ'
    model, scaler_X, scaler_y = load_saved_model(model_dir)
    
    # بارگذاری داده‌های اصلی
    print("\nبارگذاری داده‌های اصلی...")
    data = pd.read_csv('bitcoin_20170101_20250310_1h.csv', skiprows=[1, 2])
    data.rename(columns={'Price': 'Date'}, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # تبدیل ستون‌ها به عددی
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # آماده‌سازی ویژگی‌ها
    data_with_features = prepare_features(data)
    
    # گرفتن آخرین 24 ساعت داده
    last_sequence = data_with_features.iloc[-24:].copy()
    
    # نرمال‌سازی داده‌ها
    X_scaled = scaler_X.transform(last_sequence)
    
    # پیش‌بینی
    print("\nانجام پیش‌بینی...")
    predictions, dates = predict_next_hours(model, X_scaled, scaler_X, scaler_y)
    
    # نمایش نتایج
    print("\nپیش‌بینی قیمت‌ها برای 24 ساعت آینده:")
    for date, price in zip(dates, predictions):
        print(f"{date}: ${price:,.2f}")
    
    # رسم نمودار
    plot_predictions(predictions, dates)
    print("\nنمودار پیش‌بینی‌ها در فایل predictions.png ذخیره شد.")

if __name__ == "__main__":
    main() 