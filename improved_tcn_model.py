import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN
from tensorflow.keras.models import Sequential, Model, save_model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, LSTM, Bidirectional
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import joblib
import random
import string
import shutil
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta
from pykalman import KalmanFilter
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# تنظیمات فونت فارسی برای نمودارها
plt.rcParams['font.family'] = 'Tahoma'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # برای نمایش صحیح علامت منفی

def custom_loss(y_true, y_pred):
    """
    تابع زیان سفارشی که ترکیبی از MSE و زیان جهت‌دار است
    با نسبت 3:1 (75% MSE و 25% زیان جهت‌دار)
    """
    # اطمینان از ابعاد درست
    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)
    
    # MSE (میانگین مربعات خطا)
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # زیان جهت‌دار (اطمینان از پیش‌بینی صحیح جهت حرکت قیمت)
    y_true_diff = y_true[1:] - y_true[:-1]
    y_pred_diff = y_pred[1:] - y_pred[:-1]
    
    # تبدیل به مسئله طبقه‌بندی باینری
    direction_true = tf.cast(y_true_diff > 0, tf.float32)
    direction_pred = tf.cast(y_pred_diff > 0, tf.float32)
    
    dir_loss = tf.keras.losses.binary_crossentropy(direction_true, direction_pred)
    
    return 0.75 * mse + 0.25 * tf.reduce_mean(dir_loss)

class OneCycleLR(Callback):
    """پیاده‌سازی استراتژی OneCycle برای نرخ یادگیری"""
    def __init__(self, max_lr, steps_per_epoch, epochs, div_factor=25.0, pct_start=0.3):
        super(OneCycleLR, self).__init__()
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.total_steps = steps_per_epoch * epochs
        self.step_size_up = int(self.total_steps * pct_start)
        self.step_size_down = self.total_steps - self.step_size_up
        self.init_lr = max_lr / div_factor
        self.current_step = 0
        
    def on_train_batch_begin(self, batch, logs=None):
        self.current_step += 1
        if self.current_step <= self.step_size_up:
            computed_lr = self.init_lr + (self.max_lr - self.init_lr) * (self.current_step / self.step_size_up)
        else:
            decay_steps = self.current_step - self.step_size_up
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_steps / self.step_size_down))
            computed_lr = self.init_lr + (self.max_lr - self.init_lr) * cosine_decay
        
        tf.keras.backend.set_value(self.model.optimizer.lr, computed_lr)
        
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr.numpy()
        print(f"\nLearning rate for epoch {epoch+1}: {lr:.6f}")

def create_random_run_name():
    """ایجاد یک نام تصادفی برای اجرای مدل"""
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{random_str}"

def save_model_and_scalers(model, scaler, model_params, run_name, save_dir='saved_models'):
    """ذخیره مدل و اسکالرها"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    run_dir = os.path.join(save_dir, run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    if model is not None:
        model_path = os.path.join(run_dir, 'improved_tcn_model.h5')
        save_model(model, model_path)
        print(f"- مدل: {model_path}")
    
    if scaler is not None:
        scaler_path = os.path.join(run_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"- اسکالر: {scaler_path}")
    
    if model_params is not None:
        params_path = os.path.join(run_dir, 'model_params.json')
        import json
        with open(params_path, 'w') as f:
            json.dump(model_params, f)
        print(f"- پارامترها: {params_path}")
    
    current_file = os.path.abspath(__file__)
    model_file_dest = os.path.join(run_dir, 'improved_tcn_model.py')
    shutil.copy2(current_file, model_file_dest)
    print(f"- فایل کد: {model_file_dest}")
    
    print(f"\nمدل و اسکالرها در مسیر {run_dir} ذخیره شدند.")
    return run_dir

def create_sequences(data, seq_length):
    """تبدیل داده‌ها به توالی‌های مناسب برای TCN"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def predict_future_days(model, last_sequence, scaler, future_days=30):
    """پیش‌بینی قیمت برای روزهای آینده"""
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    # اطمینان از ابعاد درست
    if len(current_sequence.shape) == 2:
        current_sequence = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
    
    for _ in range(future_days):
        # پیش‌بینی قیمت بعدی
        next_pred = model.predict(current_sequence)
        future_predictions.append(next_pred[0, 0])
        
        # ایجاد ردیف جدید با مقادیر پیش‌بینی شده
        new_row = np.zeros((1, current_sequence.shape[2]))
        new_row[0, 0] = next_pred[0, 0]  # قیمت پیش‌بینی شده
        new_row[0, 1] = next_pred[0, 0]  # Open
        new_row[0, 2] = next_pred[0, 0] * 1.01  # High
        new_row[0, 3] = next_pred[0, 0] * 0.99  # Low
        new_row[0, 4] = current_sequence[0, -1, 4]  # Volume
        
        # به‌روزرسانی توالی
        current_sequence = np.vstack([current_sequence[0, 1:], new_row])
        current_sequence = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
    
    return np.array(future_predictions)

def calculate_metrics(y_true, y_pred):
    """محاسبه معیارهای مختلف ارزیابی مدل"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    direction_accuracy = np.mean(direction_true == direction_pred) * 100
    
    win_rate = np.mean(np.diff(y_pred) > 0) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Direction Accuracy': direction_accuracy,
        'Win Rate': win_rate
    }

def add_technical_features(data):
    """اضافه کردن ویژگی‌های تکنیکال به داده‌ها"""
    df = data.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # میانگین‌های متحرک
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # تغییرات قیمت
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_7d'] = df['Close'].pct_change(periods=7)
    
    return df.dropna()

def apply_kalman_filter(data):
    """اعمال فیلتر کالمن برای کاهش نویز"""
    df = data.copy()
    
    # بررسی وجود داده‌های معتبر
    if df.empty:
        raise ValueError("داده‌های ورودی خالی هستند")
    
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            # بررسی وجود مقادیر نامعتبر
            if df[col].isnull().any():
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            kf = KalmanFilter(
                initial_state_mean=df[col].iloc[0],
                observation_covariance=1.0,
                transition_covariance=0.01,
                initial_state_covariance=1.0
            )
            state_means, _ = kf.filter(df[col].values)
            df[f'{col}_kf'] = state_means
    return df

def add_fear_greed_index(data):
    """محاسبه شاخص ترس و طمع"""
    df = data.copy()
    if 'Close' in df.columns:
        df['volatility'] = df['Close'].pct_change().rolling(window=20).std() * 100
        df['ma200'] = df['Close'].rolling(window=200).mean()
        df['ma_distance'] = ((df['Close'] - df['ma200']) / df['ma200'] * 100)
        
        df['fear_greed_index'] = (
            0.4 * (df['RSI'] - 30) / 40 +
            0.3 * (1 - df['volatility'] / df['volatility'].quantile(0.95)) +
            0.3 * ((df['ma_distance'] + 20) / 40).clip(0, 1)
        ) * 100
        
        df['fear_greed_index'] = df['fear_greed_index'].clip(0, 100)
        df['fear_greed_index'].fillna(df['fear_greed_index'].mean(), inplace=True)
    
    return df

def plot_additional_analysis(data, train_predictions, test_predictions, y_train_original, y_test_original,
                           future_predictions, future_dates, train_size, seq_length, run_dir):
    """رسم نمودارهای تحلیلی"""
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # محاسبه شاخص‌های تست
    test_start_idx = train_size + seq_length
    test_end_idx = test_start_idx + len(y_test_original)
    
    # نمودار 1: مقایسه قیمت واقعی و پیش‌بینی شده
    plt.figure(figsize=(14, 6))
    plt.plot(data.index[test_start_idx:test_end_idx], y_test_original, label='قیمت واقعی')
    plt.plot(data.index[test_start_idx:test_end_idx], test_predictions, label='قیمت پیش‌بینی شده')
    plt.fill_between(data.index[test_start_idx:test_end_idx], 
                    y_test_original.flatten() - test_predictions.flatten(),
                    y_test_original.flatten() + test_predictions.flatten(),
                    alpha=0.2, color='gray', label='محدوده خطا')
    plt.title('مقایسه قیمت واقعی و پیش‌بینی شده')
    plt.xlabel('زمان')
    plt.ylabel('قیمت (USD)')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, '01_price_comparison.png'))
    plt.close()
    
    # نمودار 2: توزیع خطاها
    plt.figure(figsize=(14, 6))
    errors = y_test_original.flatten() - test_predictions.flatten()
    plt.hist(errors, bins=50, alpha=0.7)
    plt.title('توزیع خطاهای پیش‌بینی')
    plt.xlabel('خطا (USD)')
    plt.ylabel('تعداد')
    plt.savefig(os.path.join(plots_dir, '02_error_distribution.png'))
    plt.close()
    
    # نمودار 3: پیش‌بینی‌های آینده
    plt.figure(figsize=(14, 6))
    plt.plot(future_dates, future_predictions, 'b-', label='پیش‌بینی قیمت')
    plt.fill_between(future_dates, 
                    future_predictions.flatten() * 0.95,
                    future_predictions.flatten() * 1.05,
                    alpha=0.2, color='blue', label='محدوده اطمینان ۵٪')
    plt.title('پیش‌بینی قیمت برای ۳۰ روز آینده')
    plt.xlabel('تاریخ')
    plt.ylabel('قیمت (USD)')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, '03_future_predictions.png'))
    plt.close()
    
    # نمودار 4: تغییرات روزانه پیش‌بینی شده
    plt.figure(figsize=(14, 6))
    daily_changes = np.diff(future_predictions.flatten()) / future_predictions[:-1].flatten() * 100
    plt.bar(range(len(daily_changes)), daily_changes, alpha=0.7)
    plt.title('تغییرات روزانه پیش‌بینی شده')
    plt.xlabel('روز')
    plt.ylabel('درصد تغییرات')
    plt.savefig(os.path.join(plots_dir, '04_daily_changes.png'))
    plt.close()
    
    # نمودار 5: مقایسه نوسانات
    plt.figure(figsize=(14, 6))
    actual_volatility = np.std(np.diff(y_test_original.flatten()))
    predicted_volatility = np.std(np.diff(test_predictions.flatten()))
    plt.bar(['نوسانات واقعی', 'نوسانات پیش‌بینی شده'], 
            [actual_volatility, predicted_volatility])
    plt.title('مقایسه نوسانات واقعی و پیش‌بینی شده')
    plt.ylabel('انحراف معیار تغییرات قیمت')
    plt.savefig(os.path.join(plots_dir, '05_volatility_comparison.png'))
    plt.close()
    
    # نمودار 6: دقت پیش‌بینی در مقیاس زمانی
    plt.figure(figsize=(14, 6))
    window_size = 10
    accuracy_rolling = []
    for i in range(0, len(y_test_original) - window_size):
        true_direction = np.sign(np.diff(y_test_original[i:i+window_size].flatten()))
        pred_direction = np.sign(np.diff(test_predictions[i:i+window_size].flatten()))
        accuracy = np.mean(true_direction == pred_direction) * 100
        accuracy_rolling.append(accuracy)
    plt.plot(accuracy_rolling)
    plt.title('دقت پیش‌بینی در مقیاس زمانی')
    plt.xlabel('زمان')
    plt.ylabel('دقت (%)')
    plt.savefig(os.path.join(plots_dir, '06_accuracy_over_time.png'))
    plt.close()
    
    # نمودار 7: تحلیل حجم معاملات
    plt.figure(figsize=(14, 6))
    volume_data = data['Volume'].iloc[test_start_idx:test_end_idx]
    plt.plot(volume_data.index, volume_data.values, 'g-', label='حجم معاملات')
    plt.title('تحلیل حجم معاملات')
    plt.xlabel('تاریخ')
    plt.ylabel('حجم')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, '07_volume_analysis.png'))
    plt.close()
    
    # نمودار 8: همبستگی بین خطا و حجم معاملات
    plt.figure(figsize=(14, 6))
    errors = y_test_original.flatten() - test_predictions.flatten()
    plt.scatter(volume_data.values, np.abs(errors), alpha=0.5)
    plt.title('همبستگی بین خطا و حجم معاملات')
    plt.xlabel('حجم معاملات')
    plt.ylabel('خطای مطلق')
    plt.savefig(os.path.join(plots_dir, '08_error_volume_correlation.png'))
    plt.close()
    
    # نمودار 9: تحلیل RSI
    plt.figure(figsize=(14, 6))
    rsi_data = data['RSI'].iloc[test_start_idx:test_end_idx]
    plt.plot(rsi_data.index, rsi_data.values, 'r-', label='RSI')
    plt.axhline(y=70, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='r', linestyle='--', alpha=0.5)
    plt.title('تحلیل شاخص RSI')
    plt.xlabel('تاریخ')
    plt.ylabel('RSI')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, '09_rsi_analysis.png'))
    plt.close()
    
    # نمودار 10: تحلیل میانگین‌های متحرک
    plt.figure(figsize=(14, 6))
    sma50 = data['SMA_21'].iloc[test_start_idx:test_end_idx]
    sma200 = data['SMA_21'].iloc[test_start_idx:test_end_idx]
    plt.plot(sma50.index, sma50.values, 'b-', label='SMA 21')
    plt.plot(sma200.index, sma200.values, 'r-', label='SMA 21')
    plt.title('تحلیل میانگین‌های متحرک')
    plt.xlabel('تاریخ')
    plt.ylabel('قیمت (USD)')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, '10_moving_averages.png'))
    plt.close()
    
    # نمودار 11: تحلیل نوسانات
    plt.figure(figsize=(14, 6))
    volatility_data = data['Price_Change'].iloc[test_start_idx:test_end_idx].rolling(window=20).std() * 100
    plt.plot(volatility_data.index, volatility_data.values, 'm-', label='نوسانات')
    plt.title('نوسانات قیمت در طول زمان')
    plt.xlabel('تاریخ')
    plt.ylabel('انحراف معیار')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, '11_volatility_over_time.png'))
    plt.close()

    # نمودار 12: تحلیل دقت پیش‌بینی در شرایط مختلف بازار
    plt.figure(figsize=(14, 6))
    market_conditions = []
    accuracies = []
    
    for i in range(len(y_test_original) - window_size):
        if volatility_data.iloc[i] > volatility_data.mean():
            market_conditions.append(1)  # نوسانات بالا
        elif rsi_data.iloc[i] > 70:
            market_conditions.append(2)  # اشباع خرید
        elif rsi_data.iloc[i] < 30:
            market_conditions.append(3)  # اشباع فروش
        else:
            market_conditions.append(0)  # عادی
            
        true_direction = np.sign(np.diff(y_test_original[i:i+window_size].flatten()))
        pred_direction = np.sign(np.diff(test_predictions[i:i+window_size].flatten()))
        accuracy = np.mean(true_direction == pred_direction) * 100
        accuracies.append(accuracy)
    
    plt.bar(['عادی', 'نوسانات بالا', 'اشباع خرید', 'اشباع فروش'],
            [np.mean([acc for i, acc in enumerate(accuracies) if market_conditions[i] == 0]),
             np.mean([acc for i, acc in enumerate(accuracies) if market_conditions[i] == 1]),
             np.mean([acc for i, acc in enumerate(accuracies) if market_conditions[i] == 2]),
             np.mean([acc for i, acc in enumerate(accuracies) if market_conditions[i] == 3])])
    plt.title('دقت پیش‌بینی در شرایط مختلف بازار')
    plt.ylabel('دقت (%)')
    plt.savefig(os.path.join(plots_dir, '12_market_conditions_accuracy.png'))
    plt.close()

    # نمودار ترکیبی 1: قیمت، حجم و RSI
    plt.figure(figsize=(15, 10))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # نمودار قیمت
    ax1.plot(data.index[test_start_idx:test_end_idx], y_test_original, 'b-', label='قیمت واقعی')
    ax1.plot(data.index[test_start_idx:test_end_idx], test_predictions, 'r-', label='قیمت پیش‌بینی شده')
    ax1.set_title('تحلیل ترکیبی: قیمت، حجم و RSI')
    ax1.set_ylabel('قیمت (USD)')
    ax1.legend()
    ax1.grid(True)
    
    # نمودار حجم معاملات
    ax2.plot(volume_data.index, volume_data.values, 'g-', label='حجم معاملات')
    ax2.set_ylabel('حجم')
    ax2.legend()
    ax2.grid(True)
    
    # نمودار RSI
    ax3.plot(rsi_data.index, rsi_data.values, 'r-', label='RSI')
    ax3.axhline(y=70, color='g', linestyle='--', alpha=0.5)
    ax3.axhline(y=30, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('تاریخ')
    ax3.set_ylabel('RSI')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '13_combined_price_volume_rsi.png'))
    plt.close()

    # نمودار ترکیبی 2: میانگین‌های متحرک و نوسانات
    plt.figure(figsize=(15, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # نمودار میانگین‌های متحرک
    ax1.plot(sma50.index, sma50.values, 'b-', label='SMA 21')
    ax1.plot(sma200.index, sma200.values, 'r-', label='SMA 21')
    ax1.plot(data.index[test_start_idx:test_end_idx], y_test_original, 'g-', label='قیمت واقعی', alpha=0.5)
    ax1.set_title('تحلیل ترکیبی: میانگین‌های متحرک و نوسانات')
    ax1.set_ylabel('قیمت (USD)')
    ax1.legend()
    ax1.grid(True)
    
    # نمودار نوسانات
    ax2.plot(volatility_data.index, volatility_data.values, 'm-', label='نوسانات')
    ax2.set_xlabel('تاریخ')
    ax2.set_ylabel('انحراف معیار')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '14_combined_moving_averages_volatility.png'))
    plt.close()

    # نمودار ترکیبی 3: دقت پیش‌بینی و شرایط بازار
    plt.figure(figsize=(15, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # نمودار دقت پیش‌بینی
    ax1.plot(data.index[test_start_idx:test_start_idx + len(accuracy_rolling)], accuracy_rolling, 'b-', label='دقت پیش‌بینی')
    ax1.set_title('تحلیل ترکیبی: دقت پیش‌بینی و شرایط بازار')
    ax1.set_ylabel('دقت (%)')
    ax1.legend()
    ax1.grid(True)
    
    # نمودار شرایط بازار
    market_conditions_plot = []
    for i in range(len(y_test_original) - window_size):
        if volatility_data.iloc[i] > volatility_data.mean():
            market_conditions_plot.append(1)  # نوسانات بالا
        elif rsi_data.iloc[i] > 70:
            market_conditions_plot.append(2)  # اشباع خرید
        elif rsi_data.iloc[i] < 30:
            market_conditions_plot.append(3)  # اشباع فروش
        else:
            market_conditions_plot.append(0)  # عادی
    
    ax2.plot(data.index[test_start_idx:test_start_idx + len(market_conditions_plot)], market_conditions_plot, 'r-', label='شرایط بازار')
    ax2.set_xlabel('زمان')
    ax2.set_ylabel('شرایط بازار')
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['عادی', 'نوسانات بالا', 'اشباع خرید', 'اشباع فروش'])
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '15_combined_accuracy_market_conditions.png'))
    plt.close()

def save_plots_and_report(run_dir, data, train_predictions, test_predictions, y_train_original, y_test_original,
                         future_predictions, future_dates, model, scaler, model_params,
                         train_metrics, test_metrics, history):
    """ذخیره نمودارها و گزارش ارزیابی"""
    start_time = time.time()
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_additional_analysis(data, train_predictions, test_predictions, y_train_original, y_test_original,
                           future_predictions, future_dates, len(y_train_original), model_params['seq_length'], run_dir)
    
    # رسم نمودار تاریخچه آموزش
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='خطای آموزش')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='خطای اعتبارسنجی')
    plt.title('روند خطای مدل در طول آموزش')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'loss_history.png'))
    plt.close()
    
    # نوشتن گزارش ارزیابی
    report_path = os.path.join(run_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("گزارش ارزیابی مدل پیش‌بینی قیمت بیت‌کوین با TCN پیشرفته\n")
        f.write("==================================================\n\n")
        
        f.write(f"تاریخ و زمان اجرا: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # اطلاعات دیتاست
        f.write("اطلاعات دیتاست:\n")
        f.write("-"*30 + "\n\n")
        f.write(f"تاریخ شروع داده‌ها: {data.index[0]}\n")
        f.write(f"تاریخ پایان داده‌ها: {data.index[-1]}\n")
        f.write(f"تعداد کل رکوردها: {len(data)}\n")
        f.write(f"تعداد ویژگی‌ها: {len(data.columns)}\n")
        f.write(f"ویژگی‌های موجود: {', '.join(data.columns)}\n\n")
        
        # اطلاعات تقسیم داده‌ها
        f.write("اطلاعات تقسیم داده‌ها:\n")
        f.write("-"*30 + "\n\n")
        train_size = int(len(data) * (1 - model_params['validation_split']))
        f.write(f"تعداد داده‌های آموزشی: {train_size} ({(1-model_params['validation_split'])*100}%)\n")
        f.write(f"تعداد داده‌های تست: {len(data) - train_size} ({model_params['validation_split']*100}%)\n")
        f.write(f"تاریخ شروع داده‌های آموزشی: {data.index[0]}\n")
        f.write(f"تاریخ پایان داده‌های آموزشی: {data.index[train_size-1]}\n")
        f.write(f"تاریخ شروع داده‌های تست: {data.index[train_size]}\n")
        f.write(f"تاریخ پایان داده‌های تست: {data.index[-1]}\n\n")
        
        f.write("پارامترهای مدل:\n")
        for key, value in model_params.items():
            f.write(f"- {key}: {value}\n")
        
        f.write("\nنتایج ارزیابی مدل در مجموعه آموزش:\n")
        for metric, value in train_metrics.items():
            f.write(f"- {metric}: {value:.4f}\n")
        
        f.write("\nنتایج ارزیابی مدل در مجموعه آزمون:\n")
        for metric, value in test_metrics.items():
            f.write(f"- {metric}: {value:.4f}\n")
        
        f.write("\nپیش‌بینی قیمت برای ۳۰ روز آینده:\n")
        for date, price in zip(future_dates, future_predictions):
            f.write(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}\n")
    
    processing_time = time.time() - start_time
    print(f"\nگزارش و نمودارها در {processing_time:.2f} ثانیه ذخیره شدند.")
    print(f"مسیر گزارش: {report_path}")
    print(f"مسیر نمودارها: {plots_dir}")

def predict_with_improved_tcn(data, epochs=35, batch_size=64, seq_length=60, validation_split=0.1, save_model_flag=True):
    """
    پیش‌بینی قیمت بیت‌کوین با استفاده از مدل TCN پیشرفته با بهینه‌سازی‌های زیر:
    1. افزایش تعداد استک‌ها به 3 (nb_stacks=3)
    2. گسترش دیلیشن‌ها به [1,2,4,8,16,32,64,128]
    3. افزایش nb_filters به 128 همراه با تنظیم‌کننده L2 
    4. اضافه کردن لایه BatchNormalization بعد از هر بلوک TCN
    5. تابع زیان سفارشی ترکیبی MSE با Directional Loss
    """
    start_time = time.time()
    
    # پردازش داده‌ها
    data = add_technical_features(data)
    data = apply_kalman_filter(data)
    data = add_fear_greed_index(data)
    
    print(f"\nویژگی‌های استفاده شده در مدل: {', '.join(data.columns)}")
    
    # ایجاد نام تصادفی برای این اجرا
    run_name = create_random_run_name()
    run_dir = os.path.join('saved_models', run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # تنظیم پارامترهای مدل
    nb_stacks = 3
    dilations = [1, 2, 4, 8, 16, 32, 64, 128]
    nb_filters = 128
    kernel_size = 2
    dropout_rate = 0.2
    
    model_params = {
        'epochs': epochs,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'validation_split': validation_split,
        'nb_stacks': nb_stacks,
        'nb_filters': nb_filters,
        'kernel_size': kernel_size,
        'dilations': dilations,
        'dropout_rate': dropout_rate,
        'l2_regularization': 0.001,
        'run_name': run_name
    }
    
    # آماده‌سازی داده‌ها
    features = data[['Close', 'Open', 'High', 'Low', 'Volume']].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # ایجاد توالی‌ها
    X, y = create_sequences(scaled_features, seq_length)
    train_size = int(len(X) * (1 - validation_split))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # ساخت مدل
    model = Sequential([
        TCN(
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_stacks,
            dilations=dilations,
            padding='causal',
            use_skip_connections=True,
            dropout_rate=dropout_rate,
            return_sequences=False,
            input_shape=(seq_length, scaled_features.shape[1])
        ),
        BatchNormalization(),
        Dense(1, kernel_regularizer=l2(0.001))
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=custom_loss
    )
    
    # آموزش مدل
    callbacks = [
        OneCycleLR(max_lr=0.01, steps_per_epoch=len(X_train)//batch_size, epochs=epochs),
        EarlyStopping(patience=10, restore_best_weights=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    # پیش‌بینی
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # برگرداندن مقیاس پیش‌بینی‌ها
    temp_array = np.zeros((len(y_train), features.shape[1]))
    temp_array[:, 0] = y_train
    y_train_original = scaler.inverse_transform(temp_array)[:, 0]
    
    temp_array = np.zeros((len(train_pred), features.shape[1]))
    temp_array[:, 0] = train_pred.flatten()
    train_pred_inv = scaler.inverse_transform(temp_array)[:, 0]
    
    temp_array = np.zeros((len(y_test), features.shape[1]))
    temp_array[:, 0] = y_test
    y_test_original = scaler.inverse_transform(temp_array)[:, 0]
    
    temp_array = np.zeros((len(test_pred), features.shape[1]))
    temp_array[:, 0] = test_pred.flatten()
    test_pred_inv = scaler.inverse_transform(temp_array)[:, 0]
    
    # پیش‌بینی آینده
    last_sequence = scaled_features[-seq_length:]
    future_predictions = predict_future_days(model, last_sequence, scaler)
    future_dates = [data.index[-1] + timedelta(days=x) for x in range(1, 31)]
    
    # محاسبه معیارهای ارزیابی
    train_metrics = calculate_metrics(y_train_original, train_pred_inv)
    test_metrics = calculate_metrics(y_test_original, test_pred_inv)
    
    # ذخیره نتایج
    if save_model_flag:
        save_model_and_scalers(model, scaler, model_params, run_name)
        save_plots_and_report(
            run_dir, data, train_pred_inv, test_pred_inv,
            y_train_original, y_test_original,
            future_predictions, future_dates,
            model, scaler, model_params,
            train_metrics, test_metrics, history
        )
    
    processing_time = time.time() - start_time
    print(f"\nزمان کل اجرا: {processing_time:.2f} ثانیه")
    
    return model, history, scaler, future_predictions, future_dates
