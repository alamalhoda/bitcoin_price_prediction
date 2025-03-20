import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import joblib
import random
import string
import shutil
from tensorflow.keras.models import save_model
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta

# تنظیمات فونت فارسی برای نمودارها
plt.rcParams['font.family'] = 'Tahoma'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # برای نمایش صحیح علامت منفی

def create_random_run_name():
    """
    ایجاد یک نام تصادفی برای اجرای مدل
    
    Returns:
        str: نام تصادفی شامل تاریخ، زمان و رشته تصادفی ۸ کاراکتری
    """
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{random_str}"

def save_model_and_scalers(model, scaler, model_params, run_name, save_dir='saved_models'):
    """
    ذخیره مدل و اسکالرها برای استفاده‌های بعدی
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    run_dir = os.path.join(save_dir, run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # ذخیره مدل و اسکالرها فقط اگر موجود باشند
    if model is not None:
        model_path = os.path.join(run_dir, 'tcn_model.h5')
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
    
    # کپی فایل کد مدل به دایرکتوری اجرا
    current_file = os.path.abspath(__file__)
    model_file_dest = os.path.join(run_dir, 'tcn_model.py')
    shutil.copy2(current_file, model_file_dest)
    print(f"- فایل کد: {model_file_dest}")
    
    print(f"\nمدل و اسکالرها در مسیر {run_dir} ذخیره شدند.")
    
    return run_dir

def plot_additional_analysis(data, train_predictions, test_predictions, y_train_original, y_test_original,
                           future_predictions, future_dates, train_size, seq_length, run_dir):
    """
    رسم نمودارهای تحلیلی بیشتر و ذخیره آنها
    """
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # نمودار قیمت واقعی و پیش‌بینی شده
    plt.figure(figsize=(14, 7))
    plt.plot(y_train_original, label='قیمت واقعی (داده‌های آموزشی)')
    plt.plot(train_predictions, label='پیش‌بینی (داده‌های آموزشی)')
    plt.title('مقایسه قیمت واقعی و پیش‌بینی شده برای داده‌های آموزشی')
    plt.xlabel('زمان')
    plt.ylabel('قیمت (USD)')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'train_predictions.png'))
    plt.close()

    # نمودار خطای پیش‌بینی
    plt.figure(figsize=(14, 7))
    train_errors = np.abs(y_train_original - train_predictions)
    test_errors = np.abs(y_test_original - test_predictions)
    plt.plot(train_errors, label='خطای داده‌های آموزشی')
    plt.plot(test_errors, label='خطای داده‌های تست')
    plt.title('خطای پیش‌بینی در طول زمان')
    plt.xlabel('زمان')
    plt.ylabel('خطای مطلق (USD)')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'prediction_errors.png'))
    plt.close()

    # نمودار توزیع خطاها
    plt.figure(figsize=(10, 6))
    plt.hist(train_errors, bins=50, alpha=0.5, label='داده‌های آموزشی')
    plt.hist(test_errors, bins=50, alpha=0.5, label='داده‌های تست')
    plt.title('توزیع خطاهای پیش‌بینی')
    plt.xlabel('خطای مطلق (USD)')
    plt.ylabel('تعداد')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'error_distribution.png'))
    plt.close()

    # نمودار پیش‌بینی آینده
    plt.figure(figsize=(14, 7))
    plt.plot(future_dates, future_predictions, '--', label='پیش‌بینی آینده')
    plt.title('پیش‌بینی قیمت بیت‌کوین برای ۳۰ روز آینده')
    plt.xlabel('تاریخ')
    plt.ylabel('قیمت (USD)')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'future_predictions.png'))
    plt.close()
    
    # رسم نمودار تغییرات روزانه
    plt.figure(figsize=(14, 7))
    daily_changes = np.diff(future_predictions) / future_predictions[:-1] * 100
    plt.bar(future_dates[1:], daily_changes, color=['g' if x >= 0 else 'r' for x in daily_changes])
    plt.title('درصد تغییرات روزانه پیش‌بینی شده')
    plt.xlabel('تاریخ')
    plt.ylabel('درصد تغییرات')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'daily_changes.png'))
    plt.close()

def save_plots_and_report(run_dir, data, train_predictions, test_predictions, y_train_original, y_test_original,
                         future_predictions, future_dates, model, scaler, model_params,
                         train_metrics, test_metrics, history):
    """
    ذخیره نمودارها و گزارش ارزیابی
    """
    start_time = time.time()
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # رسم نمودارهای تحلیلی
    plot_additional_analysis(data, train_predictions, test_predictions, y_train_original, y_test_original,
                           future_predictions, future_dates, len(y_train_original), model_params['seq_length'], plots_dir)
    
    # رسم نمودار تاریخچه آموزش
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='خطای آموزش')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='خطای اعتبارسنجی')
    plt.title('روند خطای مدل در طول آموزش')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'loss_history.png'))
    plt.close()
    
    # نوشتن گزارش ارزیابی
    report_path = os.path.join(run_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("گزارش ارزیابی مدل پیش‌بینی قیمت بیت‌کوین با TCN\n")
        f.write("==================================================\n\n")
        
        f.write(f"تاریخ و زمان اجرا: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
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

def create_sequences(data, seq_length):
    """تبدیل داده‌ها به توالی‌های مناسب برای TCN با پشتیبانی از چند ویژگی"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]  # فقط Close را پیش‌بینی می‌کنیم
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def predict_future_days(model, last_sequence, scaler, future_days=30):
    """پیش‌بینی قیمت برای روزهای آینده"""
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(future_days):
        next_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
        future_predictions.append(next_pred[0, 0])
        
        new_row = np.zeros((1, current_sequence.shape[1]))
        new_row[0, 0] = next_pred[0, 0]  # قیمت پیش‌بینی شده
        new_row[0, 1] = next_pred[0, 0]  # Open
        new_row[0, 2] = next_pred[0, 0] * 1.01  # High
        new_row[0, 3] = next_pred[0, 0] * 0.99  # Low
        new_row[0, 4] = current_sequence[-1, 4]  # Volume
        
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    return np.array(future_predictions)

def calculate_metrics(y_true, y_pred):
    """
    محاسبه معیارهای مختلف ارزیابی مدل
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    # محاسبه دقت جهت پیش‌بینی
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    direction_accuracy = np.mean(direction_true == direction_pred) * 100
    
    # محاسبه نسبت نوسانات
    volatility_true = np.std(np.diff(y_true) / y_true[:-1])
    volatility_pred = np.std(np.diff(y_pred) / y_pred[:-1])
    volatility_ratio = volatility_pred / volatility_true
    
    # محاسبه نسبت شارپ
    returns_true = np.diff(y_true) / y_true[:-1]
    returns_pred = np.diff(y_pred) / y_pred[:-1]
    sharpe_true = np.mean(returns_true) / np.std(returns_true) * np.sqrt(252)
    sharpe_pred = np.mean(returns_pred) / np.std(returns_pred) * np.sqrt(252)
    sharpe_ratio = sharpe_pred / sharpe_true
    
    # محاسبه نرخ برد
    win_rate = np.mean(returns_pred > 0) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Direction Accuracy': direction_accuracy,
        'Volatility Ratio': volatility_ratio,
        'Sharpe Ratio': sharpe_ratio,
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
    
    # باندهای بولینگر
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # حجم نسبی
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # تغییرات قیمت
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_7d'] = df['Close'].pct_change(periods=7)
    
    # نوسان (ATR)
    from ta.volatility import AverageTrueRange
    atr_indicator = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR'] = atr_indicator.average_true_range()
    
    # حذف ردیف‌های با مقادیر NaN
    df = df.dropna()
    
    return df

def predict_with_tcn(data, epochs=35, batch_size=64, seq_length=60, validation_split=0.1, nb_filters=64, 
                    kernel_size=2, dilations=None, dropout_rate=0.2, save_model_flag=True):
    """پیش‌بینی قیمت بیت‌کوین با استفاده از TCN و چندین ویژگی"""
    start_time = time.time()
    
    # اضافه کردن ویژگی‌های تکنیکال
    data = add_technical_features(data)
    print(f"\nویژگی‌های استفاده شده در مدل: {', '.join(data.columns)}")
    
    # ایجاد نام تصادفی برای این اجرا
    run_name = create_random_run_name()
    
    # ایجاد پوشه برای ذخیره نتایج
    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # تنظیم پارامترهای مدل
    if dilations is None:
        dilations = [1, 2, 4, 8, 16, 32]
    
    model_params = {
        'epochs': epochs,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'validation_split': validation_split,
        'nb_filters': nb_filters,
        'kernel_size': kernel_size,
        'dilations': dilations,
        'dropout_rate': dropout_rate,
        'run_name': run_name
    }
    
    # آماده‌سازی داده‌ها با ویژگی‌های بیشتر
    features = data[['Close', 'Open', 'High', 'Low', 'Volume']].values
    
    # نرمال‌سازی همه ویژگی‌ها
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # ایجاد توالی‌های داده
    X, y = create_sequences(scaled_features, seq_length)
    
    # تقسیم داده‌ها به مجموعه‌های آموزش و آزمون
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # ساخت مدل TCN با استفاده از Functional API
    input_layer = Input(shape=(seq_length, 5))  # 5 ویژگی
    tcn_layer = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=1,
        dilations=dilations,
        padding='causal',
        use_skip_connections=True,
        dropout_rate=dropout_rate,
        return_sequences=False
    )(input_layer)
    output_layer = Dense(1)(tcn_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # نمایش خلاصه مدل
    print("\nساختار مدل TCN:")
    model.summary()
    
    # آموزش مدل
    print("\nشروع آموزش مدل...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    
    # پیش‌بینی برای داده‌های آموزشی
    y_train_pred = model.predict(X_train)

    # برگرداندن مقیاس پیش‌بینی‌های آموزشی به حالت اصلی
    temp_array = np.zeros((len(y_train), 5))
    temp_array[:, 0] = y_train
    y_train_original = scaler.inverse_transform(temp_array)[:, 0]

    temp_array = np.zeros((len(y_train_pred), 5))
    temp_array[:, 0] = y_train_pred.flatten()
    y_train_pred_inv = scaler.inverse_transform(temp_array)[:, 0]
    
    # پیش‌بینی
    print("\nانجام پیش‌بینی...")
    y_pred = model.predict(X_test)
    
    # پیش‌بینی روزهای آینده
    print("\nپیش‌بینی قیمت برای روزهای آینده...")
    last_sequence = scaled_features[-seq_length:]
    future_predictions = predict_future_days(model, last_sequence, scaler, future_days=30)
    
    # برگرداندن مقیاس پیش‌بینی‌ها به حالت اصلی
    temp_array = np.zeros((len(y_test), 5))
    temp_array[:, 0] = y_test
    y_test_original = scaler.inverse_transform(temp_array)[:, 0]
    
    temp_array = np.zeros((len(y_pred), 5))
    temp_array[:, 0] = y_pred.flatten()
    y_test_pred_inv = scaler.inverse_transform(temp_array)[:, 0]
    
    # برگرداندن مقیاس پیش‌بینی‌های آینده
    temp_array = np.zeros((len(future_predictions), 5))
    temp_array[:, 0] = future_predictions
    future_predictions_inv = scaler.inverse_transform(temp_array)[:, 0]
    
    # ایجاد تاریخ‌های آینده
    last_date = pd.to_datetime(data.index[-1])
    future_dates = [last_date + timedelta(days=x) for x in range(1, 31)]
    
    # رسم نتایج
    plt.figure(figsize=(15, 8))
    plt.plot(data.index[-len(y_test):], y_test_original, label='مقادیر واقعی', alpha=0.8)
    plt.plot(data.index[-len(y_pred):], y_test_pred_inv, label='پیش‌بینی مدل', alpha=0.8)
    plt.plot(future_dates, future_predictions_inv, label='پیش‌بینی آینده', linestyle='--', alpha=0.8)
    
    plt.title('پیش‌بینی قیمت بیت‌کوین با استفاده از TCN چند متغیره')
    plt.xlabel('تاریخ')
    plt.ylabel('قیمت (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'tcn_predictions.png'))
    plt.show()

    # نمودار جدید برای پیش‌بینی 30 روز آینده
    plt.figure(figsize=(15, 6))
    
    # محاسبه درصد تغییرات روزانه
    daily_changes = np.diff(future_predictions_inv) / future_predictions_inv[:-1] * 100
    
    # ایجاد نمودار با استفاده از gridspec
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # نمودار قیمت
    ax1.plot(future_dates, future_predictions_inv, 'b-', label='قیمت پیش‌بینی شده', linewidth=2)
    ax1.fill_between(future_dates, 
                     future_predictions_inv * 0.95,  # حد پایین باند اطمینان
                     future_predictions_inv * 1.05,  # حد بالای باند اطمینان
                     alpha=0.2, color='blue')
    
    # اضافه کردن برچسب قیمت به نقاط مهم
    for i, (date, price) in enumerate(zip(future_dates, future_predictions_inv)):
        if i % 5 == 0:  # نمایش هر 5 روز
            ax1.annotate(f'${price:,.0f}', 
                        (date, price),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=8)
    
    ax1.set_title('پیش‌بینی قیمت بیت‌کوین برای 30 روز آینده')
    ax1.set_xlabel('تاریخ')
    ax1.set_ylabel('قیمت (USD)')
    ax1.grid(True)
    ax1.legend()
    
    # نمودار درصد تغییرات
    bars = ax2.bar(future_dates[1:], daily_changes, 
                   color=['g' if x >= 0 else 'r' for x in daily_changes],
                   alpha=0.6,
                   label='درصد تغییرات روزانه')
    
    # اضافه کردن برچسب درصد به نقاط مهم
    for i, (date, change) in enumerate(zip(future_dates[1:], daily_changes)):
        if abs(change) > 2:  # نمایش تغییرات بیشتر از 2 درصد
            ax2.annotate(f'{change:.1f}%', 
                        (date, change),
                        textcoords="offset points",
                        xytext=(0, 10 if change >= 0 else -15),
                        ha='center',
                        fontsize=8)
    
    ax2.set_title('درصد تغییرات روزانه پیش‌بینی شده')
    ax2.set_xlabel('تاریخ')
    ax2.set_ylabel('درصد تغییرات')
    ax2.grid(True)
    ax2.legend()
    
    # تنظیمات نهایی
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'tcn_future_predictions.png'))
    plt.show()
    
    # رسم نمودار loss
    plt.figure(figsize=(15, 6))
    plt.plot(history.history['loss'], label='خطای آموزش')
    plt.plot(history.history['val_loss'], label='خطای اعتبارسنجی')
    plt.title('روند خطای مدل در طول آموزش')
    plt.xlabel('epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'tcn_loss_history.png'))
    plt.show()

    # نمودار ترکیبی آموزش و پیش‌بینی آینده
    plt.figure(figsize=(15, 8))
    train_dates = data.index[-len(y_train_original):]
    plt.plot(train_dates, y_train_original, label='قیمت واقعی (داده‌های آموزشی)', alpha=0.8)
    plt.plot(train_dates, y_train_pred_inv, label='پیش‌بینی (داده‌های آموزشی)', alpha=0.8)
    plt.plot(future_dates, future_predictions_inv, label='پیش‌بینی آینده', linestyle='--', alpha=0.8)
    
    plt.title('مقایسه قیمت واقعی، پیش‌بینی و پیش‌بینی آینده')
    plt.xlabel('تاریخ')
    plt.ylabel('قیمت (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'tcn_combined_predictions.png'))
    plt.show()
    
    # محاسبه و نمایش معیارهای ارزیابی
    train_metrics = calculate_metrics(y_train_original, y_train_pred_inv)
    test_metrics = calculate_metrics(y_test_original, y_test_pred_inv)
    
    print('\nنتایج ارزیابی مدل در مجموعه آموزش:')
    for metric, value in train_metrics.items():
        print(f'{metric}: {value:.4f}')

    print('\nنتایج ارزیابی مدل در مجموعه آزمون:')
    for metric, value in test_metrics.items():
        print(f'{metric}: {value:.4f}')
    
    # ذخیره مدل و نتایج
    if save_model_flag:
        run_dir = save_model_and_scalers(model, scaler, model_params, run_name)
        save_plots_and_report(run_dir, data, y_train_pred_inv, y_test_pred_inv, y_train_original, y_test_original,
                             future_predictions_inv, future_dates, model, scaler, model_params,
                             train_metrics, test_metrics, history)
        
        # نمایش مدت زمان اجرا
        processing_time = time.time() - start_time
        print(f"\nزمان کل اجرا: {processing_time:.2f} ثانیه")
    
    return model, history, scaler, future_predictions_inv, future_dates 

def generate_evaluation_report(model, data, train_predictions, test_predictions, future_predictions, 
                            train_metrics, test_metrics, model_params, run_dir):
    """تولید گزارش ارزیابی جامع"""
    report_path = os.path.join(run_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("گزارش ارزیابی مدل پیش‌بینی قیمت بیت‌کوین با TCN\n")
        f.write("="*50 + "\n\n")
        
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
        
        # پارامترهای مدل
        f.write("پارامترهای مدل:\n")
        f.write("-"*30 + "\n\n")
        for key, value in model_params.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")
        
        # نتایج ارزیابی
        f.write("نتایج ارزیابی مدل در مجموعه آموزش:\n")
        f.write("-"*30 + "\n\n")
        for metric, value in train_metrics.items():
            f.write(f"- {metric}: {value:.4f}\n")
        f.write("\n")
        
        f.write("نتایج ارزیابی مدل در مجموعه آزمون:\n")
        f.write("-"*30 + "\n\n")
        for metric, value in test_metrics.items():
            f.write(f"- {metric}: {value:.4f}\n")
        f.write("\n")
        
        # پیش‌بینی‌های آینده
        f.write("پیش‌بینی قیمت برای ۳۰ روز آینده:\n")
        f.write("-"*30 + "\n\n")
        for date, price in future_predictions.items():
            f.write(f"{date}: ${price:.2f}\n")
        
        print(f"\nگزارش ارزیابی در {report_path} ذخیره شد.") 