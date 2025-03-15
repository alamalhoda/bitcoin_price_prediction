"""
مدل پیش‌بینی قیمت بیت‌کوین با استفاده از شبکه عصبی LSTM

این مدل از شبکه عصبی LSTM برای پیش‌بینی قیمت بیت‌کوین استفاده می‌کند. 
ویژگی‌های اصلی مدل:

معیارهای آموزشی (Features):
- قیمت‌های گذشته (lag_days روز قبل)
- حجم معاملات
- میانگین متحرک ۵۰ روزه (SMA_50)
- میانگین متحرک ۲۰۰ روزه (SMA_200)
- تغییرات روزانه قیمت (درصد)
- شاخص قدرت نسبی (RSI)
- نوسانات قیمت (Volatility)

توزیع داده‌ها:
- ۸۰٪ داده‌ها برای آموزش
- ۲۰٪ داده‌ها برای تست
- داده‌ها به صورت زمانی مرتب شده‌اند
- از MinMaxScaler برای نرمال‌سازی داده‌ها استفاده می‌شود

نحوه وزن‌دهی داده‌ها:
- وزن‌های نمایی از ۱ تا e برای داده‌های اخیر
- نرمال‌سازی وزن‌ها برای حفظ تعادل در آموزش
- تأکید بیشتر روی داده‌های اخیر برای پیش‌بینی دقیق‌تر

معماری مدل:
- لایه ورودی: دریافت توالی زمانی با طول lag_days
- لایه LSTM اول: ۵۰ نورون با return_sequences=True
- Dropout (0.3) برای جلوگیری از overfitting
- لایه LSTM دوم: ۳۰ نورون
- Dropout (0.3)
- لایه خروجی: یک نورون برای پیش‌بینی قیمت

پارامترهای قابل تنظیم:
- lag_days: تعداد روزهای گذشته برای پیش‌بینی (پیش‌فرض: ۳۰)
- epochs: تعداد دوره‌های آموزش (پیش‌فرض: ۵۰)
- batch_size: اندازه batch برای آموزش (پیش‌فرض: ۱۲۸)
- validation_split: نسبت داده‌های validation (پیش‌فرض: ۰.۱)
- lstm1_units: تعداد نورون‌های لایه LSTM اول (پیش‌فرض: ۵۰)
- lstm2_units: تعداد نورون‌های لایه LSTM دوم (پیش‌فرض: ۳۰)
- dropout1: نرخ dropout لایه اول (پیش‌فرض: ۰.۳)
- dropout2: نرخ dropout لایه دوم (پیش‌فرض: ۰.۳)
- early_stopping_patience: تعداد epoch‌های صبر برای early stopping (پیش‌فرض: ۱۰)
- restore_best_weights: ذخیره بهترین وزن‌ها (پیش‌فرض: True)
- optimizer: بهینه‌ساز مورد استفاده (پیش‌فرض: 'adam')
- loss: تابع loss مورد استفاده (پیش‌فرض: 'mse')

Early Stopping و ReduceLROnPlateau:
- نظارت بر validation loss
- patience=10 برای Early Stopping (توقف اگر validation loss در ۱۰ epoch بهبود نیابد)
- restore_best_weights=True (بازگشت به بهترین وزن‌ها)
- ReduceLROnPlateau با factor=0.2 و patience=5 برای کاهش خودکار learning rate

معیارهای ارزیابی:
۱. معیارهای خطای پایه:
   - MSE (Mean Squared Error): معیار خطای میانگین مربعات
   - RMSE (Root Mean Squared Error): جذر میانگین مربعات خطا
   - MAE (Mean Absolute Error): خطای مطلق میانگین
   - MAPE (Mean Absolute Percentage Error): خطای درصدی مطلق میانگین

۲. معیارهای دقت:
   - R² (R-squared): ضریب تعیین
   - Direction Accuracy: دقت پیش‌بینی جهت حرکت قیمت
   - Max Error: حداکثر خطای مطلق
   - Median Error: خطای میانه

۳. معیارهای معاملاتی:
   - Volatility Ratio: نسبت نوسانات پیش‌بینی شده به نوسانات واقعی
   - Sharpe Ratio: نسبت بازده به ریسک
   - Win Rate: درصد پیش‌بینی‌های صحیح
   - Profit Factor: نسبت سود به ضرر

ویژگی‌های اضافی:
- محاسبه RSI با دوره ۱۴ روزه
- پیش‌بینی قیمت برای ۳۰ روز آینده
- نمایش نمودار loss برای نظارت بر روند آموزش
- نمایش نمودار پیش‌بینی‌ها شامل:
  * قیمت واقعی
  * پیش‌بینی قیمت برای داده‌های تست
  * پیش‌بینی قیمت برای ۳۰ روز آینده
- نمودارهای تحلیل پیش‌بینی شامل:
  * نمودار مقایسه قیمت واقعی و پیش‌بینی شده با محدوده خطا
  * نمودار توزیع خطاهای پیش‌بینی
  * نمودار پیش‌بینی قیمت برای ۳۰ روز آینده با محدوده اطمینان
  * نمودار تغییرات روزانه پیش‌بینی شده
  * نمودار مقایسه نوسانات واقعی و پیش‌بینی شده
  * نمودار دقت پیش‌بینی در مقیاس زمانی
  * نمودار تحلیل حجم معاملات و قیمت
  * نمودار همبستگی بین خطا و حجم معاملات
  * نمودار تحلیل RSI و پیش‌بینی‌ها
  * نمودار تحلیل میانگین‌های متحرک
  * نمودار تحلیل نوسانات در مقیاس زمانی
  * نمودار تحلیل دقت پیش‌بینی در شرایط مختلف بازار
- گزارش جامع ارزیابی شامل:
  * تحلیل معیارهای خطا
  * تحلیل دقت پیش‌بینی‌ها
  * تحلیل معیارهای معاملاتی
  * تحلیل روند پیش‌بینی‌های آینده
  * پیش‌بینی‌های کلیدی
  * توصیه‌های بهبود مدل

نکات مهم:
- مدل از توالی‌های زمانی استفاده می‌کند
- از Dropout برای جلوگیری از overfitting استفاده می‌شود
- وزن‌دهی نمایی داده‌های اخیر برای بهبود دقت پیش‌بینی‌های کوتاه‌مدت
- استفاده از ReduceLROnPlateau برای تنظیم خودکار learning rate
- امکان تنظیم پارامترهای مختلف برای بهینه‌سازی عملکرد
- ارزیابی جامع مدل با معیارهای مختلف برای اطمینان از عملکرد مناسب
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import os
import joblib
from tensorflow.keras.models import save_model
import datetime
import random
import string
import shutil
import psutil
import tensorflow as tf
import time

def create_random_run_name():
    """
    ایجاد یک نام تصادفی برای اجرای مدل
    
    Returns:
        str: نام تصادفی شامل تاریخ، زمان و رشته تصادفی ۸ کاراکتری
    """
    # ایجاد یک رشته تصادفی 8 کاراکتری
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    # اضافه کردن تاریخ و زمان
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{random_str}"

def save_model_and_scalers(model, scaler_X, scaler_y, model_params, run_name, save_dir='saved_models'):
    """
    ذخیره مدل و اسکالرها برای استفاده‌های بعدی
    
    Args:
        model: مدل LSTM آموزش دیده
        scaler_X: اسکالر ویژگی‌های ورودی
        scaler_y: اسکالر قیمت هدف
        model_params: پارامترهای مدل
        run_name: نام اجرای فعلی
        save_dir: مسیر ذخیره‌سازی
    
    Returns:
        str: مسیر دایرکتوری اجرای فعلی
    """
    # ایجاد دایرکتوری اصلی اگر وجود نداشته باشد
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # ایجاد دایرکتوری برای اجرای فعلی
    run_dir = os.path.join(save_dir, run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # ذخیره مدل
    model_path = os.path.join(run_dir, 'lstm_model.h5')
    save_model(model, model_path)
    
    # ذخیره اسکالرها
    scaler_X_path = os.path.join(run_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(run_dir, 'scaler_y.pkl')
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    
    # ذخیره پارامترهای مدل
    params_path = os.path.join(run_dir, 'model_params.json')
    import json
    with open(params_path, 'w') as f:
        json.dump(model_params, f)
    
    # ذخیره فایل lstm_model.py
    import shutil
    current_file = os.path.abspath(__file__)
    model_file_dest = os.path.join(run_dir, 'lstm_model.py')
    shutil.copy2(current_file, model_file_dest)
    
    print(f"\nمدل و اسکالرها در مسیر {run_dir} ذخیره شدند:")
    print(f"- مدل: {model_path}")
    print(f"- اسکالر X: {scaler_X_path}")
    print(f"- اسکالر y: {scaler_y_path}")
    print(f"- پارامترها: {params_path}")
    print(f"- فایل کد: {model_file_dest}")
    
    return run_dir

def save_plots_and_report(run_dir, data, train_predictions, test_predictions, y_train_original, y_test_original,
                           future_predictions, future_dates, model, scaler_X, scaler_y, model_params,
                           train_metrics, test_metrics, history):
    """ذخیره نمودارها و گزارش ارزیابی مدل"""
    # ثبت زمان شروع
    start_time = time.time()
    
    # ایجاد دایرکتوری برای ذخیره نمودارها
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # محاسبه train_size
    train_size = int(len(data) * model_params['train_split_ratio'])
    
    # محاسبه شاخص‌های تست
    test_start_idx = train_size + model_params['lag_days']
    test_end_idx = test_start_idx + len(y_test_original)
    
    # نمودار مقایسه قیمت واقعی و پیش‌بینی شده در مرحله تست
    plt.figure(figsize=(15, 8))
    
    # رسم داده‌های تست
    plt.plot(data.index[test_start_idx:test_end_idx], y_test_original, label='قیمت واقعی', color='blue')
    plt.plot(data.index[test_start_idx:test_end_idx], test_predictions, label='قیمت پیش‌بینی شده', color='red', alpha=0.7)
    
    # رسم پیش‌بینی‌های آینده
    plt.plot(future_dates, future_predictions, label='پیش‌بینی ۳۰ روز آینده', color='green', linestyle='--', alpha=0.8)
    
    plt.title('مقایسه قیمت واقعی و پیش‌بینی شده در مرحله تست')
    plt.xlabel('تاریخ')
    plt.ylabel('قیمت (دلار)')
    plt.grid(True)
    plt.legend()
    
    # تنظیم فرمت تاریخ در محور x
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'test_predictions_comparison.png'))
    plt.close()
    
    # ذخیره نمودارها
    plot_additional_analysis(data, train_predictions, test_predictions, y_train_original, y_test_original,
                           future_predictions, future_dates, train_size, model_params['lag_days'], plots_dir)
    
    # نمودار پیش‌بینی ۳۰ روز آینده
    plt.figure(figsize=(15, 8))
    plt.plot(future_dates, future_predictions, 'b-', label='پیش‌بینی قیمت')
    plt.fill_between(future_dates, 
                    future_predictions.flatten() * 0.95,
                    future_predictions.flatten() * 1.05,
                    alpha=0.2, color='blue', label='محدوده اطمینان ۵٪')
    plt.title('پیش‌بینی قیمت بیت‌کوین برای ۳۰ روز آینده')
    plt.xlabel('تاریخ')
    plt.ylabel('قیمت (دلار)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'future_predictions_30_days.png'))
    plt.close()
    
    # نمودار تابع loss
    plt.figure(figsize=(15, 8))
    plt.plot(history.history['loss'], label='Loss آموزش')
    plt.plot(history.history['val_loss'], label='Loss اعتبارسنجی')
    plt.title('نمودار تابع Loss در طول آموزش')
    plt.xlabel('دوره (Epoch)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'loss_function.png'))
    plt.close()
    
    # ذخیره گزارش
    report_path = os.path.join(run_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("گزارش ارزیابی مدل پیش‌بینی قیمت بیت‌کوین\n")
        f.write("==================================================\n\n")
        
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
        f.write(f"تعداد داده‌های آموزشی: {train_size} ({model_params['train_split_ratio']*100}%)\n")
        f.write(f"تعداد داده‌های تست: {len(data) - train_size} ({(1-model_params['train_split_ratio'])*100}%)\n")
        f.write(f"تاریخ شروع داده‌های آموزشی: {data.index[0]}\n")
        f.write(f"تاریخ پایان داده‌های آموزشی: {data.index[train_size-1]}\n")
        f.write(f"تاریخ شروع داده‌های تست: {data.index[train_size]}\n")
        f.write(f"تاریخ پایان داده‌های تست: {data.index[-1]}\n\n")
        
        # اطلاعات حافظه
        memory_info = psutil.Process().memory_info()
        f.write(f"اطلاعات حافظه:\n")
        f.write(f"- مصرف حافظه: {memory_info.rss / 1024 / 1024:.2f} MB\n\n")
        
        # محاسبه سرعت پردازش
        end_time = time.time()
        processing_time = end_time - start_time
        f.write(f"سرعت پردازش:\n")
        f.write(f"- زمان کل پردازش: {processing_time:.2f} ثانیه\n")
        f.write(f"- زمان پردازش به ازای هر نمونه: {processing_time / len(data):.4f} ثانیه\n\n")
        
        # معیارهای ارزیابی
        f.write("تحلیل معیارهای ارزیابی:\n")
        f.write("-"*30 + "\n\n")
        
        f.write("۱. معیارهای خطای پایه:\n")
        f.write("-"*30 + "\n\n")
        f.write("MSE (خطای میانگین مربعات):\n")
        f.write(f"- نتیجه آموزش: {train_metrics['MSE']:.2f}\n")
        f.write(f"- نتیجه تست: {test_metrics['MSE']:.2f}\n\n")
        
        f.write("RMSE (جذر میانگین مربعات خطا):\n")
        f.write(f"- نتیجه آموزش: {train_metrics['RMSE']:.2f}\n")
        f.write(f"- نتیجه تست: {test_metrics['RMSE']:.2f}\n\n")
        
        f.write("MAE (خطای مطلق میانگین):\n")
        f.write(f"- نتیجه آموزش: {train_metrics['MAE']:.2f}\n")
        f.write(f"- نتیجه تست: {test_metrics['MAE']:.2f}\n\n")
        
        f.write("MAPE (خطای درصدی مطلق میانگین):\n")
        f.write(f"- نتیجه آموزش: {train_metrics['MAPE']:.2f}%\n")
        f.write(f"- نتیجه تست: {test_metrics['MAPE']:.2f}%\n\n")
        
        f.write("۲. معیارهای دقت:\n")
        f.write("-"*30 + "\n\n")
        f.write("R² (ضریب تعیین):\n")
        f.write(f"- نتیجه آموزش: {train_metrics['R2']:.4f}\n")
        f.write(f"- نتیجه تست: {test_metrics['R2']:.4f}\n\n")
        
        f.write("دقت پیش‌بینی جهت حرکت (Direction Accuracy):\n")
        f.write(f"- نتیجه آموزش: {train_metrics['Direction_Accuracy']:.2f}%\n")
        f.write(f"- نتیجه تست: {test_metrics['Direction_Accuracy']:.2f}%\n\n")
        
        f.write("حداکثر خطا (Max Error):\n")
        f.write(f"- نتیجه آموزش: ${train_metrics['Max_Error']:.2f}\n")
        f.write(f"- نتیجه تست: ${test_metrics['Max_Error']:.2f}\n\n")
        
        f.write("خطای میانه (Median Error):\n")
        f.write(f"- نتیجه آموزش: ${train_metrics['Median_Error']:.2f}\n")
        f.write(f"- نتیجه تست: ${test_metrics['Median_Error']:.2f}\n\n")
        
        f.write("۳. معیارهای معاملاتی:\n")
        f.write("-"*30 + "\n\n")
        f.write("نسبت نوسانات (Volatility Ratio):\n")
        f.write(f"- نتیجه آموزش: {train_metrics['Volatility_Ratio']:.2f}\n")
        f.write(f"- نتیجه تست: {test_metrics['Volatility_Ratio']:.2f}\n\n")
        
        f.write("نسبت شارپ (Sharpe Ratio):\n")
        f.write(f"- نتیجه آموزش: {train_metrics['Sharpe_Ratio']:.2f}\n")
        f.write(f"- نتیجه تست: {test_metrics['Sharpe_Ratio']:.2f}\n\n")
        
        f.write("نرخ برد (Win Rate):\n")
        f.write(f"- نتیجه آموزش: {train_metrics['Win_Rate']:.2f}%\n")
        f.write(f"- نتیجه تست: {test_metrics['Win_Rate']:.2f}%\n\n")
        
        f.write("فاکتور سود (Profit Factor):\n")
        f.write(f"- نتیجه آموزش: {train_metrics['Profit_Factor']:.2f}\n")
        f.write(f"- نتیجه تست: {test_metrics['Profit_Factor']:.2f}\n\n")
        
        # جزئیات معماری و پارامترهای مدل
        f.write("جزئیات معماری و پارامترهای مدل:\n")
        f.write("--------------------------------\n")
        f.write(f"معماری شبکه:\n")
        f.write(f"- لایه ورودی: دریافت توالی زمانی با طول {model_params['lag_days']}\n")
        f.write(f"- لایه LSTM: 32 نورون با تابع فعال‌سازی tanh\n")
        f.write(f"- Dropout (0.2)\n")
        f.write(f"- لایه خروجی: یک نورون\n\n")
        
        f.write(f"پارامترهای آموزش:\n")
        f.write(f"- تعداد روزهای گذشته (lag_days): {model_params['lag_days']}\n")
        f.write(f"- تعداد دوره‌های آموزش (epochs): {model_params['epochs']}\n")
        f.write(f"- اندازه batch: {model_params['batch_size']}\n")
        f.write(f"- نسبت validation: {model_params['validation_split']}\n")
        f.write(f"- بهینه‌ساز: {model_params['optimizer']}\n")
        f.write(f"- تابع loss: {model_params['loss']}\n")
        f.write(f"- Early Stopping: patience={model_params['early_stopping_patience']}, restore_best_weights=True\n\n")
        
        # پیش‌بینی‌های آینده
        f.write("پیش‌بینی قیمت برای ۳۰ روز آینده:\n")
        for date, price in zip(future_dates, future_predictions):
            f.write(f"{date}: ${price[0]:.2f}\n")
        
        f.write("\n==================================================\n")

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

def predict_future_days(model, last_sequence, scaler_X, scaler_y, data, future_days=30):
    """
    پیش‌بینی قیمت برای روزهای آینده با به‌روزرسانی شاخص‌های تکنیکال
    
    Args:
        model: مدل LSTM آموزش دیده
        last_sequence: آخرین توالی داده‌های ورودی
        scaler_X: اسکالر ویژگی‌های ورودی
        scaler_y: اسکالر قیمت هدف
        data: DataFrame اصلی داده‌ها
        future_days: تعداد روزهای آینده برای پیش‌بینی (پیش‌فرض: ۳۰)
    
    Returns:
        array: پیش‌بینی‌های قیمت برای روزهای آینده
    """
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    # ایجاد یک کپی از داده‌های اصلی برای به‌روزرسانی شاخص‌ها
    future_data = data.copy()
    
    for _ in range(future_days):
        # پیش‌بینی قیمت روز بعد
        next_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
        future_predictions.append(next_pred[0])
        
        # به‌روزرسانی قیمت در داده‌های آینده
        next_date = future_data.index[-1] + pd.Timedelta(days=1)
        next_price = scaler_y.inverse_transform(next_pred)[0][0]
        
        # اضافه کردن ردیف جدید به داده‌های آینده
        new_row = pd.DataFrame({
            'Close': next_price,
            'Open': next_price * 0.99,  # تخمین قیمت باز شدن
            'High': next_price * 1.01,  # تخمین بالاترین قیمت
            'Low': next_price * 0.99,   # تخمین پایین‌ترین قیمت
            'Volume': future_data['Volume'].mean(),  # استفاده از میانگین حجم معاملات
            'Daily_Change': (next_price - future_data['Close'].iloc[-1]) / future_data['Close'].iloc[-1] * 100
        }, index=[next_date])
        
        future_data = pd.concat([future_data, new_row])
        
        # به‌روزرسانی شاخص‌های تکنیکال
        future_data['SMA_50'] = future_data['Close'].rolling(window=50).mean()
        future_data['SMA_200'] = future_data['Close'].rolling(window=200).mean()
        future_data['RSI'] = calculate_rsi(future_data)
        future_data['Volatility'] = future_data['Close'].rolling(window=14).std()
        
        # آماده‌سازی توالی جدید برای پیش‌بینی بعدی
        new_sequence = []
        for i in range(1, current_sequence.shape[0] + 1):
            lag_features = []
            # اضافه کردن قیمت‌های گذشته
            for j in range(1, 31):  # 30 روز گذشته
                lag_features.append(future_data['Close'].iloc[-i-j+1])
            # اضافه کردن سایر ویژگی‌ها
            lag_features.extend([
                future_data['Volume'].iloc[-i],
                future_data['SMA_50'].iloc[-i],
                future_data['SMA_200'].iloc[-i],
                future_data['Daily_Change'].iloc[-i],
                future_data['RSI'].iloc[-i],
                future_data['Volatility'].iloc[-i]
            ])
            new_sequence.append(lag_features)
        
        # نرمال‌سازی توالی جدید
        new_sequence = np.array(new_sequence)
        new_sequence = scaler_X.transform(new_sequence)
        current_sequence = new_sequence
    
    # تبدیل پیش‌بینی‌ها به مقیاس اصلی
    future_predictions = np.array(future_predictions)
    future_predictions = scaler_y.inverse_transform(future_predictions)
    
    return future_predictions

def generate_evaluation_report(train_metrics, test_metrics, future_predictions, future_dates, model_params):
    """تولید گزارش ارزیابی مدل"""
    print("\n==================================================")
    print("گزارش ارزیابی مدل پیش‌بینی قیمت بیت‌کوین")
    print("==================================================\n")
    
    print("تحلیل معیارهای ارزیابی:")
    print("-"*30 + "\n")
    
    print("۱. معیارهای خطای پایه:")
    print("-"*30 + "\n")
    print("MSE (خطای میانگین مربعات):")
    print(f"- نتیجه آموزش: {train_metrics['MSE']:.2f}")
    print(f"- نتیجه تست: {test_metrics['MSE']:.2f}\n")
    
    print("RMSE (جذر میانگین مربعات خطا):")
    print(f"- نتیجه آموزش: {train_metrics['RMSE']:.2f}")
    print(f"- نتیجه تست: {test_metrics['RMSE']:.2f}\n")
    
    print("MAE (خطای مطلق میانگین):")
    print(f"- نتیجه آموزش: {train_metrics['MAE']:.2f}")
    print(f"- نتیجه تست: {test_metrics['MAE']:.2f}\n")
    
    print("MAPE (خطای درصدی مطلق میانگین):")
    print(f"- نتیجه آموزش: {train_metrics['MAPE']:.2f}%")
    print(f"- نتیجه تست: {test_metrics['MAPE']:.2f}%\n")
    
    print("۲. معیارهای دقت:")
    print("-"*30 + "\n")
    print("R² (ضریب تعیین):")
    print(f"- نتیجه آموزش: {train_metrics['R2']:.4f}")
    print(f"- نتیجه تست: {test_metrics['R2']:.4f}\n")
    
    print("دقت پیش‌بینی جهت حرکت (Direction Accuracy):")
    print(f"- نتیجه آموزش: {train_metrics['Direction_Accuracy']:.2f}%")
    print(f"- نتیجه تست: {test_metrics['Direction_Accuracy']:.2f}%\n")
    
    print("حداکثر خطا (Max Error):")
    print(f"- نتیجه آموزش: ${train_metrics['Max_Error']:.2f}")
    print(f"- نتیجه تست: ${test_metrics['Max_Error']:.2f}\n")
    
    print("خطای میانه (Median Error):")
    print(f"- نتیجه آموزش: ${train_metrics['Median_Error']:.2f}")
    print(f"- نتیجه تست: ${test_metrics['Median_Error']:.2f}\n")
    
    print("۳. معیارهای معاملاتی:")
    print("-"*30 + "\n")
    print("نسبت نوسانات (Volatility Ratio):")
    print(f"- نتیجه آموزش: {train_metrics['Volatility_Ratio']:.2f}")
    print(f"- نتیجه تست: {test_metrics['Volatility_Ratio']:.2f}\n")
    
    print("نسبت شارپ (Sharpe Ratio):")
    print(f"- نتیجه آموزش: {train_metrics['Sharpe_Ratio']:.2f}")
    print(f"- نتیجه تست: {test_metrics['Sharpe_Ratio']:.2f}\n")
    
    print("نرخ برد (Win Rate):")
    print(f"- نتیجه آموزش: {train_metrics['Win_Rate']:.2f}%")
    print(f"- نتیجه تست: {test_metrics['Win_Rate']:.2f}%\n")
    
    print("فاکتور سود (Profit Factor):")
    print(f"- نتیجه آموزش: {train_metrics['Profit_Factor']:.2f}")
    print(f"- نتیجه تست: {test_metrics['Profit_Factor']:.2f}\n")
    
    print("۴. تحلیل روند پیش‌بینی‌های آینده:")
    print("-"*30 + "\n")
    price_change = ((future_predictions[-1][0] - future_predictions[0][0]) / future_predictions[0][0]) * 100
    print(f"تغییر قیمت پیش‌بینی شده در ۳۰ روز آینده: {price_change:.2f}%\n")
    
    print("۵. پیش‌بینی‌های کلیدی:")
    print("-"*30 + "\n")
    print(f"حداقل قیمت پیش‌بینی شده: ${min(future_predictions)[0]:.2f}")
    print(f"حداکثر قیمت پیش‌بینی شده: ${max(future_predictions)[0]:.2f}")
    print(f"میانگین قیمت پیش‌بینی شده: ${np.mean(future_predictions):.2f}\n")
    
    print("۶. جزئیات معماری و پارامترهای مدل:")
    print("-"*30 + "\n")
    print("معماری شبکه:")
    print(f"- لایه ورودی: دریافت توالی زمانی با طول {model_params['lag_days']}")
    print(f"- لایه LSTM اول: ۵۰ نورون با تابع فعال‌سازی tanh")
    print(f"- Dropout (0.3)")
    print(f"- لایه LSTM دوم: ۳۰ نورون")
    print(f"- Dropout (0.3)")
    print(f"- لایه خروجی: یک نورون\n")
    
    print("پارامترهای آموزش:")
    print(f"- تعداد روزهای گذشته (lag_days): {model_params['lag_days']}")
    print(f"- تعداد دوره‌های آموزش (epochs): {model_params['epochs']}")
    print(f"- اندازه batch: {model_params['batch_size']}")
    print(f"- نسبت validation: {model_params['validation_split']}")
    print(f"- بهینه‌ساز: {model_params['optimizer']}")
    print(f"- تابع loss: {model_params['loss']}")
    print(f"- Early Stopping: patience={model_params['early_stopping_patience']}, restore_best_weights=True")
    print(f"- ReduceLROnPlateau: factor=0.2, patience=5\n")
    
    print("==================================================\n")

def plot_additional_analysis(data, train_predictions, test_predictions, y_train_original, y_test_original,
                           future_predictions, future_dates, train_size, lag_days, run_dir):
    """
    رسم نمودارهای اضافی برای تحلیل پیش‌بینی‌ها و ذخیره آنها در مسیر اجرای فعلی
    """
    # محاسبه شاخص‌های تست
    test_start_idx = train_size + lag_days
    test_end_idx = test_start_idx + len(y_test_original)
    
    # نمودار 1: مقایسه قیمت واقعی و پیش‌بینی شده
    plt.figure(figsize=(14, 6))
    plt.plot(data.index[test_start_idx:test_end_idx], y_test_original, label='Actual Price')
    plt.plot(data.index[test_start_idx:test_end_idx], test_predictions, label='Predicted Price')
    plt.fill_between(data.index[test_start_idx:test_end_idx], 
                    y_test_original.flatten() - test_predictions.flatten(),
                    y_test_original.flatten() + test_predictions.flatten(),
                    alpha=0.2, color='gray', label='Error Range')
    plt.title('Comparison of Actual and Predicted Prices\nمقایسه قیمت واقعی و پیش‌بینی شده')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig(os.path.join(run_dir, '01_price_comparison.png'))
    plt.close()
    
    # نمودار 2: توزیع خطاها
    plt.figure(figsize=(14, 6))
    errors = y_test_original.flatten() - test_predictions.flatten()
    plt.hist(errors, bins=50, alpha=0.7)
    plt.title('Distribution of Prediction Errors\nتوزیع خطاهای پیش‌بینی')
    plt.xlabel('Error (USD)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(run_dir, '02_error_distribution.png'))
    plt.close()
    
    # نمودار 3: پیش‌بینی‌های آینده
    plt.figure(figsize=(14, 6))
    plt.plot(future_dates, future_predictions, 'b-', label='Price Prediction')
    plt.fill_between(future_dates, 
                    future_predictions.flatten() * 0.95,
                    future_predictions.flatten() * 1.05,
                    alpha=0.2, color='blue', label='5% Confidence Interval')
    plt.title('30-Day Price Prediction\nپیش‌بینی قیمت برای ۳۰ روز آینده')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig(os.path.join(run_dir, '03_future_predictions.png'))
    plt.close()
    
    # نمودار 4: تغییرات روزانه پیش‌بینی شده
    plt.figure(figsize=(14, 6))
    daily_changes = np.diff(future_predictions.flatten()) / future_predictions[:-1].flatten() * 100
    plt.bar(range(len(daily_changes)), daily_changes, alpha=0.7)
    plt.title('Predicted Daily Changes\nتغییرات روزانه پیش‌بینی شده')
    plt.xlabel('Day')
    plt.ylabel('Percentage Change')
    plt.savefig(os.path.join(run_dir, '04_daily_changes.png'))
    plt.close()
    
    # نمودار 5: مقایسه نوسانات
    plt.figure(figsize=(14, 6))
    actual_volatility = np.std(np.diff(y_test_original.flatten()))
    predicted_volatility = np.std(np.diff(test_predictions.flatten()))
    plt.bar(['Actual Volatility', 'Predicted Volatility'], 
            [actual_volatility, predicted_volatility])
    plt.title('Comparison of Actual and Predicted Volatility\nمقایسه نوسانات واقعی و پیش‌بینی شده')
    plt.ylabel('Standard Deviation of Price Changes')
    plt.savefig(os.path.join(run_dir, '05_volatility_comparison.png'))
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
    plt.plot(data.index[test_start_idx:test_start_idx + len(accuracy_rolling)], accuracy_rolling)
    plt.title('Prediction Accuracy Over Time\nدقت پیش‌بینی در مقیاس زمانی')
    plt.xlabel('Time')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(run_dir, '06_accuracy_over_time.png'))
    plt.close()
    
    # نمودار 7: تحلیل حجم معاملات
    plt.figure(figsize=(14, 6))
    volume_data = data['Volume'].iloc[test_start_idx:test_end_idx]
    plt.plot(volume_data.index, volume_data.values, 'g-', label='Trading Volume')
    plt.title('Trading Volume Analysis\nتحلیل حجم معاملات')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.savefig(os.path.join(run_dir, '07_volume_analysis.png'))
    plt.close()
    
    # نمودار 8: همبستگی بین خطا و حجم معاملات
    plt.figure(figsize=(14, 6))
    errors = y_test_original.flatten() - test_predictions.flatten()
    plt.scatter(volume_data.values, np.abs(errors), alpha=0.5)
    plt.title('Error vs Trading Volume Correlation\nهمبستگی بین خطا و حجم معاملات')
    plt.xlabel('Trading Volume')
    plt.ylabel('Absolute Error')
    plt.savefig(os.path.join(run_dir, '08_error_volume_correlation.png'))
    plt.close()
    
    # نمودار 9: تحلیل RSI
    plt.figure(figsize=(14, 6))
    rsi_data = data['RSI'].iloc[test_start_idx:test_end_idx]
    plt.plot(rsi_data.index, rsi_data.values, 'r-', label='RSI')
    plt.axhline(y=70, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='r', linestyle='--', alpha=0.5)
    plt.title('RSI Analysis\nتحلیل شاخص RSI')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.savefig(os.path.join(run_dir, '09_rsi_analysis.png'))
    plt.close()
    
    # نمودار 10: تحلیل میانگین‌های متحرک
    plt.figure(figsize=(14, 6))
    sma50 = data['SMA_50'].iloc[test_start_idx:test_end_idx]
    sma200 = data['SMA_200'].iloc[test_start_idx:test_end_idx]
    plt.plot(sma50.index, sma50.values, 'b-', label='SMA 50')
    plt.plot(sma200.index, sma200.values, 'r-', label='SMA 200')
    plt.title('Moving Averages Analysis\nتحلیل میانگین‌های متحرک')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig(os.path.join(run_dir, '10_moving_averages.png'))
    plt.close()
    
    # نمودار 11: تحلیل نوسانات
    plt.figure(figsize=(14, 6))
    volatility_data = data['Volatility'].iloc[test_start_idx:test_end_idx]
    plt.plot(volatility_data.index, volatility_data.values, 'm-', label='Volatility')
    plt.title('Price Volatility Over Time\nنوسانات قیمت در طول زمان')
    plt.xlabel('Date')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.savefig(os.path.join(run_dir, '11_volatility_over_time.png'))
    plt.close()

    # نمودار 12: تحلیل دقت پیش‌بینی در شرایط مختلف بازار
    plt.figure(figsize=(14, 6))
    market_conditions = []
    accuracies = []
    for i in range(0, len(y_test_original) - window_size):
        true_direction = np.sign(np.diff(y_test_original[i:i+window_size].flatten()))
        pred_direction = np.sign(np.diff(test_predictions[i:i+window_size].flatten()))
        accuracy = np.mean(true_direction == pred_direction) * 100
        
        # تعیین شرایط بازار
        if volatility_data.iloc[i] > volatility_data.mean():
            condition = 'High Volatility'
        elif rsi_data.iloc[i] > 70:
            condition = 'Overbought'
        elif rsi_data.iloc[i] < 30:
            condition = 'Oversold'
        else:
            condition = 'Normal'
            
        market_conditions.append(condition)
        accuracies.append(accuracy)
    
    # محاسبه میانگین دقت برای هر شرایط بازار
    conditions = list(set(market_conditions))
    avg_accuracies = [np.mean([acc for cond, acc in zip(market_conditions, accuracies) if cond == c]) for c in conditions]
    
    plt.bar(conditions, avg_accuracies)
    plt.title('Prediction Accuracy by Market Condition\nدقت پیش‌بینی در شرایط مختلف بازار')
    plt.xlabel('Market Condition')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(run_dir, '12_accuracy_by_market_condition.png'))
    plt.close()

    # نمودارهای ترکیبی
    # نمودار ترکیبی 1: قیمت، حجم معاملات و RSI
    plt.figure(figsize=(15, 10))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # نمودار قیمت
    ax1.plot(data.index[test_start_idx:test_end_idx], y_test_original, label='Actual Price', color='blue')
    ax1.plot(data.index[test_start_idx:test_end_idx], test_predictions, label='Predicted Price', color='red', alpha=0.7)
    ax1.set_title('Combined Analysis: Price, Volume, and RSI\nتحلیل ترکیبی: قیمت، حجم معاملات و RSI')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)
    
    # نمودار حجم معاملات
    ax2.plot(volume_data.index, volume_data.values, 'g-', label='Trading Volume')
    ax2.set_ylabel('Volume')
    ax2.legend()
    ax2.grid(True)
    
    # نمودار RSI
    ax3.plot(rsi_data.index, rsi_data.values, 'r-', label='RSI')
    ax3.axhline(y=70, color='g', linestyle='--', alpha=0.5)
    ax3.axhline(y=30, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('RSI')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, '13_combined_price_volume_rsi.png'))
    plt.close()

    # نمودار ترکیبی 2: میانگین‌های متحرک و نوسانات
    plt.figure(figsize=(15, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # نمودار میانگین‌های متحرک
    ax1.plot(sma50.index, sma50.values, 'b-', label='SMA 50')
    ax1.plot(sma200.index, sma200.values, 'r-', label='SMA 200')
    ax1.plot(data.index[test_start_idx:test_end_idx], y_test_original, 'g-', label='Actual Price', alpha=0.5)
    ax1.set_title('Combined Analysis: Moving Averages and Volatility\nتحلیل ترکیبی: میانگین‌های متحرک و نوسانات')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)
    
    # نمودار نوسانات
    ax2.plot(volatility_data.index, volatility_data.values, 'm-', label='Volatility')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Standard Deviation')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, '14_combined_moving_averages_volatility.png'))
    plt.close()

    # نمودار ترکیبی 3: دقت پیش‌بینی و شرایط بازار
    plt.figure(figsize=(15, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # نمودار دقت پیش‌بینی
    ax1.plot(data.index[test_start_idx:test_start_idx + len(accuracy_rolling)], accuracy_rolling, 'b-', label='Prediction Accuracy')
    ax1.set_title('Combined Analysis: Prediction Accuracy and Market Conditions\nتحلیل ترکیبی: دقت پیش‌بینی و شرایط بازار')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)
    
    # نمودار شرایط بازار
    market_conditions_plot = []
    for i in range(len(y_test_original) - window_size):
        if volatility_data.iloc[i] > volatility_data.mean():
            market_conditions_plot.append(1)  # High Volatility
        elif rsi_data.iloc[i] > 70:
            market_conditions_plot.append(2)  # Overbought
        elif rsi_data.iloc[i] < 30:
            market_conditions_plot.append(3)  # Oversold
        else:
            market_conditions_plot.append(0)  # Normal
    
    ax2.plot(data.index[test_start_idx:test_start_idx + len(market_conditions_plot)], market_conditions_plot, 'r-', label='Market Condition')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Market Condition')
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['Normal', 'High Vol.', 'Overbought', 'Oversold'])
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, '15_combined_accuracy_market_conditions.png'))
    plt.close()

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
    
    # محاسبه ویژگی‌های اضافی
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    
    # محاسبه RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # محاسبه نوسانات
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    # حذف ردیف‌های با مقادیر NaN
    df = df.dropna()
    
    return df

def calculate_direction_accuracy(y_true, y_pred):
    """
    محاسبه دقت پیش‌بینی جهت حرکت قیمت
    
    Args:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    Returns:
        float: درصد دقت پیش‌بینی جهت حرکت
    """
    true_direction = np.sign(np.diff(y_true.flatten()))
    pred_direction = np.sign(np.diff(y_pred.flatten()))
    return np.mean(true_direction == pred_direction) * 100

def calculate_volatility_ratio(y_true, y_pred):
    """
    محاسبه نسبت نوسانات پیش‌بینی شده به نوسانات واقعی
    
    Args:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    Returns:
        float: نسبت نوسانات
    """
    true_volatility = np.std(np.diff(y_true.flatten()))
    pred_volatility = np.std(np.diff(y_pred.flatten()))
    return pred_volatility / true_volatility if true_volatility != 0 else 0

def calculate_sharpe_ratio(y_true, y_pred):
    """
    محاسبه نسبت شارپ
    
    Args:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    Returns:
        float: نسبت شارپ
    """
    returns = np.diff(y_pred.flatten()) / y_pred[:-1].flatten()
    if len(returns) == 0 or np.std(returns) == 0:
        return 0
    return np.mean(returns) / np.std(returns) * np.sqrt(252)  # سالانه

def calculate_win_rate(y_true, y_pred):
    """
    محاسبه نرخ برد
    
    Args:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    Returns:
        float: درصد نرخ برد
    """
    true_returns = np.diff(y_true.flatten())
    pred_returns = np.diff(y_pred.flatten())
    correct_predictions = np.sum((true_returns > 0) == (pred_returns > 0))
    return (correct_predictions / len(true_returns)) * 100

def calculate_profit_factor(y_true, y_pred):
    """
    محاسبه فاکتور سود
    
    Args:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    Returns:
        float: فاکتور سود
    """
    true_returns = np.diff(y_true.flatten())
    pred_returns = np.diff(y_pred.flatten())
    correct_predictions = (true_returns > 0) == (pred_returns > 0)
    profits = np.sum(np.abs(true_returns[correct_predictions]))
    losses = np.sum(np.abs(true_returns[~correct_predictions]))
    return profits / losses if losses != 0 else float('inf')

def predict_with_lstm(data, lag_days=10, epochs=50, batch_size=32, validation_split=0.15,
                      dropout=0.2, early_stopping_patience=10, restore_best_weights=True,
                      optimizer='adam', loss='mse', save_model=True, train_split_ratio=0.80):
    # ثبت زمان شروع
    start_time = time.time()
    
    # ایجاد نام اجرای تصادفی
    run_name = create_random_run_name()
    run_dir = os.path.join('saved_models', run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # ذخیره پارامترهای مدل
    model_params = {
        'lag_days': lag_days,
        'epochs': epochs,
        'batch_size': batch_size,
        'validation_split': validation_split,
        'dropout': dropout,
        'early_stopping_patience': early_stopping_patience,
        'restore_best_weights': restore_best_weights,
        'optimizer': optimizer,
        'loss': loss,
        'train_split_ratio': train_split_ratio,
        'run_name': run_name
    }
    
    # محاسبه train_size
    train_size = int(len(data) * train_split_ratio)
    
    data['Target'] = data['Close'].shift(-1)
    for i in range(1, lag_days + 1):
        data[f'Close_Lag_{i}'] = data['Close'].shift(i)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Daily_Change'] = (data['Close'] - data['Open']) / data['Open'] * 100
    data['RSI'] = calculate_rsi(data)
    data['Volatility'] = data['Close'].rolling(window=14).std()
    data = data.dropna()
    
    # اطمینان از اینکه شاخص از نوع datetime است
    data.index = pd.to_datetime(data.index)
    
    # تقسیم داده‌ها
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # نمایش تاریخ شروع و پایان داده‌های آموزشی و تست
    print("تاریخ شروع داده‌های آموزشی:", train_data.index[0])
    print("تاریخ پایان داده‌های آموزشی:", train_data.index[-1])
    print("تاریخ شروع داده‌های تست:", test_data.index[0])
    print("تاریخ پایان داده‌های تست:", test_data.index[-1])
    
    # آماده‌سازی ویژگی‌های ورودی و متغیر هدف
    X = data[[f'Close_Lag_{i}' for i in range(1, lag_days + 1)] + ['Volume', 'SMA_50', 'SMA_200', 'Daily_Change', 'RSI', 'Volatility']]
    y = data['Target']
    
    # نرمال‌سازی فقط روی دیتای ترن
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    # آماده‌سازی برای LSTM
    X_lstm_train = []
    y_lstm_train = []
    for i in range(lag_days, len(X_train_scaled)):
        X_lstm_train.append(X_train_scaled[i-lag_days:i])
        y_lstm_train.append(y_train_scaled[i])
    X_lstm_test = []
    y_lstm_test = []
    for i in range(lag_days, len(X_test_scaled)):
        X_lstm_test.append(X_test_scaled[i-lag_days:i])
        y_lstm_test.append(y_test_scaled[i])
    X_lstm_train, y_lstm_train = np.array(X_lstm_train), np.array(y_lstm_train)
    X_lstm_test, y_lstm_test = np.array(X_lstm_test), np.array(y_lstm_test)
    
    print(f"Length of X_lstm_train: {len(X_lstm_train)}, Length of y_lstm_train: {len(y_lstm_train)}")
    print(f"Length of X_lstm_test: {len(X_lstm_test)}, Length of y_lstm_test: {len(y_lstm_test)}")
    
    # تعریف وزن‌دهی برای داده‌های اخیر
    sample_weights = np.exp(np.linspace(0, 0.5, len(X_lstm_train)))  # کاهش شدت از 1 به 0.5
    sample_weights = sample_weights / sample_weights.mean()
    
    # معماری مدل
    model = Sequential([
        Input(shape=(lag_days, X.shape[1])),
        LSTM(50, activation='tanh', return_sequences=True),
        Dropout(0.3),
        LSTM(30, activation='tanh'),
        Dropout(0.3),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', weighted_metrics=['mse'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    history = model.fit(X_lstm_train, y_lstm_train, epochs=epochs, batch_size=128, validation_split=0.15, 
              sample_weight=sample_weights, callbacks=[early_stopping, reduce_lr], verbose=1)
    
    train_predictions = model.predict(X_lstm_train)
    test_predictions = model.predict(X_lstm_test)
    
    train_predictions = scaler_y.inverse_transform(train_predictions)
    test_predictions = scaler_y.inverse_transform(test_predictions)
    y_train_original = scaler_y.inverse_transform(y_lstm_train)
    y_test_original = scaler_y.inverse_transform(y_lstm_test)
    
    # محاسبه پیش‌بینی‌های آینده
    last_sequence = X_test_scaled[-lag_days:]
    future_predictions = predict_future_days(model, last_sequence, scaler_X, scaler_y, data)
    
    # ایجاد تاریخ‌های آینده
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
    
    # محاسبه معیارها
    train_mse = mean_squared_error(y_train_original, train_predictions)
    test_mse = mean_squared_error(y_test_original, test_predictions)
    
    # محاسبه RMSE
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    # محاسبه MAE
    train_mae = mean_absolute_error(y_train_original, train_predictions)
    test_mae = mean_absolute_error(y_test_original, test_predictions)
    
    # محاسبه MAPE
    train_mape = np.mean(np.abs((y_train_original - train_predictions) / y_train_original)) * 100
    test_mape = np.mean(np.abs((y_test_original - test_predictions) / y_test_original)) * 100
    
    # محاسبه معیارهای ارزیابی اضافی
    # محاسبه R²
    train_r2 = r2_score(y_train_original, train_predictions)
    test_r2 = r2_score(y_test_original, test_predictions)
    
    print(f"Train MSE (LSTM): {train_mse}")
    print(f"Test MSE (LSTM): {test_mse}")
    print(f"Train RMSE (LSTM): {train_rmse}")
    print(f"Test RMSE (LSTM): {test_rmse}")
    print(f"Train R2 Score (LSTM): {train_r2}")
    print(f"Test R2 Score (LSTM): {test_r2}")
    
    train_direction_accuracy = calculate_direction_accuracy(y_train_original, train_predictions)
    test_direction_accuracy = calculate_direction_accuracy(y_test_original, test_predictions)
    
    # محاسبه خطای بیشینه و میانه
    train_max_error = np.max(np.abs(y_train_original - train_predictions))
    test_max_error = np.max(np.abs(y_test_original - test_predictions))
    
    train_median_error = np.median(np.abs(y_train_original - train_predictions))
    test_median_error = np.median(np.abs(y_test_original - test_predictions))
    
    # محاسبه معیارهای ارزیابی
    train_metrics = {
        'MSE': train_mse,
        'RMSE': train_rmse,
        'MAE': train_mae,
        'MAPE': train_mape,
        'R2': train_r2,
        'Direction_Accuracy': train_direction_accuracy,
        'Max_Error': train_max_error,
        'Median_Error': train_median_error
    }
    
    test_metrics = {
        'MSE': test_mse,
        'RMSE': test_rmse,
        'MAE': test_mae,
        'MAPE': test_mape,
        'R2': test_r2,
        'Direction_Accuracy': test_direction_accuracy,
        'Max_Error': test_max_error,
        'Median_Error': test_median_error
    }
    
    # محاسبه معیارهای ارزیابی اضافی
    # محاسبه نسبت نوسانات
    train_volatility_ratio = calculate_volatility_ratio(y_train_original, train_predictions)
    test_volatility_ratio = calculate_volatility_ratio(y_test_original, test_predictions)
    
    # محاسبه نسبت شارپ
    train_sharpe_ratio = calculate_sharpe_ratio(y_train_original, train_predictions)
    test_sharpe_ratio = calculate_sharpe_ratio(y_test_original, test_predictions)
    
    # محاسبه نرخ برد
    train_win_rate = calculate_win_rate(y_train_original, train_predictions)
    test_win_rate = calculate_win_rate(y_test_original, test_predictions)
    
    # محاسبه فاکتور سود
    train_profit_factor = calculate_profit_factor(y_train_original, train_predictions)
    test_profit_factor = calculate_profit_factor(y_test_original, test_predictions)
    
    # اضافه کردن معیارهای جدید به دیکشنری‌ها
    train_metrics.update({
        'Volatility_Ratio': train_volatility_ratio,
        'Sharpe_Ratio': train_sharpe_ratio,
        'Win_Rate': train_win_rate,
        'Profit_Factor': train_profit_factor
    })
    
    test_metrics.update({
        'Volatility_Ratio': test_volatility_ratio,
        'Sharpe_Ratio': test_sharpe_ratio,
        'Win_Rate': test_win_rate,
        'Profit_Factor': test_profit_factor
    })
    
    # ذخیره مدل و اسکالرها
    if save_model:
        run_dir = save_model_and_scalers(model, scaler_X, scaler_y, model_params, run_name)
        
        # ایجاد DataFrame برای پیش‌بینی‌های آینده
        future_predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_predictions.flatten()
        })
        
        # ذخیره نمودارها و گزارش
        save_plots_and_report(run_dir, data, train_predictions, test_predictions, y_train_original, y_test_original, 
                            future_predictions, future_dates, model, scaler_X, scaler_y, model_params,
                            train_metrics, test_metrics, history)
    
    # نمایش پیش‌بینی‌های آینده
    print("\nپیش‌بینی قیمت برای ۳۰ روز آینده:")
    for date, price in zip(future_dates, future_predictions):
        print(f"{date.strftime('%Y-%m-%d')}: ${price[0]:.2f}")
    
    return model, scaler_X, scaler_y, model_params, history

if __name__ == "__main__":
    # خواندن داده‌ها از فایل CSV
    data = load_data('bitcoin_data.csv')
    
    # اجرای مدل با پارامترهای پیش‌فرض
    model, scaler_X, scaler_y, model_params, history = predict_with_lstm(
        data,
        lag_days=30,
        epochs=50,
        train_split_ratio=0.90  # 90٪ برای آموزش و 10٪ برای تست 
    )