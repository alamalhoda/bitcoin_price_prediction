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
- ۸۸٪ داده‌ها برای آموزش
- ۱۲٪ داده‌ها برای تست
- داده‌ها به صورت زمانی مرتب شده‌اند
- از MinMaxScaler برای نرمال‌سازی داده‌ها استفاده می‌شود

نحوه وزن‌دهی داده‌ها:
- وزن‌های خطی از ۱ تا ۲ برای داده‌های اخیر
- نرمال‌سازی وزن‌ها برای حفظ تعادل در آموزش
- تأکید بیشتر روی داده‌های اخیر برای پیش‌بینی دقیق‌تر

معماری مدل:
- لایه ورودی: دریافت توالی زمانی با طول lag_days
- لایه LSTM اول: ۵۰ نورون با return_sequences=True
- Dropout (0.3) برای جلوگیری از overfitting
- لایه LSTM دوم: ۳۰ نورون
- Dropout (0.2)
- لایه خروجی: یک نورون برای پیش‌بینی قیمت

پارامترهای قابل تنظیم:
- lag_days: تعداد روزهای گذشته برای پیش‌بینی (پیش‌فرض: ۱۰)
- epochs: تعداد دوره‌های آموزش (پیش‌فرض: ۳۰)
- batch_size: اندازه batch برای آموزش (پیش‌فرض: ۳۲)
- validation_split: نسبت داده‌های validation (پیش‌فرض: ۰.۱)
- lstm1_units: تعداد نورون‌های لایه LSTM اول (پیش‌فرض: ۵۰)
- lstm2_units: تعداد نورون‌های لایه LSTM دوم (پیش‌فرض: ۳۰)
- dropout1: نرخ dropout لایه اول (پیش‌فرض: ۰.۳)
- dropout2: نرخ dropout لایه دوم (پیش‌فرض: ۰.۲)
- early_stopping_patience: تعداد epoch‌های صبر برای early stopping (پیش‌فرض: ۱۰)
- restore_best_weights: ذخیره بهترین وزن‌ها (پیش‌فرض: True)
- optimizer: بهینه‌ساز مورد استفاده (پیش‌فرض: 'adam')
- loss: تابع loss مورد استفاده (پیش‌فرض: 'mse')

Early Stopping:
- نظارت بر validation loss
- patience=10 (توقف اگر validation loss در ۱۰ epoch بهبود نیابد)
- restore_best_weights=True (بازگشت به بهترین وزن‌ها)

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
- وزن‌دهی داده‌های اخیر برای بهبود دقت پیش‌بینی‌های کوتاه‌مدت
- امکان تنظیم پارامترهای مختلف برای بهینه‌سازی عملکرد
- ارزیابی جامع مدل با معیارهای مختلف برای اطمینان از عملکرد مناسب
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
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
                           train_metrics, test_metrics):
    """ذخیره نمودارها و گزارش ارزیابی مدل"""
    # ثبت زمان شروع
    start_time = time.time()
    
    # ایجاد دایرکتوری برای ذخیره نمودارها
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # ذخیره نمودارها
    plot_additional_analysis(data, train_predictions, test_predictions, y_train_original, y_test_original,
                           future_predictions, future_dates, len(y_train_original), model_params['lag_days'], plots_dir)
    
    # ذخیره گزارش
    report_path = os.path.join(run_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("گزارش ارزیابی مدل پیش‌بینی قیمت بیت‌کوین\n")
        f.write("==================================================\n\n")
        
        # اطلاعات دیتاست
        f.write("اطلاعات دیتاست:\n")
        f.write("-"*30 + "\n\n")
        f.write(f"تاریخ و ساعت شروع داده‌ها: {data.index[0]}\n")
        f.write(f"تاریخ و ساعت پایان داده‌ها: {data.index[-1]}\n")
        f.write(f"تعداد کل رکوردها: {len(data)}\n")
        f.write(f"تعداد ویژگی‌های اصلی: {len(data.columns)}\n")
        f.write(f"فرکانس داده‌ها: ساعتی\n")
        f.write(f"تعداد روزهای داده: {len(data) // 24}\n")
        f.write(f"تعداد هفته‌های داده: {len(data) // (24 * 7)}\n")
        f.write(f"تعداد ماه‌های داده: {len(data) // (24 * 30)}\n\n")
        
        f.write("ویژگی‌های موجود:\n")
        f.write("- ویژگی‌های قیمت: Close, Open, High, Low\n")
        f.write("- ویژگی‌های حجم: Volume, Volume_Ratio\n")
        f.write("- ویژگی‌های زمانی: Hour, Day_of_Week, Month, Is_Weekend\n")
        f.write("- میانگین‌های متحرک: SMA_24, SMA_50, SMA_200\n")
        f.write("- شاخص‌های تکنیکال: RSI, Volatility_12h, Volatility_24h\n")
        f.write("- ویژگی‌های مشتق شده: Hourly_Return, Daily_Return, Price_Gap, Price_Range\n\n")
        
        # اطلاعات تقسیم داده‌ها
        f.write("اطلاعات تقسیم داده‌ها:\n")
        f.write("-"*30 + "\n\n")
        train_size = int(len(data) * 0.88)
        f.write(f"تعداد داده‌های آموزشی: {train_size} ({88}%)\n")
        f.write(f"تعداد داده‌های تست: {len(data) - train_size} ({12}%)\n")
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

def predict_future_days(model, last_sequence, scaler_X, scaler_y, future_hours=24):
    """
    پیش‌بینی قیمت‌های آینده
    
    Args:
        model: مدل آموزش دیده
        last_sequence: آخرین توالی داده‌ها
        scaler_X: مقیاس‌کننده ویژگی‌ها
        scaler_y: مقیاس‌کننده هدف
        future_hours: تعداد ساعت‌های آینده برای پیش‌بینی
    
    Returns:
        future_predictions: آرایه پیش‌بینی‌های آینده
        future_dates: آرایه تاریخ‌های آینده
    """
    print(f"\nپیش‌بینی قیمت‌ها برای {future_hours} ساعت آینده...")
    
    # تبدیل پیش‌بینی‌ها به مقیاس اصلی
    last_sequence_orig = scaler_X.inverse_transform(last_sequence.reshape(-1, last_sequence.shape[-1]))
    
    # ایجاد آرایه‌های خالی برای ذخیره نتایج
    future_predictions = []
    future_dates = []
    
    # آخرین تاریخ در داده‌ها
    last_date = pd.to_datetime(last_sequence_orig[-1, 0])
    
    # پیش‌بینی برای هر ساعت آینده
    for i in range(future_hours):
        # پیش‌بینی قیمت بعدی
        next_pred = model.predict(last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1]))
        next_pred_orig = scaler_y.inverse_transform(next_pred)
        
        # اضافه کردن پیش‌بینی به لیست
        future_predictions.append(next_pred_orig[0][0])
        
        # محاسبه تاریخ بعدی
        next_date = last_date + pd.Timedelta(hours=1)
        future_dates.append(next_date)
        
        # به‌روزرسانی توالی برای پیش‌بینی بعدی
        new_row = np.zeros((1, last_sequence.shape[-1]))
        new_row[0, 0] = next_pred_orig[0][0]  # قیمت بسته شدن
        new_row[0, 1] = next_pred_orig[0][0] * 1.001  # قیمت بالا
        new_row[0, 2] = next_pred_orig[0][0] * 0.999  # قیمت پایین
        new_row[0, 3] = next_pred_orig[0][0]  # قیمت باز شدن
        new_row[0, 4] = last_sequence_orig[-1, 4]  # حجم (همان حجم آخر)
        
        # اضافه کردن ویژگی‌های زمانی
        new_row[0, 5] = next_date.hour
        new_row[0, 6] = next_date.dayofweek
        new_row[0, 7] = next_date.month
        new_row[0, 8] = 1 if next_date.dayofweek >= 5 else 0
        
        # محاسبه سایر ویژگی‌ها
        new_row[0, 9] = (new_row[0, 0] - last_sequence_orig[-1, 0]) / last_sequence_orig[-1, 0]  # تغییر ساعتی
        new_row[0, 10] = (new_row[0, 0] - last_sequence_orig[-24, 0]) / last_sequence_orig[-24, 0]  # تغییر روزانه
        
        # محاسبه میانگین‌های متحرک
        new_row[0, 11] = np.mean(last_sequence_orig[-24:, 0])  # SMA_24
        new_row[0, 12] = np.mean(last_sequence_orig[-50:, 0])  # SMA_50
        new_row[0, 13] = np.mean(last_sequence_orig[-200:, 0])  # SMA_200
        
        # محاسبه RSI
        new_row[0, 14] = calculate_rsi_for_row(new_row[0], last_sequence_orig[-1])
        
        # محاسبه نوسانات
        new_row[0, 15] = np.std(last_sequence_orig[-12:, 0])  # Volatility_12h
        new_row[0, 16] = np.std(last_sequence_orig[-24:, 0])  # Volatility_24h
        
        # محاسبه شکاف قیمتی
        new_row[0, 17] = new_row[0, 3] - last_sequence_orig[-1, 0]  # Price_Gap
        
        # محاسبه نسبت حجم
        new_row[0, 18] = new_row[0, 4] / np.mean(last_sequence_orig[-24:, 4])  # Volume_Ratio
        
        # محاسبه دامنه قیمت
        new_row[0, 19] = new_row[0, 1] - new_row[0, 2]  # Price_Range
        
        # نرمال‌سازی ردیف جدید
        new_row_scaled = scaler_X.transform(new_row)
        
        # به‌روزرسانی توالی
        last_sequence = np.vstack([last_sequence[1:], new_row_scaled])
        last_sequence_orig = np.vstack([last_sequence_orig[1:], new_row])
        last_date = next_date
    
    return np.array(future_predictions), np.array(future_dates)

def calculate_rsi_for_row(row, last_row, periods=14):
    """
    محاسبه RSI برای یک ردیف داده
    """
    # این تابع باید با توجه به داده‌های تاریخی RSI را محاسبه کند
    # برای سادگی، از مقدار میانگین استفاده می‌کنیم
    return 50.0  # مقدار پیش‌فرض

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
    print(f"- لایه LSTM: 32 نورون با تابع فعال‌سازی tanh")
    print(f"- Dropout (0.2)")
    print("- لایه خروجی: یک نورون\n")
    
    print("پارامترهای آموزش:")
    print(f"- تعداد روزهای گذشته (lag_days): {model_params['lag_days']}")
    print(f"- تعداد دوره‌های آموزش (epochs): {model_params['epochs']}")
    print(f"- اندازه batch: {model_params['batch_size']}")
    print(f"- نسبت validation: {model_params['validation_split']}")
    print(f"- بهینه‌ساز: {model_params['optimizer']}")
    print(f"- تابع loss: {model_params['loss']}")
    print(f"- Early Stopping: patience={model_params['early_stopping_patience']}, restore_best_weights=True\n")
    
    print("==================================================\n")

def plot_additional_analysis(data, train_predictions, test_predictions, y_train_original, y_test_original,
                           future_predictions, future_dates, train_size, lag_days, run_dir):
    """
    رسم نمودارهای اضافی برای تحلیل پیش‌بینی‌ها و ذخیره آنها در مسیر اجرای فعلی
    """
    # نمودار 1: مقایسه قیمت واقعی و پیش‌بینی شده
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_original, label='Actual Price')
    plt.plot(test_predictions, label='Predicted Price')
    plt.fill_between(range(len(y_test_original)), 
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
    plt.plot(accuracy_rolling)
    plt.title('Prediction Accuracy Over Time\nدقت پیش‌بینی در مقیاس زمانی')
    plt.xlabel('Time')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(run_dir, '06_accuracy_over_time.png'))
    plt.close()
    
    # نمودار 7: تحلیل حجم معاملات
    plt.figure(figsize=(14, 6))
    volume_data = data['Volume'].iloc[train_size + lag_days:train_size + lag_days + len(y_test_original)]
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
    rsi_data = data['RSI'].iloc[train_size + lag_days:train_size + lag_days + len(y_test_original)]
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
    sma50 = data['SMA_50'].iloc[train_size + lag_days:train_size + lag_days + len(y_test_original)]
    sma200 = data['SMA_200'].iloc[train_size + lag_days:train_size + lag_days + len(y_test_original)]
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
    volatility_data = data['Volatility'].iloc[train_size + lag_days:train_size + lag_days + len(y_test_original)]
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
    ax1.plot(y_test_original, label='Actual Price', color='blue')
    ax1.plot(test_predictions, label='Predicted Price', color='red', alpha=0.7)
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
    ax1.plot(y_test_original, 'g-', label='Actual Price', alpha=0.5)
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
    ax1.plot(accuracy_rolling, 'b-', label='Prediction Accuracy')
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
    
    ax2.plot(market_conditions_plot, 'r-', label='Market Condition')
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
    بارگذاری و پیش‌پردازش داده‌های ساعتی
    
    Args:
        file_path: مسیر فایل CSV
        
    Returns:
        DataFrame: داده‌های پیش‌پردازش شده
    """
    print(f"\nخواندن داده‌ها از فایل: {file_path}")
    
    try:
        # خواندن فایل CSV با رد کردن دو سطر دوم و سوم
    df = pd.read_csv(file_path, skiprows=[1, 2])
        print(f"شکل اولیه داده‌ها: {df.shape}")
        print(f"ستون‌های اولیه: {df.columns.tolist()}\n")
        
        # حذف ستون Daily Return چون خالی است
        df = df.drop('Daily Return', axis=1)
        
        # تبدیل ستون Price به datetime و تنظیم به عنوان ایندکس
        df['Price'] = pd.to_datetime(df['Price'])
        df.set_index('Price', inplace=True)
        df.index.name = 'Date'
    
    # تبدیل ستون‌ها به عددی
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
        print("نمونه‌ای از داده‌های اولیه:")
        print(df.head(), "\n")
        
        print("اضافه کردن ویژگی‌های جدید...")
        
        # اضافه کردن ویژگی‌های زمانی
        df['Hour'] = df.index.hour
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Is_Weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        # محاسبه تغییرات ساعتی و روزانه
        df['Hourly_Return'] = df['Close'].pct_change()
        df['Daily_Return'] = df['Close'].pct_change(24)  # 24 ساعت برای تغییرات روزانه
        
        # محاسبه میانگین‌های متحرک
        df['SMA_24'] = df['Close'].rolling(window=24).mean()  # میانگین متحرک 24 ساعته
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # محاسبه RSI
        df['RSI'] = calculate_rsi(df, periods=14)
    
    # محاسبه نوسانات
        df['Volatility_12h'] = df['Hourly_Return'].rolling(window=12).std()
        df['Volatility_24h'] = df['Hourly_Return'].rolling(window=24).std()
        
        # محاسبه شکاف قیمتی
        df['Price_Gap'] = df['Open'] - df['Close'].shift(1)
        
        # محاسبه نسبت حجم
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=24).mean()
        
        # محاسبه دامنه قیمت
        df['Price_Range'] = df['High'] - df['Low']
        
        print("\nقبل از حذف NaN:")
        print(f"تعداد رکوردها: {len(df)}")
        print(f"تعداد ویژگی‌ها: {len(df.columns)}")
        print("تعداد NaN در هر ستون:")
        print(df.isna().sum())
    
    # حذف ردیف‌های با مقادیر NaN
        df_cleaned = df.dropna()
        
        if len(df_cleaned) == 0:
            raise ValueError("پس از حذف NaN هیچ داده‌ای باقی نمانده است!")
        
        print("\nپس از حذف NaN:")
        print(f"تعداد رکوردها: {len(df_cleaned)}")
        print(f"تعداد ویژگی‌ها: {len(df_cleaned.columns)}")
        print(f"بازه زمانی: از {df_cleaned.index[0]} تا {df_cleaned.index[-1]}\n")
        
        return df_cleaned
        
    except Exception as e:
        print(f"خطا در خواندن یا پردازش داده‌ها: {str(e)}")
        raise

def predict_with_lstm(data, lag_days=24, epochs=100, batch_size=32, validation_split=0.1,
                     dropout=0.2, early_stopping_patience=10, restore_best_weights=True,
                     optimizer='adam', loss='mse', save_model=True):
    """
    پیش‌بینی قیمت بیت‌کوین با استفاده از LSTM
    
    Args:
        data: DataFrame داده‌های پیش‌پردازش شده
        lag_days: تعداد روزهای قبل برای پیش‌بینی
        epochs: تعداد دوره‌های آموزش
        batch_size: اندازه دسته
        validation_split: نسبت داده‌های اعتبارسنجی
        dropout: نرخ dropout
        early_stopping_patience: تعداد دوره‌های صبر برای early stopping
        restore_best_weights: بازگرداندن بهترین وزن‌ها
        optimizer: بهینه‌ساز
        loss: تابع خطا
        save_model: ذخیره مدل
    """
    # تنظیم پارامترهای مدل
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
        'save_model': save_model
    }
    
    print("\nبررسی داده‌های ورودی:")
    print(f"شکل داده‌ها: {data.shape}")
    print(f"ستون‌های موجود: {data.columns.tolist()}")
    print("نمونه‌ای از داده‌ها:")
    print(data.head())
    
    # جداسازی ویژگی‌ها و هدف
    X = data.drop(['Close'], axis=1)
    y = data['Close']
    
    # تقسیم داده‌ها به آموزش و تست
    X_train, X_test = X[:int(len(X) * 0.88)], X[int(len(X) * 0.88):]
    y_train, y_test = y[:int(len(y) * 0.88)], y[int(len(y) * 0.88):]
    
    # نرمال‌سازی داده‌ها
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
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
    
    # معماری مدل
    model = Sequential([
        Input(shape=(lag_days, X.shape[1])),
        LSTM(256, activation='tanh', return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(128, activation='tanh', return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, activation='tanh', return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, activation='tanh',
             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(1)
    ])
    
    # کامپایل مدل با نرخ یادگیری پایین‌تر
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=loss)
    
    # تنظیم Early Stopping و ReduceLROnPlateau
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=restore_best_weights
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # آموزش مدل
    history = model.fit(
        X_lstm_train, y_lstm_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # پیش‌بینی
    train_predictions = model.predict(X_lstm_train)
    test_predictions = model.predict(X_lstm_test)
    
    # تبدیل پیش‌بینی‌ها به مقیاس اصلی
    train_predictions = scaler_y.inverse_transform(train_predictions)
    test_predictions = scaler_y.inverse_transform(test_predictions)
    y_train_original = scaler_y.inverse_transform(y_lstm_train)
    y_test_original = scaler_y.inverse_transform(y_lstm_test)
    
    # محاسبه معیارهای ارزیابی
    train_metrics = {
        'mse': mean_squared_error(y_train_original, train_predictions),
        'rmse': np.sqrt(mean_squared_error(y_train_original, train_predictions)),
        'mae': mean_absolute_error(y_train_original, train_predictions),
        'r2': r2_score(y_train_original, train_predictions)
    }
    
    test_metrics = {
        'mse': mean_squared_error(y_test_original, test_predictions),
        'rmse': np.sqrt(mean_squared_error(y_test_original, test_predictions)),
        'mae': mean_absolute_error(y_test_original, test_predictions),
        'r2': r2_score(y_test_original, test_predictions)
    }
    
    print("\nنتایج ارزیابی مدل:")
    print("معیارهای آموزش:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nمعیارهای تست:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    if save_model:
        run_name = create_random_run_name()
        save_model_and_scalers(model, scaler_X, scaler_y, model_params, run_name)
        
        # پیش‌بینی آینده
        last_sequence = X_lstm_test[-1:]
        future_predictions, future_dates = predict_future_days(model, last_sequence, scaler_X, scaler_y)
        
        # ذخیره نمودارها و گزارش
        save_plots_and_report(
            f'saved_models/{run_name}',
            data,
            train_predictions,
            test_predictions,
            y_train_original,
            y_test_original,
            future_predictions,
            future_dates,
            model,
            scaler_X,
            scaler_y,
            model_params,
            train_metrics,
            test_metrics
        )
    
    return model, scaler_X, scaler_y, model_params

if __name__ == "__main__":
    # خواندن داده‌ها از فایل CSV
    data = load_data('bitcoin_20170101_20250310_1h.csv')
    
    # اجرای مدل با پارامترهای جدید
    model, scaler_X, scaler_y, model_params = predict_with_lstm(
        data,
        lag_days=24,  # یک روز
        epochs=5,    # کاهش تعداد اپوک‌ها
        batch_size=32,
        validation_split=0.1,
        dropout=0.3,  # افزایش نرخ dropout
        early_stopping_patience=10,
        restore_best_weights=True,
        optimizer='adam',
        loss='mse',
        save_model=True
    )