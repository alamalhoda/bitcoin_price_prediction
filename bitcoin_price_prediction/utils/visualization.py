"""
توابع رسم نمودار برای پروژه پیش‌بینی قیمت بیت‌کوین
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib as mpl

# تنظیم فونت فارسی
try:
    mpl.rc('font', family='sans-serif')
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def plot_training_history(history, output_path=None):
    """
    رسم نمودار تاریخچه آموزش مدل
    
    Args:
        history: تاریخچه آموزش مدل
        output_path (str, optional): مسیر ذخیره نمودار
    """
    plt.figure(figsize=(10, 6))
    
    # رسم نمودار loss در مجموعه آموزش و اعتبارسنجی
    plt.plot(history.history['loss'], label='خطای آموزش')
    plt.plot(history.history['val_loss'], label='خطای اعتبارسنجی')
    
    plt.title('روند خطای مدل در طول آموزش')
    plt.ylabel('خطا (MSE)')
    plt.xlabel('تعداد دوره (Epoch)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # بهینه‌سازی نمایش
    plt.tight_layout()
    
    # ذخیره نمودار اگر مسیر مشخص شده باشد
    if output_path:
        # اطمینان از وجود دایرکتوری
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"نمودار تاریخچه آموزش در مسیر {output_path} ذخیره شد.")
    
    plt.show()

def plot_predictions(test_dates, y_test, predictions, output_path=None):
    """
    رسم نمودار مقایسه مقادیر واقعی و پیش‌بینی شده
    
    Args:
        test_dates (list/array): تاریخ‌های داده‌های تست
        y_test (numpy.array): مقادیر واقعی
        predictions (numpy.array): مقادیر پیش‌بینی شده
        output_path (str, optional): مسیر ذخیره نمودار
    """
    plt.figure(figsize=(12, 6))
    
    # رسم مقادیر واقعی
    plt.plot(test_dates, y_test, label='مقادیر واقعی', color='blue', linewidth=2)
    
    # رسم مقادیر پیش‌بینی شده
    plt.plot(test_dates, predictions, label='مقادیر پیش‌بینی شده', color='red', linestyle='--', linewidth=2)
    
    plt.title('مقایسه قیمت واقعی و پیش‌بینی شده بیت‌کوین')
    plt.xlabel('تاریخ')
    plt.ylabel('قیمت (دلار)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # تنظیم محور X برای نمایش بهتر تاریخ‌ها
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()  # چرخش برچسب‌های تاریخ
    
    # بهینه‌سازی نمایش
    plt.tight_layout()
    
    # ذخیره نمودار اگر مسیر مشخص شده باشد
    if output_path:
        # اطمینان از وجود دایرکتوری
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"نمودار پیش‌بینی در مسیر {output_path} ذخیره شد.")
    
    plt.show()

def plot_future_predictions(future_dates, future_predictions, historical_dates=None, historical_prices=None, output_path=None):
    """
    رسم نمودار پیش‌بینی‌های آینده
    
    Args:
        future_dates (list): تاریخ‌های آینده
        future_predictions (numpy.array): پیش‌بینی‌های قیمت برای روزهای آینده
        historical_dates (list, optional): تاریخ‌های تاریخی
        historical_prices (numpy.array, optional): قیمت‌های تاریخی
        output_path (str, optional): مسیر ذخیره نمودار
    """
    plt.figure(figsize=(12, 6))
    
    # رسم داده‌های تاریخی اگر موجود باشند
    if historical_dates is not None and historical_prices is not None:
        plt.plot(historical_dates, historical_prices, label='قیمت‌های تاریخی', color='blue', linewidth=2)
    
    # رسم پیش‌بینی‌های آینده
    plt.plot(future_dates, future_predictions, label='پیش‌بینی آینده', color='red', linestyle='--', linewidth=2, marker='o')
    
    # ناحیه سایه برای پیش‌بینی‌های آینده
    plt.fill_between(future_dates, 
                    future_predictions * 0.9,  # حد پایین (10% کمتر)
                    future_predictions * 1.1,  # حد بالا (10% بیشتر)
                    color='red', alpha=0.2)
    
    plt.title('پیش‌بینی قیمت بیت‌کوین برای روزهای آینده')
    plt.xlabel('تاریخ')
    plt.ylabel('قیمت (دلار)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # تنظیم محور X برای نمایش بهتر تاریخ‌ها
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()  # چرخش برچسب‌های تاریخ
    
    # بهینه‌سازی نمایش
    plt.tight_layout()
    
    # ذخیره نمودار اگر مسیر مشخص شده باشد
    if output_path:
        # اطمینان از وجود دایرکتوری
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"نمودار پیش‌بینی آینده در مسیر {output_path} ذخیره شد.")
    
    plt.show()

def plot_metrics(model_metrics, output_path=None):
    """
    رسم نمودار مقایسه متریک‌های مدل
    
    Args:
        model_metrics (dict): دیکشنری حاوی متریک‌های مدل
        output_path (str, optional): مسیر ذخیره نمودار
    """
    # استخراج متریک‌ها
    metrics = list(model_metrics.keys())
    values = list(model_metrics.values())
    
    # رسم نمودار متریک‌ها
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
    
    # افزودن برچسب مقادیر روی نمودار
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.title('متریک‌های ارزیابی مدل')
    plt.ylabel('مقدار')
    plt.xlabel('متریک')
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # بهینه‌سازی نمایش
    plt.tight_layout()
    
    # ذخیره نمودار اگر مسیر مشخص شده باشد
    if output_path:
        # اطمینان از وجود دایرکتوری
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"نمودار متریک‌ها در مسیر {output_path} ذخیره شد.")
    
    plt.show() 