"""
نمودارهای مربوط به پیش‌بینی‌های مدل
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_prediction_comparison(data_index, y_true, y_pred, future_dates=None, future_predictions=None, 
                             title=None, save_path=None):
    """
    رسم نمودار مقایسه قیمت واقعی و پیش‌بینی شده
    
    Args:
        data_index: ایندکس داده‌ها (تاریخ‌ها)
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
        future_dates: تاریخ‌های آینده (اختیاری)
        future_predictions: پیش‌بینی‌های آینده (اختیاری)
        title: عنوان نمودار (اختیاری)
        save_path: مسیر ذخیره نمودار (اختیاری)
    
    Returns:
        fig: شیء نمودار
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # رسم داده‌های واقعی و پیش‌بینی شده
    ax.plot(data_index, y_true, label='قیمت واقعی', color='blue')
    ax.plot(data_index, y_pred, label='قیمت پیش‌بینی شده', color='red', alpha=0.7)
    
    # اضافه کردن پیش‌بینی‌های آینده اگر موجود باشند
    if future_dates is not None and future_predictions is not None:
        ax.plot(future_dates, future_predictions, label='پیش‌بینی ۳۰ روز آینده', color='green', linestyle='--', alpha=0.8)
    
    # تنظیم عنوان و برچسب‌ها
    if title:
        ax.set_title(title)
    else:
        ax.set_title('مقایسه قیمت واقعی و پیش‌بینی شده')
    
    ax.set_xlabel('تاریخ')
    ax.set_ylabel('قیمت (دلار)')
    ax.grid(True)
    ax.legend()
    
    # تنظیم فرمت تاریخ در محور x
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"نمودار در مسیر {save_path} ذخیره شد.")
    
    return fig

def plot_future_predictions(future_dates, future_predictions, confidence_interval=0.05, title=None, save_path=None):
    """
    رسم نمودار پیش‌بینی‌های آینده با محدوده اطمینان
    
    Args:
        future_dates: تاریخ‌های آینده
        future_predictions: پیش‌بینی‌های آینده
        confidence_interval: محدوده اطمینان به صورت درصد (پیش‌فرض: ۵٪)
        title: عنوان نمودار (اختیاری)
        save_path: مسیر ذخیره نمودار (اختیاری)
    
    Returns:
        fig: شیء نمودار
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    future_predictions = np.array(future_predictions).flatten()
    
    # رسم پیش‌بینی‌ها
    ax.plot(future_dates, future_predictions, 'b-', label='پیش‌بینی قیمت')
    
    # اضافه کردن محدوده اطمینان
    ax.fill_between(future_dates, 
                   future_predictions * (1 - confidence_interval),
                   future_predictions * (1 + confidence_interval),
                   alpha=0.2, color='blue', label=f'محدوده اطمینان {confidence_interval*100}٪')
    
    # تنظیم عنوان و برچسب‌ها
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'پیش‌بینی قیمت بیت‌کوین برای {len(future_dates)} روز آینده')
    
    ax.set_xlabel('تاریخ')
    ax.set_ylabel('قیمت (دلار)')
    ax.grid(True)
    ax.legend()
    
    # تنظیم فرمت تاریخ در محور x
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"نمودار در مسیر {save_path} ذخیره شد.")
    
    return fig

def plot_loss_history(history, title=None, save_path=None):
    """
    رسم نمودار تابع loss در طول آموزش
    
    Args:
        history: تاریخچه آموزش مدل
        title: عنوان نمودار (اختیاری)
        save_path: مسیر ذخیره نمودار (اختیاری)
    
    Returns:
        fig: شیء نمودار
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.plot(history.history['loss'], label='Loss آموزش')
    
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Loss اعتبارسنجی')
    
    # تنظیم عنوان و برچسب‌ها
    if title:
        ax.set_title(title)
    else:
        ax.set_title('نمودار تابع Loss در طول آموزش')
    
    ax.set_xlabel('دوره (Epoch)')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"نمودار در مسیر {save_path} ذخیره شد.")
    
    return fig

def plot_error_distribution(y_true, y_pred, bins=50, title=None, save_path=None):
    """
    رسم نمودار توزیع خطاهای پیش‌بینی
    
    Args:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
        bins: تعداد ستون‌های هیستوگرام
        title: عنوان نمودار (اختیاری)
        save_path: مسیر ذخیره نمودار (اختیاری)
    
    Returns:
        fig: شیء نمودار
    """
    # تبدیل به آرایه‌های یک بعدی
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # محاسبه خطاها
    errors = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # رسم هیستوگرام خطاها
    n, bins, patches = ax.hist(errors, bins=bins, alpha=0.7, color='skyblue')
    
    # اضافه کردن خط میانگین خطا
    mean_error = np.mean(errors)
    ax.axvline(x=mean_error, color='red', linestyle='--', label=f'میانگین خطا: {mean_error:.2f}')
    
    # تنظیم عنوان و برچسب‌ها
    if title:
        ax.set_title(title)
    else:
        ax.set_title('توزیع خطاهای پیش‌بینی')
    
    ax.set_xlabel('خطا (قیمت واقعی - پیش‌بینی)')
    ax.set_ylabel('تعداد')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"نمودار در مسیر {save_path} ذخیره شد.")
    
    return fig

def plot_daily_changes(predictions, title=None, save_path=None):
    """
    رسم نمودار تغییرات روزانه پیش‌بینی شده
    
    Args:
        predictions: پیش‌بینی‌های قیمت
        title: عنوان نمودار (اختیاری)
        save_path: مسیر ذخیره نمودار (اختیاری)
    
    Returns:
        fig: شیء نمودار
    """
    # تبدیل به آرایه یک بعدی
    predictions = np.array(predictions).flatten()
    
    # محاسبه تغییرات روزانه
    daily_changes = np.diff(predictions) / predictions[:-1] * 100
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # رسم نمودار میله‌ای تغییرات روزانه
    ax.bar(range(len(daily_changes)), daily_changes, alpha=0.7, color='skyblue')
    
    # اضافه کردن خط صفر
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    
    # تنظیم عنوان و برچسب‌ها
    if title:
        ax.set_title(title)
    else:
        ax.set_title('تغییرات روزانه پیش‌بینی شده')
    
    ax.set_xlabel('روز')
    ax.set_ylabel('تغییرات روزانه (٪)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"نمودار در مسیر {save_path} ذخیره شد.")
    
    return fig

def create_all_prediction_plots(data_index, y_true, y_pred, future_dates=None, future_predictions=None, 
                              history=None, output_dir=None):
    """
    ایجاد و ذخیره تمام نمودارهای پیش‌بینی در یک دایرکتوری
    
    Args:
        data_index: ایندکس داده‌ها (تاریخ‌ها)
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
        future_dates: تاریخ‌های آینده (اختیاری)
        future_predictions: پیش‌بینی‌های آینده (اختیاری)
        history: تاریخچه آموزش مدل (اختیاری)
        output_dir: مسیر دایرکتوری خروجی (اختیاری)
    """
    if output_dir is None:
        output_dir = 'prediction_plots'
    
    # ایجاد دایرکتوری خروجی اگر وجود نداشته باشد
    os.makedirs(output_dir, exist_ok=True)
    
    # نمودار مقایسه قیمت واقعی و پیش‌بینی شده
    plot_prediction_comparison(
        data_index, y_true, y_pred, 
        future_dates=future_dates, 
        future_predictions=future_predictions,
        save_path=os.path.join(output_dir, '01_prediction_comparison.png')
    )
    
    # نمودار توزیع خطاهای پیش‌بینی
    plot_error_distribution(
        y_true, y_pred,
        save_path=os.path.join(output_dir, '02_error_distribution.png')
    )
    
    # نمودار تابع loss اگر تاریخچه آموزش موجود باشد
    if history is not None:
        plot_loss_history(
            history,
            save_path=os.path.join(output_dir, '03_loss_history.png')
        )
    
    # نمودار پیش‌بینی‌های آینده اگر موجود باشند
    if future_dates is not None and future_predictions is not None:
        plot_future_predictions(
            future_dates, 
            future_predictions,
            save_path=os.path.join(output_dir, '04_future_predictions.png')
        )
        
        # نمودار تغییرات روزانه پیش‌بینی شده
        plot_daily_changes(
            future_predictions,
            save_path=os.path.join(output_dir, '05_daily_changes.png')
        )
    
    print(f"تمام نمودارهای پیش‌بینی در مسیر {output_dir} ذخیره شدند.") 