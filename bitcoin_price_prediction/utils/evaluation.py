"""
توابع ارزیابی مدل برای پروژه پیش‌بینی قیمت بیت‌کوین
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
from .data_utils import inverse_transform_predictions

def calculate_metrics(y_true, y_pred):
    """
    محاسبه متریک‌های مختلف ارزیابی
    
    Args:
        y_true (numpy.array): مقادیر واقعی
        y_pred (numpy.array): مقادیر پیش‌بینی شده
        
    Returns:
        dict: دیکشنری حاوی متریک‌های مختلف
    """
    # محاسبه خطای میانگین مربعات
    mse = mean_squared_error(y_true, y_pred)
    
    # محاسبه جذر خطای میانگین مربعات
    rmse = np.sqrt(mse)
    
    # محاسبه میانگین خطای مطلق
    mae = mean_absolute_error(y_true, y_pred)
    
    # محاسبه درصد میانگین خطای مطلق
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # محاسبه ضریب تعیین
    r2 = r2_score(y_true, y_pred)
    
    # ایجاد دیکشنری نتایج
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }
    
    return metrics

def evaluate_model(model, X_test, y_test, scaler, return_predictions=True):
    """
    ارزیابی مدل با استفاده از داده‌های تست
    
    Args:
        model: مدل آموزش دیده
        X_test (numpy.array): داده‌های تست ورودی
        y_test (numpy.array): مقادیر واقعی تست
        scaler: مقیاس‌کننده استفاده شده در آماده‌سازی داده‌ها
        return_predictions (bool): آیا مقادیر پیش‌بینی شده نیز بازگردانده شوند
        
    Returns:
        tuple: (metrics, predictions) اگر return_predictions=True، در غیر این صورت فقط metrics
    """
    # انجام پیش‌بینی
    predictions = model.predict(X_test)
    
    # تبدیل پیش‌بینی‌ها و مقادیر واقعی به مقیاس اصلی
    predictions_original = inverse_transform_predictions(predictions, scaler)
    y_test_original = inverse_transform_predictions(y_test.reshape(-1, 1), scaler)
    
    # محاسبه متریک‌ها
    metrics = calculate_metrics(y_test_original, predictions_original)
    
    # نمایش متریک‌ها
    print("\nنتایج ارزیابی مدل:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    if return_predictions:
        return metrics, predictions_original, y_test_original
    else:
        return metrics

def save_metrics(metrics, filepath):
    """
    ذخیره متریک‌های ارزیابی در فایل JSON
    
    Args:
        metrics (dict): متریک‌های ارزیابی
        filepath (str): مسیر فایل JSON برای ذخیره‌سازی
    """
    # اطمینان از وجود دایرکتوری
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # تبدیل numpy.float64 به float استاندارد برای سازگاری با JSON
    metrics_json = {k: float(v) for k, v in metrics.items()}
    
    # ذخیره متریک‌ها در فایل JSON
    with open(filepath, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    print(f"متریک‌های ارزیابی در مسیر {filepath} ذخیره شدند.") 