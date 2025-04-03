"""
معیارهای ارزیابی برای مدل‌های پیش‌بینی سری زمانی
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_rmse(y_true, y_pred):
    """
    محاسبه جذر میانگین مربعات خطا (RMSE)
    
    Args:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    Returns:
        float: جذر میانگین مربعات خطا
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    """
    محاسبه میانگین درصد خطای مطلق (MAPE)
    
    Args:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    Returns:
        float: میانگین درصد خطای مطلق
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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

def calculate_max_error(y_true, y_pred):
    """
    محاسبه حداکثر خطای مطلق
    
    Args:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    Returns:
        float: حداکثر خطای مطلق
    """
    return np.max(np.abs(y_true - y_pred))

def calculate_median_error(y_true, y_pred):
    """
    محاسبه خطای میانه
    
    Args:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    Returns:
        float: خطای میانه
    """
    return np.median(np.abs(y_true - y_pred))

def calculate_all_metrics(y_true, y_pred):
    """
    محاسبه تمام معیارهای ارزیابی
    
    Args:
        y_true: مقادیر واقعی
        y_pred: مقادیر پیش‌بینی شده
    
    Returns:
        dict: دیکشنری شامل تمام معیارهای ارزیابی
    """
    # تبدیل به آرایه‌های یک بعدی
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Direction_Accuracy': calculate_direction_accuracy(y_true, y_pred),
        'Max_Error': calculate_max_error(y_true, y_pred),
        'Median_Error': calculate_median_error(y_true, y_pred),
        'Volatility_Ratio': calculate_volatility_ratio(y_true, y_pred),
        'Sharpe_Ratio': calculate_sharpe_ratio(y_true, y_pred),
        'Win_Rate': calculate_win_rate(y_true, y_pred),
        'Profit_Factor': calculate_profit_factor(y_true, y_pred)
    }
    
    return metrics

def print_metrics(metrics, title='Metrics'):
    """
    چاپ معیارهای ارزیابی
    
    Args:
        metrics: دیکشنری شامل معیارهای ارزیابی
        title: عنوان برای چاپ
    """
    print(f"\n{title}:")
    print("-"*40)
    
    print("۱. معیارهای خطای پایه:")
    print("-"*30)
    print(f"MSE (خطای میانگین مربعات): {metrics['MSE']:.4f}")
    print(f"RMSE (جذر میانگین مربعات خطا): {metrics['RMSE']:.4f}")
    print(f"MAE (خطای مطلق میانگین): {metrics['MAE']:.4f}")
    print(f"MAPE (خطای درصدی مطلق میانگین): {metrics['MAPE']:.4f}%")
    
    print("\n۲. معیارهای دقت:")
    print("-"*30)
    print(f"R² (ضریب تعیین): {metrics['R2']:.4f}")
    print(f"دقت پیش‌بینی جهت حرکت: {metrics['Direction_Accuracy']:.4f}%")
    print(f"حداکثر خطا: ${metrics['Max_Error']:.4f}")
    print(f"خطای میانه: ${metrics['Median_Error']:.4f}")
    
    print("\n۳. معیارهای معاملاتی:")
    print("-"*30)
    print(f"نسبت نوسانات: {metrics['Volatility_Ratio']:.4f}")
    print(f"نسبت شارپ: {metrics['Sharpe_Ratio']:.4f}")
    print(f"نرخ برد: {metrics['Win_Rate']:.4f}%")
    print(f"فاکتور سود: {metrics['Profit_Factor']:.4f}")

def compare_models(model_results, names=None):
    """
    مقایسه نتایج چندین مدل
    
    Args:
        model_results: لیست دیکشنری‌های معیارهای ارزیابی
        names: لیست نام‌های مدل‌ها (اختیاری)
    
    Returns:
        dict: دیکشنری مقایسه مدل‌ها
    """
    if names is None:
        names = [f"Model {i+1}" for i in range(len(model_results))]
    
    # معیارهای اصلی برای مقایسه
    key_metrics = ['RMSE', 'MAE', 'R2', 'Direction_Accuracy', 'Sharpe_Ratio', 'Win_Rate']
    
    comparison = {}
    for metric in key_metrics:
        comparison[metric] = {name: result[metric] for name, result in zip(names, model_results)}
    
    # چاپ نتایج مقایسه
    print("\nمقایسه مدل‌ها:")
    print("-"*50)
    
    for metric in key_metrics:
        print(f"\n{metric}:")
        for name in names:
            print(f"{name}: {comparison[metric][name]:.4f}")
    
    # تعیین بهترین مدل برای هر معیار
    best_models = {}
    for metric in key_metrics:
        if metric in ['RMSE', 'MAE']:  # معیارهایی که کمتر بودن بهتر است
            best_model = min(comparison[metric].items(), key=lambda x: x[1])[0]
        else:  # معیارهایی که بیشتر بودن بهتر است
            best_model = max(comparison[metric].items(), key=lambda x: x[1])[0]
        best_models[metric] = best_model
    
    print("\nبهترین مدل برای هر معیار:")
    for metric, model in best_models.items():
        print(f"{metric}: {model}")
    
    return comparison 