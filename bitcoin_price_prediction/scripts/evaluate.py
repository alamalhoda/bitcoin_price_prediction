"""
اسکریپت ارزیابی مدل‌های پیش‌بینی قیمت بیت‌کوین
"""

import os
import sys
import argparse
from datetime import datetime
import json

# افزودن مسیر ریشه پروژه به path برای import کردن ماژول‌های پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitcoin_price_prediction.utils.data_utils import load_data, prepare_data
from bitcoin_price_prediction.utils.evaluation import evaluate_model, save_metrics
from bitcoin_price_prediction.utils.visualization import plot_predictions, plot_metrics
from bitcoin_price_prediction.models.lstm_model import load_saved_model

def evaluate_saved_model(args):
    """
    ارزیابی مدل ذخیره شده روی داده‌های جدید
    
    Args:
        args: آرگومان‌های ورودی
        
    Returns:
        dict: نتایج ارزیابی
    """
    # بررسی پارامترهای ورودی
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"فایل مدل در مسیر {args.model_path} یافت نشد!")
    
    # بارگذاری داده‌ها
    print(f"بارگذاری داده‌ها از {args.data_path}...")
    data = load_data(args.data_path)
    
    # بارگذاری مدل
    print(f"\nبارگذاری مدل از مسیر {args.model_path}...")
    model = load_saved_model(args.model_path)
    
    # استخراج lag_days از شکل ورودی مدل
    lag_days = model.input_shape[1]
    print(f"تعداد روزهای گذشته استفاده شده در مدل: {lag_days}")
    
    # آماده‌سازی داده‌ها برای ارزیابی
    print("\nآماده‌سازی داده‌ها برای ارزیابی...")
    X_train, y_train, X_test, y_test, scaler, _ = prepare_data(
        data=data,
        target_column='Close',
        lag_days=lag_days,
        train_split_ratio=args.train_split_ratio
    )
    
    print(f"تعداد نمونه‌های آموزش: {len(X_train)}")
    print(f"تعداد نمونه‌های تست: {len(X_test)}")
    
    # ایجاد دایرکتوری خروجی
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"evaluation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    
    # ارزیابی مدل روی داده‌های تست
    print("\nارزیابی مدل روی داده‌های تست...")
    test_dates = data.index[-len(X_test):]
    metrics, test_predictions, y_test_orig = evaluate_model(model, X_test, y_test, scaler)
    
    # ذخیره متریک‌ها
    metrics_path = os.path.join(output_dir, "metrics.json")
    save_metrics(metrics, metrics_path)
    
    # ذخیره نمودار پیش‌بینی‌ها
    predictions_plot_path = os.path.join(figures_dir, "test_predictions.png")
    plot_predictions(test_dates, y_test_orig, test_predictions, output_path=predictions_plot_path)
    
    # ذخیره نمودار متریک‌ها
    metrics_plot_path = os.path.join(figures_dir, "metrics.png")
    plot_metrics(metrics, output_path=metrics_plot_path)
    
    # ذخیره پارامترهای ارزیابی
    params = {
        'model_path': args.model_path,
        'data_path': args.data_path,
        'train_split_ratio': args.train_split_ratio,
        'lag_days': lag_days,
        'evaluation_timestamp': timestamp
    }
    
    params_path = os.path.join(output_dir, "params.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"\nفرآیند ارزیابی مدل با موفقیت انجام شد.")
    print(f"نتایج در دایرکتوری {output_dir} ذخیره شدند.")
    
    # خلاصه نتایج
    results = {
        'metrics': metrics,
        'test_predictions': test_predictions,
        'y_test_orig': y_test_orig,
        'test_dates': test_dates,
        'output_dir': output_dir
    }
    
    return results

def main():
    """
    تابع اصلی
    """
    # پارس کردن آرگومان‌ها
    parser = argparse.ArgumentParser(description='ارزیابی مدل پیش‌بینی قیمت بیت‌کوین')
    
    # پارامترهای ورودی
    parser.add_argument('--model_path', type=str, required=True,
                       help='مسیر مدل ذخیره شده')
    parser.add_argument('--data_path', type=str, default='data/bitcoin_data.csv',
                       help='مسیر فایل داده‌های بیت‌کوین')
    parser.add_argument('--train_split_ratio', type=float, default=0.8,
                       help='نسبت داده‌های آموزش (پیش‌فرض: 0.8)')
    
    # پارامترهای خروجی
    parser.add_argument('--output_dir', type=str, default='output/evaluations',
                       help='مسیر دایرکتوری خروجی (پیش‌فرض: output/evaluations)')
    
    args = parser.parse_args()
    
    # ارزیابی مدل
    results = evaluate_saved_model(args)
    
    # خلاصه مطلب نتایج
    print("\nخلاصه متریک‌های ارزیابی:")
    for metric_name, metric_value in results['metrics'].items():
        print(f"{metric_name}: {metric_value:.4f}")

if __name__ == '__main__':
    main() 