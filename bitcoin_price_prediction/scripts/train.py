"""
اسکریپت آموزش مدل LSTM برای پیش‌بینی قیمت بیت‌کوین
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
from bitcoin_price_prediction.utils.visualization import plot_training_history, plot_predictions, plot_metrics
from bitcoin_price_prediction.models.lstm_model import create_lstm_model, train_lstm_model
from bitcoin_price_prediction.config import get_lstm_model_args

def train_model(args):
    """
    آموزش مدل با استفاده از پارامترهای مشخص شده
    
    Args:
        args: آرگومان‌های ورودی
        
    Returns:
        tuple: (model, run_dir) مدل آموزش دیده و مسیر دایرکتوری خروجی
    """
    # استخراج پارامترهای مدل از آرگومان‌ها
    params = get_lstm_model_args(args)
    
    # بارگذاری داده‌ها
    print(f"بارگذاری داده‌ها از {params['data_path']}...")
    data = load_data(params['data_path'])
    
    # آماده‌سازی داده‌ها
    print("\nآماده‌سازی داده‌ها برای آموزش...")
    X_train, y_train, X_test, y_test, scaler, sample_weights = prepare_data(
        data=data,
        target_column='Close',
        lag_days=params['lag_days'],
        train_split_ratio=params['train_split_ratio'],
        exponential_weighting=params['exponential_weighting'],
        weight_decay=params['weight_decay']
    )
    
    print(f"تعداد نمونه‌های آموزش: {len(X_train)}")
    print(f"تعداد نمونه‌های تست: {len(X_test)}")
    
    # ایجاد دایرکتوری خروجی با تاریخ و زمان
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(params['output_dir'], f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # تنظیم مسیرهای خروجی
    model_save_dir = os.path.join(run_dir, "models")
    figures_dir = os.path.join(run_dir, "figures")
    
    # آموزش مدل
    print("\nشروع آموزش مدل LSTM...")
    model, history = train_lstm_model(
        X_train=X_train,
        y_train=y_train,
        validation_split=params['validation_split'],
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        patience=params['patience'],
        sample_weights=sample_weights,
        save_model=params['save_model'],
        model_save_dir=model_save_dir
    )
    
    # ذخیره نمودار تاریخچه آموزش
    history_plot_path = os.path.join(figures_dir, "training_history.png")
    plot_training_history(history, output_path=history_plot_path)
    
    # ارزیابی مدل
    print("\nارزیابی مدل روی داده‌های تست...")
    
    # استخراج تاریخ‌های تست
    test_dates = data.index[-len(X_test):]
    
    # ارزیابی مدل
    metrics, test_predictions, y_test_orig = evaluate_model(model, X_test, y_test, scaler)
    
    # ذخیره متریک‌ها
    metrics_path = os.path.join(run_dir, "metrics.json")
    save_metrics(metrics, metrics_path)
    
    # ذخیره نمودار پیش‌بینی‌ها
    predictions_plot_path = os.path.join(figures_dir, "test_predictions.png")
    plot_predictions(test_dates, y_test_orig, test_predictions, output_path=predictions_plot_path)
    
    # ذخیره نمودار متریک‌ها
    metrics_plot_path = os.path.join(figures_dir, "metrics.png")
    plot_metrics(metrics, output_path=metrics_plot_path)
    
    # ذخیره پارامترهای مدل
    params_path = os.path.join(run_dir, "params.json")
    with open(params_path, 'w') as f:
        json.dump({k: str(v) if isinstance(v, (datetime, dict)) else v 
                  for k, v in params.items()}, f, indent=4)
    
    print(f"\nفرآیند آموزش و ارزیابی مدل با موفقیت انجام شد.")
    print(f"نتایج در دایرکتوری {run_dir} ذخیره شدند.")
    
    return model, run_dir

def main():
    """
    تابع اصلی
    """
    # پارس کردن آرگومان‌ها
    parser = argparse.ArgumentParser(description='آموزش مدل LSTM برای پیش‌بینی قیمت بیت‌کوین')
    
    # پارامترهای داده
    parser.add_argument('--data_path', type=str, default='data/bitcoin_data.csv',
                       help='مسیر فایل داده‌های بیت‌کوین')
    parser.add_argument('--train_split_ratio', type=float, default=0.8,
                       help='نسبت داده‌های آموزش (پیش‌فرض: 0.8)')
    
    # پارامترهای مدل
    parser.add_argument('--model_type', type=str, default='lstm',
                       help='نوع مدل برای آموزش (پیش‌فرض: lstm)')
    parser.add_argument('--lag_days', type=int, default=30,
                       help='تعداد روزهای گذشته برای پیش‌بینی (پیش‌فرض: 30)')
    parser.add_argument('--lstm1_units', type=int, default=50,
                       help='تعداد واحدهای لایه اول LSTM (پیش‌فرض: 50)')
    parser.add_argument('--lstm2_units', type=int, default=30,
                       help='تعداد واحدهای لایه دوم LSTM (پیش‌فرض: 30)')
    parser.add_argument('--dropout1', type=float, default=0.3,
                       help='نرخ dropout لایه اول (پیش‌فرض: 0.3)')
    parser.add_argument('--dropout2', type=float, default=0.3,
                       help='نرخ dropout لایه دوم (پیش‌فرض: 0.3)')
    
    # پارامترهای آموزش
    parser.add_argument('--exponential_weighting', action='store_true',
                       help='استفاده از وزن‌های نمایی برای داده‌های اخیر')
    parser.add_argument('--weight_decay', type=float, default=0.5,
                       help='ضریب کاهش وزن (پیش‌فرض: 0.5)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='تعداد دوره‌های آموزش (پیش‌فرض: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='اندازه دسته (پیش‌فرض: 32)')
    parser.add_argument('--patience', type=int, default=20,
                       help='تعداد دوره‌های صبر برای توقف زودهنگام (پیش‌فرض: 20)')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='نسبت داده‌های اعتبارسنجی (پیش‌فرض: 0.2)')
    
    # پارامترهای خروجی
    parser.add_argument('--output_dir', type=str, default='output',
                       help='مسیر دایرکتوری خروجی (پیش‌فرض: output)')
    parser.add_argument('--save_model', action='store_true',
                       help='ذخیره مدل آموزش دیده')
    
    args = parser.parse_args()
    
    # آموزش مدل
    model, run_dir = train_model(args)
    
if __name__ == '__main__':
    main() 