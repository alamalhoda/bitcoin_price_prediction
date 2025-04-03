"""
اسکریپت پیش‌بینی قیمت بیت‌کوین با استفاده از مدل موجود
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd

# افزودن مسیر ریشه پروژه به path برای import کردن ماژول‌های پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitcoin_price_prediction.utils.data_utils import load_data, prepare_data, prepare_future_data, inverse_transform_predictions
from bitcoin_price_prediction.utils.evaluation import evaluate_model, save_metrics
from bitcoin_price_prediction.utils.visualization import plot_predictions, plot_future_predictions
from bitcoin_price_prediction.models.lstm_model import load_saved_model

def load_model_and_predict(args):
    """
    بارگذاری مدل ذخیره شده و انجام پیش‌بینی
    
    Args:
        args: آرگومان‌های ورودی
        
    Returns:
        dict: نتایج پیش‌بینی
    """
    # بررسی پارامترهای ورودی
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"فایل مدل در مسیر {args.model_path} یافت نشد!")
    
    # بارگذاری داده‌ها
    print(f"بارگذاری داده‌ها از {args.data_path}...")
    data = load_data(args.data_path)
    
    # آماده‌سازی داده‌ها برای ارزیابی مدل
    print("\nآماده‌سازی داده‌ها برای ارزیابی مدل...")
    
    # بارگذاری مدل
    print(f"\nبارگذاری مدل از مسیر {args.model_path}...")
    model = load_saved_model(args.model_path)
    
    # استخراج lag_days از شکل ورودی مدل
    lag_days = model.input_shape[1]
    print(f"تعداد روزهای گذشته استفاده شده در مدل: {lag_days}")
    
    # آماده‌سازی داده‌ها برای ارزیابی
    X_train, y_train, X_test, y_test, scaler, _ = prepare_data(
        data=data,
        target_column='Close',
        lag_days=lag_days,
        train_split_ratio=args.train_split_ratio
    )
    
    # ایجاد دایرکتوری خروجی
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"prediction_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    
    # ارزیابی مدل روی داده‌های تست
    print("\nارزیابی مدل روی داده‌های تست...")
    test_dates = data.index[-len(X_test):]
    metrics, test_predictions, y_test_orig = evaluate_model(model, X_test, y_test, scaler)
    
    # ذخیره متریک‌ها
    metrics_path = os.path.join(output_dir, "metrics.json")
    save_metrics(metrics, metrics_path)
    
    # ذخیره نمودار پیش‌بینی‌های تست
    test_predictions_plot_path = os.path.join(figures_dir, "test_predictions.png")
    plot_predictions(test_dates, y_test_orig, test_predictions, output_path=test_predictions_plot_path)
    
    # پیش‌بینی برای روزهای آینده
    print(f"\nپیش‌بینی قیمت برای {args.future_days} روز آینده...")
    future_input, future_dates, future_scaler = prepare_future_data(
        data=data,
        target_column='Close',
        lag_days=lag_days,
        future_days=args.future_days
    )
    
    # انجام پیش‌بینی
    future_predictions = model.predict(future_input)
    
    # برگرداندن مقیاس پیش‌بینی‌ها
    future_predictions_orig = inverse_transform_predictions(future_predictions, scaler)
    
    # ذخیره پیش‌بینی‌های آینده
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_predictions_orig.flatten()
    })
    future_df_path = os.path.join(output_dir, "future_predictions.csv")
    future_df.to_csv(future_df_path)
    print(f"پیش‌بینی‌های آینده در مسیر {future_df_path} ذخیره شدند.")
    
    # ذخیره نمودار پیش‌بینی‌های آینده
    # استفاده از 30 روز آخر داده‌های تاریخی برای نمایش در نمودار
    historical_dates = data.index[-30:]
    historical_prices = data['Close'].values[-30:]
    
    future_predictions_plot_path = os.path.join(figures_dir, "future_predictions.png")
    plot_future_predictions(
        future_dates=future_dates,
        future_predictions=future_predictions_orig.flatten(),
        historical_dates=historical_dates,
        historical_prices=historical_prices,
        output_path=future_predictions_plot_path
    )
    
    # خلاصه نتایج
    results = {
        'metrics': metrics,
        'test_predictions': test_predictions,
        'future_predictions': future_predictions_orig.flatten(),
        'future_dates': future_dates,
        'output_dir': output_dir
    }
    
    return results

def main():
    """
    تابع اصلی
    """
    # پارس کردن آرگومان‌ها
    parser = argparse.ArgumentParser(description='پیش‌بینی قیمت بیت‌کوین با استفاده از مدل موجود')
    
    # پارامترهای ورودی
    parser.add_argument('--model_path', type=str, required=True,
                       help='مسیر مدل ذخیره شده')
    parser.add_argument('--data_path', type=str, default='data/bitcoin_data.csv',
                       help='مسیر فایل داده‌های بیت‌کوین')
    parser.add_argument('--train_split_ratio', type=float, default=0.8,
                       help='نسبت داده‌های آموزش (پیش‌فرض: 0.8)')
    
    # پارامترهای پیش‌بینی
    parser.add_argument('--future_days', type=int, default=30,
                       help='تعداد روزهای آینده برای پیش‌بینی (پیش‌فرض: 30)')
    
    # پارامترهای خروجی
    parser.add_argument('--output_dir', type=str, default='output/predictions',
                       help='مسیر دایرکتوری خروجی (پیش‌فرض: output/predictions)')
    
    args = parser.parse_args()
    
    # بارگذاری مدل و پیش‌بینی
    results = load_model_and_predict(args)
    
    print("\nفرآیند پیش‌بینی با موفقیت انجام شد.")
    print(f"نتایج در دایرکتوری {results['output_dir']} ذخیره شدند.")

if __name__ == '__main__':
    main() 