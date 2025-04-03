"""
فایل اصلی برای راه‌اندازی برنامه پیش‌بینی قیمت بیت‌کوین
"""

import os
import argparse
import sys

from bitcoin_price_prediction.scripts.train import train_model
from bitcoin_price_prediction.scripts.predict import load_model_and_predict
from bitcoin_price_prediction.config import lstm_config

def parse_arguments():
    """
    پارس کردن آرگومان‌های خط فرمان
    
    Returns:
        argparse.Namespace: آرگومان‌های پارس شده
    """
    parser = argparse.ArgumentParser(description='برنامه پیش‌بینی قیمت بیت‌کوین')
    subparsers = parser.add_subparsers(dest='command', help='دستور مورد نظر')
    
    # زیر پارسر برای دستور train
    train_parser = subparsers.add_parser('train', help='آموزش مدل جدید')
    
    # پارامترهای داده
    train_parser.add_argument('--data_path', type=str, default='data/bitcoin_data.csv',
                             help='مسیر فایل داده‌های بیت‌کوین')
    train_parser.add_argument('--train_split_ratio', type=float, default=0.8,
                             help='نسبت داده‌های آموزش (پیش‌فرض: 0.8)')
    
    # پارامترهای مدل
    train_parser.add_argument('--model_type', type=str, default='lstm',
                             help='نوع مدل برای آموزش (پیش‌فرض: lstm)')
    train_parser.add_argument('--lag_days', type=int, default=30,
                             help='تعداد روزهای گذشته برای پیش‌بینی (پیش‌فرض: 30)')
    train_parser.add_argument('--lstm1_units', type=int, default=50,
                             help='تعداد واحدهای لایه اول LSTM (پیش‌فرض: 50)')
    train_parser.add_argument('--lstm2_units', type=int, default=30,
                             help='تعداد واحدهای لایه دوم LSTM (پیش‌فرض: 30)')
    train_parser.add_argument('--dropout1', type=float, default=0.3,
                             help='نرخ dropout لایه اول (پیش‌فرض: 0.3)')
    train_parser.add_argument('--dropout2', type=float, default=0.3,
                             help='نرخ dropout لایه دوم (پیش‌فرض: 0.3)')
    
    # پارامترهای آموزش
    train_parser.add_argument('--exponential_weighting', action='store_true',
                             help='استفاده از وزن‌های نمایی برای داده‌های اخیر')
    train_parser.add_argument('--weight_decay', type=float, default=0.5,
                             help='ضریب کاهش وزن (پیش‌فرض: 0.5)')
    
    # پارامترهای پیش‌بینی
    train_parser.add_argument('--future_days', type=int, default=30,
                             help='تعداد روزهای آینده برای پیش‌بینی (پیش‌فرض: 30)')
    
    # پارامترهای خروجی
    train_parser.add_argument('--output_dir', type=str, default='output',
                             help='مسیر دایرکتوری خروجی (پیش‌فرض: output)')
    train_parser.add_argument('--save_model', action='store_true',
                             help='ذخیره مدل آموزش دیده')
    
    # زیر پارسر برای دستور predict
    predict_parser = subparsers.add_parser('predict', help='پیش‌بینی با استفاده از مدل موجود')
    
    # پارامترهای ورودی
    predict_parser.add_argument('--model_path', type=str, required=True,
                               help='مسیر مدل ذخیره شده')
    predict_parser.add_argument('--data_path', type=str, default='data/bitcoin_data.csv',
                               help='مسیر فایل داده‌های بیت‌کوین')
    predict_parser.add_argument('--train_split_ratio', type=float, default=0.8,
                               help='نسبت داده‌های آموزش (پیش‌فرض: 0.8)')
    
    # پارامترهای پیش‌بینی
    predict_parser.add_argument('--future_days', type=int, default=30,
                               help='تعداد روزهای آینده برای پیش‌بینی (پیش‌فرض: 30)')
    
    # پارامترهای خروجی
    predict_parser.add_argument('--output_dir', type=str, default='output/predictions',
                               help='مسیر دایرکتوری خروجی (پیش‌فرض: output/predictions)')
    
    return parser.parse_args()

def main():
    """
    تابع اصلی برنامه
    """
    args = parse_arguments()
    
    if args.command == 'train':
        # آموزش مدل
        print("شروع آموزش مدل جدید...")
        model, run_dir = train_model(args)
        
        if run_dir:
            print(f"آموزش و ارزیابی مدل با موفقیت انجام شد. نتایج در مسیر {run_dir} ذخیره شدند.")
        else:
            print("آموزش و ارزیابی مدل با موفقیت انجام شد.")
    
    elif args.command == 'predict':
        # پیش‌بینی با استفاده از مدل موجود
        print("شروع پیش‌بینی با استفاده از مدل موجود...")
        results = load_model_and_predict(args)
        print("\nپیش‌بینی با موفقیت انجام شد.")
    
    else:
        print("لطفاً یکی از دستورات 'train' یا 'predict' را انتخاب کنید.")
        print("برای مشاهده راهنما، از دستور 'python main.py -h' استفاده کنید.")
        sys.exit(1)

if __name__ == '__main__':
    main() 