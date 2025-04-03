"""
اسکریپت دانلود داده‌های قیمت بیت‌کوین
"""

import argparse
from datetime import datetime
import os
import sys

# افزودن مسیر ریشه پروژه به path برای import کردن ماژول‌های پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitcoin_price_prediction.utils.data_utils import download_bitcoin_data

def parse_arguments():
    """
    پارس کردن آرگومان‌های خط فرمان
    
    Returns:
        argparse.Namespace: آرگومان‌های پارس شده
    """
    parser = argparse.ArgumentParser(description='دانلود داده‌های قیمت بیت‌کوین')
    
    parser.add_argument('--start_date', type=str, default='2015-01-01',
                        help='تاریخ شروع داده‌ها (پیش‌فرض: 2015-01-01)')
    
    parser.add_argument('--end_date', type=str, default=None,
                        help='تاریخ پایان داده‌ها (پیش‌فرض: امروز)')
    
    parser.add_argument('--save_path', type=str, default='data/bitcoin_data.csv',
                        help='مسیر ذخیره‌سازی فایل CSV (پیش‌فرض: data/bitcoin_data.csv)')
    
    return parser.parse_args()

def main():
    """
    تابع اصلی
    """
    # پارس کردن آرگومان‌ها
    args = parse_arguments()
    
    # دانلود داده‌ها
    data = download_bitcoin_data(
        start_date=args.start_date,
        end_date=args.end_date,
        save_path=args.save_path
    )
    
    print("دانلود داده‌ها با موفقیت انجام شد.")

if __name__ == '__main__':
    main() 