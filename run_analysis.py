import pandas as pd
from data_handler import (
    convert_hourly_to_daily,
    extract_daily_min_max_hours,
    plot_min_max_hour_frequency,
    plot_hour_price_correlations
)

# خواندن داده‌های ساعتی
hourly_data = pd.read_csv('data/nobitex_1403-01-01_to_1403-12-30_60.csv')
hourly_data['time'] = pd.to_datetime(hourly_data['time'])
hourly_data.set_index('time', inplace=True)

# تبدیل داده‌های ساعتی به روزانه
daily_data = convert_hourly_to_daily(hourly_data, output_file='data/daily_data.csv')

# استخراج ساعت‌های حداقل و حداکثر قیمت روزانه
min_max_data = extract_daily_min_max_hours(hourly_data, output_file='data/min_max_hours.csv')

# رسم نمودار فراوانی ساعت‌های حداقل و حداکثر قیمت
plot_min_max_hour_frequency(min_max_data, output_file='plots/hour_frequency.png')

# رسم نمودار همبستگی قیمت با ساعت‌های روز
plot_hour_price_correlations(min_max_data, output_file='plots/hour_correlations.png')

print("تحلیل‌ها با موفقیت انجام شد و نتایج ذخیره شدند.") 