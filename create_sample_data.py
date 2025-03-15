import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ایجاد تاریخ‌ها برای 2 سال گذشته
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 سال

# ایجاد دیتافریم با تاریخ‌ها
dates = pd.date_range(start=start_date, end=end_date, freq='D')
df = pd.DataFrame(index=dates)

# تولید قیمت‌های تصادفی با روند صعودی
np.random.seed(42)
base_price = 30000  # قیمت پایه
trend = np.linspace(0, 10000, len(dates))  # روند صعودی
noise = np.random.normal(0, 1000, len(dates))  # نویز تصادفی
df['Close'] = base_price + trend + noise
df['Open'] = df['Close'] + np.random.normal(0, 100, len(dates))
df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.normal(0, 50, len(dates))
df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.normal(0, 50, len(dates))
df['Volume'] = np.random.normal(1000000, 200000, len(dates)).clip(0)

# ذخیره داده‌ها
df.to_csv('bitcoin_data.csv')
print("داده‌های نمونه در فایل bitcoin_data.csv ذخیره شدند.") 