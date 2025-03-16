import yfinance as yf
import pandas as pd
from datetime import datetime
import requests
import json
import jdatetime
from datetime import datetime
import os
import pytz  # برای مدیریت منطقه زمانی

def fetch_data(ticker='BTC-USD', start_date='2010-01-01', end_date='2025-03-06',interval='1d'):
    print(f"Fetching {ticker} data from {start_date} to {end_date}...")
    # data = yf.download(ticker, start=start_date, end=end_date,interval=interval)
    data = yf.download('BTC-USD', start=start_date, end=end_date, interval=interval)
    data.index.name = 'Date'
    return data

def load_data_from_csv(file_path='bitcoin_data.csv'):
    try:
        # مسیر کامل فایل
        data_dir = "data"
        full_path = os.path.join(data_dir, file_path)
        
        data = pd.read_csv(full_path, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)
        data.index.name = 'Date'
        print(f"Data loaded from {full_path} successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: File '{full_path}' not found.")
        return None

def save_data_to_csv(data, file_path='bitcoin_data.csv'):
    # ایجاد پوشه data اگر وجود نداشت
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # مسیر کامل فایل
    full_path = os.path.join(data_dir, file_path)
    
    data.to_csv(full_path)
    print(f"Data saved to {full_path}.")

def fetch_cryptocompare_data(start_date='2017-01-01', end_date='2018-01-01', interval='hour'):
    # تنظیم API
    base_url = "https://min-api.cryptocompare.com/data/v2/histohour"
    
    # تبدیل تاریخ‌ها به timestamp
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
    
    # محاسبه تعداد ساعت‌ها
    hours = int((end_ts - start_ts) / 3600)
    
    # دریافت داده‌ها (حداکثر 2000 رکورد در هر درخواست)
    all_data = []
    current_ts = end_ts
    
    while current_ts > start_ts:
        limit = min(2000, hours)
        url = f"{base_url}?fsym=BTC&tsym=USD&limit={limit}&toTs={current_ts}"
        print(f"Fetching data for timestamp: {datetime.fromtimestamp(current_ts)}")
        
        response = requests.get(url)
        json_data = response.json()
        
        if json_data.get('Response') == 'Success':
            data = json_data['Data']['Data']
            all_data = data + all_data
            
            if len(data) < limit:
                break
                
            current_ts = data[0]['time']
            hours -= len(data)
        else:
            print(f"Error: {json_data.get('Message', 'Unknown error')}")
            break
    
    # تبدیل به دیتافریم
    df = pd.DataFrame(all_data)
    df['Date'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('Date', inplace=True)
    
    # تغییر نام ستون‌ها
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volumefrom': 'Volume'
    })
    
    # فیلتر کردن داده‌ها برای بازه زمانی مورد نظر
    df = df[start_date:end_date]
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def fetch_nobitex_ohlc(start_date_shamsi, end_date_shamsi, symbol, resolution, api_token):
    """
    دریافت آمار OHLC بازار نوبیتکس با صفحه‌بندی و تبدیل زمان به ساعت تهران.
    
    Args:
        start_date_shamsi (str): تاریخ شروع به فرمت شمسی (مثال: '1403-01-01')
        end_date_shamsi (str): تاریخ پایان به فرمت شمسی (مثال: '1403-01-30')
        symbol (str): نماد جفت‌ارز (مثال: 'BTCIRT')
        resolution (str): رزولوشن کندل‌ها (مثال: '1hour')
        api_token (str): توکن معتبر API نوبیتکس
    """
    # تبدیل تاریخ شمسی به میلادی
    start_date_miladi = jdatetime.datetime.strptime(start_date_shamsi, '%Y-%m-%d').togregorian()
    end_date_miladi = jdatetime.datetime.strptime(end_date_shamsi, '%Y-%m-%d').togregorian()
    
    # فرمت تاریخ برای API (Unix Timestamp به ثانیه)
    start_time = int(start_date_miladi.timestamp())
    end_time = int(end_date_miladi.timestamp())
    
    # تنظیمات درخواست
    url = "https://api.nobitex.ir/market/udf/history"
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }
    
    # لیست برای ذخیره همه داده‌ها
    all_data = []
    
    # محاسبه تعداد تقریبی کندل‌ها
    days_diff = (end_date_miladi - start_date_miladi).days
    if resolution == "1hour":
        approx_candles = days_diff * 24
    elif resolution == "1day":
        approx_candles = days_diff
    else:
        minutes_per_day = {'1min': 1440, '5min': 288, '15min': 96, '30min': 48}
        approx_candles = days_diff * minutes_per_day.get(resolution.replace('min', ''), 24 * 60)
    
    # صفحه‌بندی
    page = 1
    while True:
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": start_time,
            "to": end_time,
            "page": page
        }
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"خطا در درخواست صفحه {page}: {response.status_code}, {response.text}")
            break
        
        data = response.json()
        if data.get("s") != "ok" or "t" not in data:
            print(f"پایان داده‌ها در صفحه {page}")
            break
        
        # استخراج آرایه‌های OHLC
        timestamps = data["t"]
        opens = data["o"]
        highs = data["h"]
        lows = data["l"]
        closes = data["c"]
        volumes = data["v"]
        
        # تبدیل به لیست دیکشنری
        for i in range(len(timestamps)):
            all_data.append({
                "time": timestamps[i] * 1000,  # تبدیل به میلی‌ثانیه برای سازگاری با pandas
                "Open": float(opens[i]),
                "High": float(highs[i]),
                "Low": float(lows[i]),
                "Close": float(closes[i]),
                "Volume": float(volumes[i])
            })
        
        # اگه تعداد کندل‌ها کمتر از ۵۰۰ بود، دیگه صفحه بعدی نداره
        if len(timestamps) < 500:
            break
        
        page += 1
    
    # تبدیل به دیتافریم pandas
    df = pd.DataFrame(all_data)
    if not df.empty:
        # تبدیل ستون time به فرمت تاریخ (ابتدا به UTC)
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        # تبدیل به ساعت تهران (UTC+3:30)
        tehran_tz = pytz.timezone('Asia/Tehran')
        df['time'] = df['time'].dt.tz_convert(tehran_tz)
        df = df[['time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # نام‌گذاری فایل‌ها بر اساس منبع، تاریخ و رزولوشن
    file_prefix = f"nobitex_{start_date_shamsi}_to_{end_date_shamsi}_{resolution}"
    
    # ایجاد پوشه data اگر وجود نداشت
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # مسیر کامل فایل‌ها
    json_file = os.path.join(data_dir, f"{file_prefix}.json")
    csv_file = os.path.join(data_dir, f"{file_prefix}.csv")
    
    # ذخیره در فایل JSON
    # برای ذخیره در JSON، زمان رو به رشته تبدیل می‌کنیم
    all_data_for_json = []
    for item in all_data:
        item_copy = item.copy()
        item_copy['time'] = pd.to_datetime(item['time'], unit='ms', utc=True).tz_convert(tehran_tz).isoformat()
        all_data_for_json.append(item_copy)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_data_for_json, f, ensure_ascii=False, indent=4)
    
    # ذخیره در فایل CSV
    if not df.empty:
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"تعداد کل کندل‌ها دریافت‌شده: {len(all_data)}")
    print(f"داده‌ها با موفقیت در {json_file} و {csv_file} ذخیره شدند.")
    return df

# مثال استفاده
if __name__ == "__main__":
    # اطلاعات ورودی
    start_date = "1403-01-01"
    end_date = "1403-12-30"
    symbol = "BTCIRT"
    resolution = "60"
    api_token = "YOUR_API_TOKEN_HERE"  # توکن خودت رو جایگزین کن
    
    # فراخوانی تابع
    result_df = fetch_nobitex_ohlc(start_date, end_date, symbol, resolution, api_token)
    if not result_df.empty:
        print(result_df.head())
        print(f"تعداد کل ردیف‌ها: {len(result_df)}")