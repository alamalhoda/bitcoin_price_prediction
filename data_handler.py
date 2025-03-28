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

def load_data_from_csv(file_path='nobitex_1403-01-01_to_1403-12-30_360.csv'):
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

def save_data_to_csv(data, file_path='nobitex_1403_360.csv'):
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


def load_hourly_data(file_path, convert_to_tehran_time=True):
    """
عملکرد: داده‌های ساعتی رو از فایل CSV می‌خونه و ویژگی‌های اضافی (مثل تغییرات قیمت، میانگین متحرک و نوسانات) رو محاسبه می‌کنه.    
    ویژگی‌های محاسبه‌شده:
    Hourly_Change: تغییرات ساعتی قیمت (درصد)
    SMA_24 و SMA_72: میانگین متحرک ۲۴ و ۷۲ ساعته
    Volatility_24h: نوسانات ۲۴ ساعته (انحراف معیار)
    Args:
        file_path (str): مسیر فایل CSV (نام فایل در پوشه data)
        convert_to_tehran_time (bool): تبدیل زمان به ساعت تهران
        
    Returns:
        DataFrame: دیتافریم با ستون‌های OHLC، Volume و ویژگی‌های اضافی.
    """
    try:
        # مسیر کامل فایل
        data_dir = "data"
        full_path = os.path.join(data_dir, file_path)
        
        # خواندن فایل CSV
        df = pd.read_csv(full_path)
        
        # بررسی ستون‌های موجود
        if 'time' in df.columns:
            # تبدیل ستون time به datetime
            if pd.api.types.is_numeric_dtype(df['time']):
                # اگر time عددی است (timestamp)، آن را به datetime تبدیل می‌کنیم
                df['time'] = pd.to_datetime(df['time'], unit='ms')
            else:
                # در غیر این صورت فرض می‌کنیم که رشته datetime است
                df['time'] = pd.to_datetime(df['time'])
                
            # تبدیل به ساعت تهران اگر درخواست شده باشد
            if convert_to_tehran_time:
                if df['time'].dt.tz is None:  # اگر timezone ندارد
                    df['time'] = df['time'].dt.tz_localize('UTC')
                tehran_tz = pytz.timezone('Asia/Tehran')
                df['time'] = df['time'].dt.tz_convert(tehran_tz)
                
            # تنظیم time به عنوان ایندکس
            df.set_index('time', inplace=True)
        
        # اطمینان از وجود ستون‌های قیمت
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"هشدار: ستون‌های {', '.join(missing_columns)} در فایل وجود ندارند.")
        
        # محاسبه ویژگی‌های اضافی مفید
        if 'Close' in df.columns and 'Open' in df.columns:
            # تغییرات قیمت در هر ساعت (درصد)
            df['Hourly_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        if 'Close' in df.columns:
            # میانگین متحرک ساده ۲۴ ساعته (یک روز)
            df['SMA_24'] = df['Close'].rolling(window=24).mean()
            
            # میانگین متحرک ساده ۷۲ ساعته (سه روز)
            df['SMA_72'] = df['Close'].rolling(window=72).mean()
            
            # نوسانات قیمت (انحراف معیار ۲۴ ساعته)
            df['Volatility_24h'] = df['Close'].rolling(window=24).std()
        
        print(f"داده‌های ساعتی از فایل {full_path} با موفقیت بارگذاری شدند.")
        print(f"تعداد کل ساعت‌ها: {len(df)}")
        print(f"بازه زمانی: از {df.index.min()} تا {df.index.max()}")
        
        return df
    
    except FileNotFoundError:
        print(f"خطا: فایل '{file_path}' در پوشه data یافت نشد.")
        return None
    except Exception as e:
        print(f"خطا در بارگذاری داده‌ها: {str(e)}")
        return None

def merge_hourly_data_files(file_paths, output_file='merged_hourly_data.csv'):
    """
    ادغام چندین فایل CSV ساعتی و ذخیره نتیجه در یک فایل جدید
    
    Args:
        file_paths (list): لیست مسیرهای فایل‌های CSV در پوشه data
        output_file (str): نام فایل خروجی برای ذخیره داده‌های ادغام شده
        
    Returns:
        DataFrame: داده‌های ساعتی ادغام شده
    """
    all_dfs = []
    
    for file_path in file_paths:
        df = load_hourly_data(file_path)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        print("هیچ داده‌ای برای ادغام یافت نشد.")
        return None
    
    # ادغام همه دیتافریم‌ها
    merged_df = pd.concat(all_dfs)
    
    # حذف ردیف‌های تکراری بر اساس ایندکس (زمان)
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
    
    # مرتب‌سازی بر اساس زمان
    merged_df.sort_index(inplace=True)
    
    # ایجاد پوشه data اگر وجود نداشت
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # ذخیره در فایل CSV
    output_path = os.path.join(data_dir, output_file)
    merged_df.to_csv(output_path)
    
    print(f"داده‌های ادغام شده با موفقیت در {output_path} ذخیره شدند.")
    print(f"تعداد کل ساعت‌ها پس از ادغام: {len(merged_df)}")
    print(f"بازه زمانی: از {merged_df.index.min()} تا {merged_df.index.max()}")
    
    return merged_df

def convert_hourly_to_daily(hourly_data, output_file=None):
    """
    تبدیل داده‌های ساعتی به داده‌های روزانه
    Daily_Change: تغییرات روزانه (درصد)
    SMA_7 و SMA_30: میانگین متحرک ۷ و ۳۰ روزه
    Volatility_7d: نوسانات ۷ روزه

    Args:
        hourly_data (DataFrame): دیتافریم داده‌های ساعتی
        output_file (str, optional): نام فایل خروجی برای ذخیره داده‌های روزانه
        
    Returns:
        DataFrame: داده‌های روزانه
    """
    if hourly_data is None or hourly_data.empty:
        print("داده‌های ساعتی خالی هستند.")
        return None
    
    # اطمینان از اینکه ایندکس از نوع datetime است
    if not isinstance(hourly_data.index, pd.DatetimeIndex):
        print("ایندکس داده‌ها باید از نوع datetime باشد.")
        return None
    
    # تبدیل به داده‌های روزانه
    daily_data = pd.DataFrame()
    
    # گروه‌بندی بر اساس تاریخ (بدون ساعت)
    grouped = hourly_data.groupby(hourly_data.index.date)
    
    # محاسبه OHLCV روزانه
    if 'Open' in hourly_data.columns:
        daily_data['Open'] = grouped['Open'].first()
    if 'High' in hourly_data.columns:
        daily_data['High'] = grouped['High'].max()
    if 'Low' in hourly_data.columns:
        daily_data['Low'] = grouped['Low'].min()
    if 'Close' in hourly_data.columns:
        daily_data['Close'] = grouped['Close'].last()
    if 'Volume' in hourly_data.columns:
        daily_data['Volume'] = grouped['Volume'].sum()
    
    # تبدیل ایندکس به datetime
    daily_data.index = pd.to_datetime(daily_data.index)
    
    # محاسبه ویژگی‌های اضافی
    if 'Close' in daily_data.columns and 'Open' in daily_data.columns:
        daily_data['Daily_Change'] = (daily_data['Close'] - daily_data['Open']) / daily_data['Open'] * 100
    
    if 'Close' in daily_data.columns:
        # میانگین متحرک ساده ۷ روزه
        daily_data['SMA_7'] = daily_data['Close'].rolling(window=7).mean()
        
        # میانگین متحرک ساده ۳۰ روزه
        daily_data['SMA_30'] = daily_data['Close'].rolling(window=30).mean()
        
        # نوسانات قیمت (انحراف معیار ۷ روزه)
        daily_data['Volatility_7d'] = daily_data['Close'].rolling(window=7).std()
    
    # ذخیره در فایل CSV اگر نام فایل مشخص شده باشد
    if output_file:
        # ایجاد پوشه data اگر وجود نداشت
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # ذخیره در فایل CSV
        output_path = os.path.join(data_dir, output_file)
        daily_data.to_csv(output_path)
        print(f"داده‌های روزانه با موفقیت در {output_path} ذخیره شدند.")
    
    print(f"تعداد کل روزها: {len(daily_data)}")
    print(f"بازه زمانی: از {daily_data.index.min()} تا {daily_data.index.max()}")
    
    return daily_data

def extract_daily_min_max_hours(hourly_data, output_file=None):
    """
    استخراج ساعت و مقدار کمترین و بیشترین قیمت بسته شدن در هر روز

    Min_Hour و Min_Close: ساعت و مقدار کمینه قیمت
    Max_Hour و Max_Close: ساعت و مقدار بیشینه قیمت
    Daily_Range و Daily_Range_Percent: دامنه تغییرات روزانه (مطلق و درصد)
    
    Args:
        hourly_data (DataFrame): دیتافریم داده‌های ساعتی
        output_file (str, optional): نام فایل خروجی برای ذخیره نتایج
        
    Returns:
        DataFrame: داده‌های روزانه با ساعت و مقدار کمترین و بیشترین قیمت
    """
    if hourly_data is None or hourly_data.empty:
        print("داده‌های ساعتی خالی هستند.")
        return None
    
    # اطمینان از اینکه ایندکس از نوع datetime است
    if not isinstance(hourly_data.index, pd.DatetimeIndex):
        print("ایندکس داده‌ها باید از نوع datetime باشد.")
        return None
    
    # اطمینان از وجود ستون Close
    if 'Close' not in hourly_data.columns:
        print("ستون 'Close' در داده‌ها وجود ندارد.")
        return None
    
    # ایجاد ستون‌های تاریخ و ساعت
    hourly_data = hourly_data.copy()
    hourly_data['Date'] = hourly_data.index.date
    hourly_data['Hour'] = hourly_data.index.hour
    
    # ایجاد دیتافریم نتیجه
    result = pd.DataFrame()
    
    # گروه‌بندی بر اساس تاریخ
    for date, group in hourly_data.groupby('Date'):
        # یافتن ردیف با کمترین قیمت بسته شدن
        min_row = group.loc[group['Close'].idxmin()]
        # یافتن ردیف با بیشترین قیمت بسته شدن
        max_row = group.loc[group['Close'].idxmax()]
        
        # محاسبه فاصله زمانی بین کمینه و بیشینه (به ساعت)
        time_diff = abs(max_row['Hour'] - min_row['Hour'])
        if time_diff > 12:  # اگر از نیمه‌شب عبور کرده باشد
            time_diff = 24 - time_diff
        
        # ایجاد ردیف جدید برای نتیجه
        new_row = {
            'Date': date,
            'Min_Hour': min_row['Hour'],
            'Min_Close': min_row['Close'],
            'Max_Hour': max_row['Hour'],
            'Max_Close': max_row['Close'],
            'Daily_Range': max_row['Close'] - min_row['Close'],
            'Daily_Range_Percent': (max_row['Close'] - min_row['Close']) / min_row['Close'] * 100,
            'Time_Diff_Hours': time_diff,  # فاصله زمانی بین کمینه و بیشینه
            'Min_Before_Max': 1 if min_row['Hour'] < max_row['Hour'] else 0  # آیا کمینه قبل از بیشینه رخ داده است؟
        }
        
        # اضافه کردن به دیتافریم نتیجه
        result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
    
    # تبدیل ستون Date به datetime
    result['Date'] = pd.to_datetime(result['Date'])
    
    # تنظیم Date به عنوان ایندکس
    result.set_index('Date', inplace=True)
    
    # مرتب‌سازی بر اساس تاریخ
    result.sort_index(inplace=True)
    
    # محاسبه همبستگی‌ها
    min_hour_corr = result['Min_Hour'].corr(result['Daily_Range_Percent'])
    max_hour_corr = result['Max_Hour'].corr(result['Daily_Range_Percent'])
    time_diff_corr = result['Time_Diff_Hours'].corr(result['Daily_Range_Percent'])
    
    # چاپ نتایج همبستگی
    print(f"\nهمبستگی ساعت کمینه با دامنه تغییرات: {min_hour_corr:.4f}")
    print(f"همبستگی ساعت بیشینه با دامنه تغییرات: {max_hour_corr:.4f}")
    print(f"همبستگی فاصله زمانی کمینه-بیشینه با دامنه تغییرات: {time_diff_corr:.4f}")
    
    # محاسبه آمار توصیفی برای الگوهای زمانی
    min_before_max_pct = result['Min_Before_Max'].mean() * 100
    print(f"\nدرصد روزهایی که کمینه قبل از بیشینه رخ داده: {min_before_max_pct:.2f}%")
    
    # محاسبه میانگین دامنه تغییرات برای هر ساعت کمینه
    min_hour_avg_range = result.groupby('Min_Hour')['Daily_Range_Percent'].mean()
    print("\nمیانگین دامنه تغییرات برای هر ساعت کمینه:")
    print(min_hour_avg_range.sort_values(ascending=False).head(5))
    
    # محاسبه میانگین دامنه تغییرات برای هر ساعت بیشینه
    max_hour_avg_range = result.groupby('Max_Hour')['Daily_Range_Percent'].mean()
    print("\nمیانگین دامنه تغییرات برای هر ساعت بیشینه:")
    print(max_hour_avg_range.sort_values(ascending=False).head(5))
    
    # ذخیره در فایل CSV اگر نام فایل مشخص شده باشد
    if output_file:
        # ایجاد پوشه data اگر وجود نداشت
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # ذخیره در فایل CSV
        output_path = os.path.join(data_dir, output_file)
        result.to_csv(output_path)
        print(f"داده‌های ساعت کمترین و بیشترین قیمت روزانه با موفقیت در {output_path} ذخیره شدند.")
    
    print(f"تعداد کل روزها: {len(result)}")
    print(f"بازه زمانی: از {result.index.min()} تا {result.index.max()}")
    
    return result

def plot_min_max_hour_frequency(min_max_data, output_file=None):
    """
    ایجاد نمودار فراوانی ساعت کمترین و بیشترین قیمت
    دو نمودار میله‌ای (یکی برای کمینه، یکی برای بیشینه)
    دیتافریم با فراوانی ساعت‌ها    
    Args:
        min_max_data (DataFrame): دیتافریم حاصل از تابع extract_daily_min_max_hours
        output_file (str, optional): نام فایل خروجی برای ذخیره نمودار
        
    Returns:
        tuple: (فراوانی ساعت کمترین قیمت، فراوانی ساعت بیشترین قیمت)
    """
    if min_max_data is None or min_max_data.empty:
        print("داده‌های ورودی خالی هستند.")
        return None
    
    # اطمینان از وجود ستون‌های مورد نیاز
    required_columns = ['Min_Hour', 'Max_Hour']
    missing_columns = [col for col in required_columns if col not in min_max_data.columns]
    
    if missing_columns:
        print(f"ستون‌های {', '.join(missing_columns)} در داده‌ها وجود ندارند.")
        return None
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # محاسبه فراوانی ساعت‌های کمترین و بیشترین قیمت
    min_hour_freq = min_max_data['Min_Hour'].value_counts().sort_index()
    max_hour_freq = min_max_data['Max_Hour'].value_counts().sort_index()
    
    # اطمینان از وجود همه ساعت‌ها (0-23)
    all_hours = np.arange(24)
    min_hour_freq = min_hour_freq.reindex(all_hours, fill_value=0)
    max_hour_freq = max_hour_freq.reindex(all_hours, fill_value=0)
    
    # ایجاد نمودار
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # نمودار فراوانی ساعت کمترین قیمت
    ax1.bar(min_hour_freq.index, min_hour_freq.values, color='red', alpha=0.7)
    ax1.set_title('فراوانی ساعت کمترین قیمت', fontsize=14)
    ax1.set_xlabel('ساعت روز', fontsize=12)
    ax1.set_ylabel('تعداد روزها', fontsize=12)
    ax1.set_xticks(all_hours)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # اضافه کردن برچسب مقدار روی هر ستون
    for i, v in enumerate(min_hour_freq.values):
        if v > 0:
            ax1.text(i, v + 0.5, str(v), ha='center', fontsize=10)
    
    # نمودار فراوانی ساعت بیشترین قیمت
    ax2.bar(max_hour_freq.index, max_hour_freq.values, color='green', alpha=0.7)
    ax2.set_title('فراوانی ساعت بیشترین قیمت', fontsize=14)
    ax2.set_xlabel('ساعت روز', fontsize=12)
    ax2.set_ylabel('تعداد روزها', fontsize=12)
    ax2.set_xticks(all_hours)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # اضافه کردن برچسب مقدار روی هر ستون
    for i, v in enumerate(max_hour_freq.values):
        if v > 0:
            ax2.text(i, v + 0.5, str(v), ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # ذخیره نمودار اگر نام فایل مشخص شده باشد
    if output_file:
        # ایجاد پوشه data اگر وجود نداشت
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # ذخیره نمودار
        output_path = os.path.join(data_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"نمودار فراوانی ساعت کمترین و بیشترین قیمت با موفقیت در {output_path} ذخیره شد.")
    
    plt.show()
    
    # ایجاد دیتافریم فراوانی‌ها برای بازگشت
    freq_df = pd.DataFrame({
        'Min_Hour_Frequency': min_hour_freq.values,
        'Max_Hour_Frequency': max_hour_freq.values
    }, index=all_hours)
    
    # چاپ خلاصه آماری
    print("\nفراوانی ساعت کمترین قیمت:")
    print(min_hour_freq)
    print("\nفراوانی ساعت بیشترین قیمت:")
    print(max_hour_freq)
    
    # ساعت‌های با بیشترین فراوانی
    most_frequent_min_hour = min_hour_freq.idxmax()
    most_frequent_max_hour = max_hour_freq.idxmax()
    
    print(f"\nساعت با بیشترین فراوانی کمترین قیمت: {most_frequent_min_hour} (تعداد: {min_hour_freq.max()})")
    print(f"ساعت با بیشترین فراوانی بیشترین قیمت: {most_frequent_max_hour} (تعداد: {max_hour_freq.max()})")
    
    return freq_df

def plot_hour_price_correlations(min_max_data, output_file=None):
    """
    تحلیل و نمایش همبستگی بین ساعت کمینه/بیشینه و تغییرات قیمت
    
    Args:
        min_max_data (DataFrame): دیتافریم حاصل از تابع extract_daily_min_max_hours
        output_file (str, optional): نام فایل خروجی برای ذخیره نمودار
        
    Returns:
        None
    """
    if min_max_data is None or min_max_data.empty:
        print("داده‌های ورودی خالی هستند.")
        return None
    
    # اطمینان از وجود ستون‌های مورد نیاز
    required_columns = ['Min_Hour', 'Max_Hour', 'Daily_Range_Percent', 'Time_Diff_Hours']
    missing_columns = [col for col in required_columns if col not in min_max_data.columns]
    
    if missing_columns:
        print(f"ستون‌های {', '.join(missing_columns)} در داده‌ها وجود ندارند.")
        return None
    
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    # ایجاد نمودار
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. نمودار پراکندگی ساعت کمینه و دامنه تغییرات
    sns.scatterplot(x='Min_Hour', y='Daily_Range_Percent', data=min_max_data, ax=axes[0, 0], alpha=0.6)
    axes[0, 0].set_title('همبستگی ساعت کمینه و دامنه تغییرات', fontsize=14)
    axes[0, 0].set_xlabel('ساعت کمینه', fontsize=12)
    axes[0, 0].set_ylabel('دامنه تغییرات (%)', fontsize=12)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # خط روند
    z = np.polyfit(min_max_data['Min_Hour'], min_max_data['Daily_Range_Percent'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(np.arange(0, 24), p(np.arange(0, 24)), "r--", alpha=0.8)
    corr = min_max_data['Min_Hour'].corr(min_max_data['Daily_Range_Percent'])
    axes[0, 0].text(0.05, 0.95, f'همبستگی: {corr:.4f}', transform=axes[0, 0].transAxes, 
                   fontsize=12, verticalalignment='top')
    
    # 2. نمودار پراکندگی ساعت بیشینه و دامنه تغییرات
    sns.scatterplot(x='Max_Hour', y='Daily_Range_Percent', data=min_max_data, ax=axes[0, 1], alpha=0.6)
    axes[0, 1].set_title('همبستگی ساعت بیشینه و دامنه تغییرات', fontsize=14)
    axes[0, 1].set_xlabel('ساعت بیشینه', fontsize=12)
    axes[0, 1].set_ylabel('دامنه تغییرات (%)', fontsize=12)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # خط روند
    z = np.polyfit(min_max_data['Max_Hour'], min_max_data['Daily_Range_Percent'], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(np.arange(0, 24), p(np.arange(0, 24)), "r--", alpha=0.8)
    corr = min_max_data['Max_Hour'].corr(min_max_data['Daily_Range_Percent'])
    axes[0, 1].text(0.05, 0.95, f'همبستگی: {corr:.4f}', transform=axes[0, 1].transAxes, 
                   fontsize=12, verticalalignment='top')
    
    # 3. نمودار پراکندگی فاصله زمانی و دامنه تغییرات
    sns.scatterplot(x='Time_Diff_Hours', y='Daily_Range_Percent', data=min_max_data, ax=axes[1, 0], alpha=0.6)
    axes[1, 0].set_title('همبستگی فاصله زمانی کمینه-بیشینه و دامنه تغییرات', fontsize=14)
    axes[1, 0].set_xlabel('فاصله زمانی (ساعت)', fontsize=12)
    axes[1, 0].set_ylabel('دامنه تغییرات (%)', fontsize=12)
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # خط روند
    z = np.polyfit(min_max_data['Time_Diff_Hours'], min_max_data['Daily_Range_Percent'], 1)
    p = np.poly1d(z)
    axes[1, 0].plot(np.arange(0, 13), p(np.arange(0, 13)), "r--", alpha=0.8)
    corr = min_max_data['Time_Diff_Hours'].corr(min_max_data['Daily_Range_Percent'])
    axes[1, 0].text(0.05, 0.95, f'همبستگی: {corr:.4f}', transform=axes[1, 0].transAxes, 
                   fontsize=12, verticalalignment='top')
    
    # 4. نمودار میانگین دامنه تغییرات برای هر ساعت کمینه
    min_hour_avg_range = min_max_data.groupby('Min_Hour')['Daily_Range_Percent'].mean()
    min_hour_avg_range = min_hour_avg_range.reindex(np.arange(24), fill_value=0)
    axes[1, 1].bar(min_hour_avg_range.index, min_hour_avg_range.values, alpha=0.7, color='purple')
    axes[1, 1].set_title('میانگین دامنه تغییرات برای هر ساعت کمینه', fontsize=14)
    axes[1, 1].set_xlabel('ساعت کمینه', fontsize=12)
    axes[1, 1].set_ylabel('میانگین دامنه تغییرات (%)', fontsize=12)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # ذخیره نمودار اگر نام فایل مشخص شده باشد
    if output_file:
        # ایجاد پوشه data اگر وجود نداشت
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # ذخیره نمودار
        output_path = os.path.join(data_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"نمودار همبستگی‌ها با موفقیت در {output_path} ذخیره شد.")
    
    plt.show()
    
    # محاسبه ماتریس همبستگی
    corr_columns = ['Min_Hour', 'Max_Hour', 'Time_Diff_Hours', 'Daily_Range_Percent']
    corr_matrix = min_max_data[corr_columns].corr()
    
    print("\nماتریس همبستگی:")
    print(corr_matrix)
    
    # نمایش ماتریس همبستگی به صورت نموداری
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.4f', linewidths=0.5)
    plt.title('ماتریس همبستگی متغیرها', fontsize=14)
    
    # ذخیره نمودار ماتریس همبستگی
    if output_file:
        corr_output_path = os.path.join(data_dir, f"correlation_matrix_{output_file}")
        plt.savefig(corr_output_path, dpi=300, bbox_inches='tight')
        print(f"ماتریس همبستگی با موفقیت در {corr_output_path} ذخیره شد.")
    
    plt.show()


def analyze_weekly_patterns(min_max_data):
    min_max_data = min_max_data.copy()
    min_max_data['Day_of_Week'] = min_max_data.index.day_name()
    min_by_day = min_max_data.groupby('Day_of_Week')['Min_Hour'].mean()
    max_by_day = min_max_data.groupby('Day_of_Week')['Max_Hour'].mean()
    print("\nمیانگین ساعت کمینه به تفکیک روز هفته:")
    print(min_by_day)
    print("\nمیانگین ساعت بیشینه به تفکیک روز هفته:")
    print(max_by_day)
    return min_by_day, max_by_day


# مثال استفاده
if __name__ == "__main__":
    # مثال ۱: دریافت داده‌های نوبیتکس
    # اطلاعات ورودی
    start_date = "1403-01-01"
    end_date = "1403-12-30"
    symbol = "BTCIRT"
    resolution = "360"
    api_token = "YOUR_API_TOKEN_HERE"  # توکن خودت رو جایگزین کن
    # فراخوانی تابع
    result_df = fetch_nobitex_ohlc(start_date, end_date, symbol, resolution, api_token)
    if not result_df.empty:
        print(result_df.head())
        save_data_to_csv(result_df, "nobitex_1403_360.csv")
        print(f"تعداد کل ردیف‌ها: {len(result_df)}")
    
    # مثال ۲: خواندن داده‌های ساعتی از فایل CSV
    # hourly_data = load_hourly_data("nobitex_1403-01-01_to_1403-12-30_60.csv")
    # if hourly_data is not None:
    #     print(hourly_data.head())
    
    # مثال ۳: ادغام چندین فایل CSV ساعتی
    # files_to_merge = [
    #     "nobitex_1403-01-01_to_1403-03-31_60.csv",
    #     "nobitex_1403-04-01_to_1403-06-31_60.csv",
    #     "nobitex_1403-07-01_to_1403-09-30_60.csv",
    #     "nobitex_1403-10-01_to_1403-12-30_60.csv"
    # ]
    # merged_data = merge_hourly_data_files(files_to_merge, "merged_1403_hourly.csv")
    
    # مثال ۴: تبدیل داده‌های ساعتی به روزانه
    # if merged_data is not None:
    #     daily_data = convert_hourly_to_daily(merged_data, "btc_1403_daily.csv")
    #     if daily_data is not None:
    #         print(daily_data.head())
    
    # مثال ۵: استخراج ساعت کمترین و بیشترین قیمت بسته شدن در هر روز
    # hourly_data = load_hourly_data("nobitex_1403-01-01_to_1403-12-30_60.csv")
    # if hourly_data is not None:
    #     min_max_hours = extract_daily_min_max_hours(hourly_data, "btc_daily_min_max_hours.csv")
    #     if min_max_hours is not None:
    #         print("\nنمونه داده‌های ساعت کمترین و بیشترین قیمت روزانه:")
    #         print(min_max_hours.head())
    #         print("\nآمار توصیفی:")
    #         print(min_max_hours.describe())
    
    # مثال ۶: ایجاد نمودار فراوانی ساعت کمترین و بیشترین قیمت
    # hourly_data = load_hourly_data("nobitex_1403-01-01_to_1403-12-30_60.csv")
    # if hourly_data is not None:
    #     min_max_hours = extract_daily_min_max_hours(hourly_data)
    #     if min_max_hours is not None:
    #         freq_df = plot_min_max_hour_frequency(min_max_hours, "btc_min_max_hour_frequency.png")
    #         if freq_df is not None:
    #             print("\nفراوانی ساعت‌های کمترین و بیشترین قیمت:")
    #             print(freq_df)
    
    # مثال ۷: تحلیل و نمایش همبستگی بین ساعت کمینه/بیشینه و تغییرات قیمت
    hourly_data = load_hourly_data("nobitex_1403-01-01_to_1403-12-30_60.csv")
    if hourly_data is not None:
        min_max_hours = extract_daily_min_max_hours(hourly_data,"extracted_daily_min_max_hours_1403-01-01_to_1403-12-30_60.csv")
        if min_max_hours is not None:
            plot_hour_price_correlations(min_max_hours, "btc_hour_price_correlations.png")
            plot_min_max_hour_frequency(min_max_hours, "btc_min_max_hour_frequency.png")
    
    if min_max_hours is not None:
        analyze_weekly_patterns(min_max_hours)
        
    print("برای استفاده از توابع، کامنت‌های مربوطه را از حالت کامنت خارج کنید.")