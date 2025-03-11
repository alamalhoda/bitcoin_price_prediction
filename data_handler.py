import yfinance as yf
import pandas as pd
from datetime import datetime
import requests

def fetch_data(ticker='BTC-USD', start_date='2010-01-01', end_date='2025-03-06',interval='1d'):
    print(f"Fetching {ticker} data from {start_date} to {end_date}...")
    # data = yf.download(ticker, start=start_date, end=end_date,interval=interval)
    data = yf.download('BTC-USD', start=start_date, end=end_date, interval=interval)
    data.index.name = 'Date'
    return data

def load_data_from_csv(file_path='bitcoin_data.csv'):
    try:
        data = pd.read_csv(file_path, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)
        data.index.name = 'Date'
        print("Data loaded from CSV successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def save_data_to_csv(data, file_path='bitcoin_data.csv'):
    data.to_csv(file_path)
    print(f"Data saved to {file_path}.")

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