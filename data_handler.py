import yfinance as yf
import pandas as pd

def fetch_data(ticker='BTC-USD', start_date='2010-01-01', end_date='2025-03-06',interval='1d'):
    print(f"Fetching {ticker} data from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date,interval=interval)
    # data = yf.download('BTC-USD', start='2018-01-01', end='2025-03-09', interval='1h')
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