import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import mplfinance as mpf

def plot_close_price(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Bitcoin Close Price')
    plt.title('Bitcoin Close Price (USD)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()

def plot_volume(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Volume'], label='Bitcoin Volume')
    plt.title('Bitcoin Volume Traded')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.show()

def plot_moving_averages(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Bitcoin Close Price')
    plt.plot(data['SMA_50'], label='50-Day SMA')
    plt.plot(data['SMA_200'], label='200-Day SMA')
    plt.title('Bitcoin Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

def plot_candlestick(data):
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Data must contain: {required_columns}")
    mpf.plot(data, type='candle', style='charles', title='Bitcoin Candlestick Chart', 
             ylabel='Price (USD)', volume=True)

def plot_daily_returns(data):
    data['Daily Return'] = data['Close'].pct_change()
    plt.figure(figsize=(14, 7))
    data['Daily Return'].hist(bins=50)
    plt.title('Histogram of Daily Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.show()