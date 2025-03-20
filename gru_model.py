import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import os
import joblib
from tensorflow.keras.models import save_model
import datetime
import random
import string
import shutil
import psutil
import tensorflow as tf
import time

# توابع کمکی و اصلی بدون تغییر (فقط معماری مدل تغییر می‌کنه)

def create_random_run_name():
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{random_str}"

def save_model_and_scalers(model, scaler_X, scaler_y, model_params, run_name, save_dir='saved_models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    run_dir = os.path.join(save_dir, run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    model_path = os.path.join(run_dir, 'gru_model.h5')  # تغییر نام فایل به gru_model
    save_model(model, model_path)
    scaler_X_path = os.path.join(run_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(run_dir, 'scaler_y.pkl')
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    params_path = os.path.join(run_dir, 'model_params.json')
    import json
    with open(params_path, 'w') as f:
        json.dump(model_params, f)
    current_file = os.path.abspath(__file__)
    model_file_dest = os.path.join(run_dir, 'gru_model.py')  # تغییر نام فایل کد
    shutil.copy2(current_file, model_file_dest)
    print(f"\nمدل و اسکالرها در مسیر {run_dir} ذخیره شدند:")
    print(f"- مدل: {model_path}")
    print(f"- اسکالر X: {scaler_X_path}")
    print(f"- اسکالر y: {scaler_y_path}")
    print(f"- پارامترها: {params_path}")
    print(f"- فایل کد: {model_file_dest}")
    return run_dir

def save_plots_and_report(run_dir, data, train_predictions, test_predictions, y_train_original, y_test_original,
                          future_predictions, future_dates, model, scaler_X, scaler_y, model_params,
                          train_metrics, test_metrics):
    start_time = time.time()
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_additional_analysis(data, train_predictions, test_predictions, y_train_original, y_test_original,
                           future_predictions, future_dates, len(y_train_original), model_params['lag_days'], plots_dir)
    report_path = os.path.join(run_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("گزارش ارزیابی مدل پیش‌بینی قیمت بیت‌کوین با GRU\n")  # تغییر متن گزارش
        f.write("==================================================\n\n")
        # بقیه بخش‌های گزارش بدون تغییر (مثل اطلاعات دیتاست، معیارها، و غیره)

def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def predict_future_days(model, last_sequence, scaler_X, scaler_y, future_days=30):
    future_predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(future_days):
        next_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
        future_predictions.append(next_pred[0])
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_pred[0]
    future_predictions = np.array(future_predictions)
    future_predictions = scaler_y.inverse_transform(future_predictions)
    return future_predictions

def generate_evaluation_report(train_metrics, test_metrics, future_predictions, future_dates, model_params):
    print("\n==================================================")
    print("گزارش ارزیابی مدل پیش‌بینی قیمت بیت‌کوین با GRU")  # تغییر متن گزارش
    print("==================================================\n")
    # بقیه بخش‌های گزارش بدون تغییر

def plot_additional_analysis(data, train_predictions, test_predictions, y_train_original, y_test_original,
                           future_predictions, future_dates, train_size, lag_days, run_dir):
    # نمودار قیمت واقعی و پیش‌بینی شده
    plt.figure(figsize=(14, 7))
    plt.plot(y_train_original, label='قیمت واقعی (داده‌های آموزشی)')
    plt.plot(train_predictions, label='پیش‌بینی (داده‌های آموزشی)')
    plt.title('مقایسه قیمت واقعی و پیش‌بینی شده برای داده‌های آموزشی')
    plt.xlabel('زمان')
    plt.ylabel('قیمت (USD)')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'train_predictions.png'))
    plt.close()

    # نمودار خطای پیش‌بینی
    plt.figure(figsize=(14, 7))
    train_errors = np.abs(y_train_original - train_predictions)
    test_errors = np.abs(y_test_original - test_predictions)
    plt.plot(train_errors, label='خطای داده‌های آموزشی')
    plt.plot(test_errors, label='خطای داده‌های تست')
    plt.title('خطای پیش‌بینی در طول زمان')
    plt.xlabel('زمان')
    plt.ylabel('خطای مطلق (USD)')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'prediction_errors.png'))
    plt.close()

    # نمودار توزیع خطاها
    plt.figure(figsize=(10, 6))
    plt.hist(train_errors, bins=50, alpha=0.5, label='داده‌های آموزشی')
    plt.hist(test_errors, bins=50, alpha=0.5, label='داده‌های تست')
    plt.title('توزیع خطاهای پیش‌بینی')
    plt.xlabel('خطای مطلق (USD)')
    plt.ylabel('تعداد')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'error_distribution.png'))
    plt.close()

    # نمودار پیش‌بینی آینده
    plt.figure(figsize=(14, 7))
    plt.plot(future_dates, future_predictions, '--', label='پیش‌بینی آینده')
    plt.title('پیش‌بینی قیمت بیت‌کوین برای ۳۰ روز آینده')
    plt.xlabel('تاریخ')
    plt.ylabel('قیمت (USD)')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'future_predictions.png'))
    plt.close()

def load_data(file_path):
    df = pd.read_csv(file_path, skiprows=[1, 2])
    df = df.rename(columns={'Price': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    numeric_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'Daily Return']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    df = df.dropna()
    return df

def predict_with_gru(data, lag_days=10, epochs=100, batch_size=32, validation_split=0.1,
                    dropout=0.2, early_stopping_patience=10, restore_best_weights=True,
                    optimizer='adam', loss='mse', save_model=True):
    start_time = time.time()
    run_name = create_random_run_name()
    run_dir = os.path.join('saved_models', run_name)
    os.makedirs(run_dir, exist_ok=True)
    model_params = {
        'lag_days': lag_days,
        'epochs': epochs,
        'batch_size': batch_size,
        'validation_split': validation_split,
        'dropout': dropout,
        'early_stopping_patience': early_stopping_patience,
        'restore_best_weights': restore_best_weights,
        'optimizer': optimizer,
        'loss': loss,
        'run_name': run_name
    }
    data['Target'] = data['Close'].shift(-1)
    for i in range(1, lag_days + 1):
        data[f'Close_Lag_{i}'] = data['Close'].shift(i)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Daily_Change'] = (data['Close'] - data['Open']) / data['Open'] * 100
    data['RSI'] = calculate_rsi(data)
    data['Volatility'] = data['Close'].rolling(window=14).std()
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    total_records = len(data)
    train_size = int(total_records * 0.88)
    test_size = total_records - train_size
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    print("تاریخ شروع داده‌های آموزشی:", train_data.index[0])
    print("تاریخ پایان داده‌های آموزشی:", train_data.index[-1])
    print("تاریخ شروع داده‌های تست:", test_data.index[0])
    print("تاریخ پایان داده‌های تست:", test_data.index[-1])
    X = data[[f'Close_Lag_{i}' for i in range(1, lag_days + 1)] + ['Volume', 'SMA_50', 'SMA_200', 'Daily_Change', 'RSI', 'Volatility']]
    y = data['Target']
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    X_lstm_train = []
    y_lstm_train = []
    for i in range(lag_days, len(X_train_scaled)):
        X_lstm_train.append(X_train_scaled[i-lag_days:i])
        y_lstm_train.append(y_train_scaled[i])
    X_lstm_test = []
    y_lstm_test = []
    for i in range(lag_days, len(X_test_scaled)):
        X_lstm_test.append(X_test_scaled[i-lag_days:i])
        y_lstm_test.append(y_test_scaled[i])
    X_lstm_train, y_lstm_train = np.array(X_lstm_train), np.array(y_lstm_train)
    X_lstm_test, y_lstm_test = np.array(X_lstm_test), np.array(y_lstm_test)
    print(f"Length of X_lstm_train: {len(X_lstm_train)}, Length of y_lstm_train: {len(y_lstm_train)}")
    print(f"Length of X_lstm_test: {len(X_lstm_test)}, Length of y_lstm_test: {len(y_lstm_test)}")
    
    # تغییر معماری به GRU
    model = Sequential([
        Input(shape=(lag_days, X.shape[1])),
        GRU(32, activation='tanh'),  # جایگزینی LSTM با GRU
        Dropout(0.2),
        Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=model_params['early_stopping_patience'],
        restore_best_weights=model_params['restore_best_weights']
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    history = model.fit(
        X_lstm_train, y_lstm_train,
        epochs=model_params['epochs'],
        batch_size=model_params['batch_size'],
        validation_split=model_params['validation_split'],
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    processing_time = time.time() - start_time
    print(f"\nزمان کل پردازش: {processing_time:.2f} ثانیه")
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'loss_plot.png'))
    plt.show()
    train_predictions = model.predict(X_lstm_train)
    test_predictions = model.predict(X_lstm_test)
    train_predictions = scaler_y.inverse_transform(train_predictions)
    test_predictions = scaler_y.inverse_transform(test_predictions)
    y_train_original = scaler_y.inverse_transform(y_lstm_train)
    y_test_original = scaler_y.inverse_transform(y_lstm_test)
    train_mse = mean_squared_error(y_train_original, train_predictions)
    test_mse = mean_squared_error(y_test_original, test_predictions)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train_original, train_predictions)
    test_mae = mean_absolute_error(y_test_original, test_predictions)
    train_mape = np.mean(np.abs((y_train_original - train_predictions) / y_train_original)) * 100
    test_mape = np.mean(np.abs((y_test_original - test_predictions) / y_test_original)) * 100
    train_r2 = r2_score(y_train_original, train_predictions)
    test_r2 = r2_score(y_test_original, test_predictions)
    def calculate_direction_accuracy(y_true, y_pred):
        true_direction = np.sign(np.diff(y_true.flatten()))
        pred_direction = np.sign(np.diff(y_pred.flatten()))
        return np.mean(true_direction == pred_direction) * 100
    train_direction_accuracy = calculate_direction_accuracy(y_train_original, train_predictions)
    test_direction_accuracy = calculate_direction_accuracy(y_test_original, test_predictions)
    train_max_error = np.max(np.abs(y_train_original - train_predictions))
    test_max_error = np.max(np.abs(y_test_original - test_predictions))
    train_median_error = np.median(np.abs(y_train_original - train_predictions))
    test_median_error = np.median(np.abs(y_test_original - test_predictions))
    def calculate_volatility_ratio(y_true, ypred):
        true_volatility = np.std(np.diff(y_true.flatten()))
        pred_volatility = np.std(np.diff(ypred.flatten()))
        return pred_volatility / true_volatility if true_volatility != 0 else 0
    def calculate_sharpe_ratio(y_true, y_pred):
        returns = np.diff(y_pred.flatten()) / y_pred[:-1].flatten()
        return np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) != 0 else 0
    def calculate_win_rate(y_true, y_pred):
        true_returns = np.diff(y_true.flatten())
        pred_returns = np.diff(y_pred.flatten())
        correct_predictions = np.sum((true_returns > 0) == (pred_returns > 0))
        return (correct_predictions / len(true_returns)) * 100
    def calculate_profit_factor(y_true, y_pred):
        true_returns = np.diff(y_true.flatten())
        pred_returns = np.diff(y_pred.flatten())
        correct_predictions = (true_returns > 0) == (pred_returns > 0)
        profits = np.sum(np.abs(true_returns[correct_predictions]))
        losses = np.sum(np.abs(true_returns[~correct_predictions]))
        return profits / losses if losses != 0 else float('inf')
    train_volatility_ratio = calculate_volatility_ratio(y_train_original, train_predictions)
    test_volatility_ratio = calculate_volatility_ratio(y_test_original, test_predictions)
    train_sharpe_ratio = calculate_sharpe_ratio(y_train_original, train_predictions)
    test_sharpe_ratio = calculate_sharpe_ratio(y_test_original, test_predictions)
    train_win_rate = calculate_win_rate(y_train_original, train_predictions)
    test_win_rate = calculate_win_rate(y_test_original, test_predictions)
    train_profit_factor = calculate_profit_factor(y_train_original, train_predictions)
    test_profit_factor = calculate_profit_factor(y_test_original, test_predictions)
    train_metrics = {
        'MSE': train_mse, 'RMSE': train_rmse, 'MAE': train_mae, 'MAPE': train_mape,
        'R2': train_r2, 'Direction_Accuracy': train_direction_accuracy,
        'Max_Error': train_max_error, 'Median_Error': train_median_error,
        'Volatility_Ratio': train_volatility_ratio, 'Sharpe_Ratio': train_sharpe_ratio,
        'Win_Rate': train_win_rate, 'Profit_Factor': train_profit_factor
    }
    test_metrics = {
        'MSE': test_mse, 'RMSE': test_rmse, 'MAE': test_mae, 'MAPE': test_mape,
        'R2': test_r2, 'Direction_Accuracy': test_direction_accuracy,
        'Max_Error': test_max_error, 'Median_Error': test_median_error,
        'Volatility_Ratio': test_volatility_ratio, 'Sharpe_Ratio': test_sharpe_ratio,
        'Win_Rate': test_win_rate, 'Profit_Factor': test_profit_factor
    }
    last_sequence = X_test_scaled[-lag_days:]
    future_predictions = predict_future_days(model, last_sequence, scaler_X, scaler_y)
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[train_size + lag_days:train_size + lag_days + len(y_test_original)], y_test_original, label='Actual Price')
    plt.plot(data.index[train_size + lag_days:train_size + lag_days + len(y_test_original)], test_predictions, label='Predicted Price')
    plt.plot(future_dates, future_predictions, '--', label='Future Predictions (30 days)')
    plt.title('Bitcoin Price Prediction with GRU')  # تغییر عنوان به GRU
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'price_prediction_plot.png'))
    plt.show()
    print("\nپیش‌بینی قیمت برای ۳۰ روز آینده:")
    for date, price in zip(future_dates, future_predictions):
        print(f"{date.strftime('%Y-%m-%d')}: ${price[0]:.2f}")
    if save_model:
        run_dir = save_model_and_scalers(model, scaler_X, scaler_y, model_params, run_name)
        save_plots_and_report(run_dir, data, train_predictions, test_predictions, y_train_original, y_test_original,
                            future_predictions, future_dates, model, scaler_X, scaler_y, model_params,
                            train_metrics, test_metrics)
    return model, scaler_X, scaler_y, model_params

if __name__ == "__main__":
    data = load_data('bitcoin_data.csv')
    model, scaler_X, scaler_y, model_params = predict_with_gru(
        data,
        lag_days=10,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        dropout=0.2,
        early_stopping_patience=10,
        restore_best_weights=True,
        optimizer='adam',
        loss='mse',
        save_model=True
    )