import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error

def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def predict_with_gru(data, lag_days=10, epochs=100, train_ratio=0.8):
    data['Target'] = data['Close'].shift(-1)
    for i in range(1, lag_days + 1):
        data[f'Close_Lag_{i}'] = data['Close'].shift(i)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Daily_Change'] = (data['Close'] - data['Open']) / data['Open'] * 100
    data['RSI'] = calculate_rsi(data)
    data['Volatility'] = data['Close'].rolling(window=14).std()
    data = data.dropna()
    
    X = data[[f'Close_Lag_{i}' for i in range(1, lag_days + 1)] + ['Volume', 'SMA_50', 'SMA_200', 'Daily_Change', 'RSI', 'Volatility']]
    y = data['Target']
    
    # نرمال‌سازی فقط روی دیتای ترن
    train_size = int(len(X) * train_ratio)  # 80% برای ترن، 20% برای تست
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    # آماده‌سازی برای GRU
    X_gru_train = []
    y_gru_train = []
    for i in range(lag_days, len(X_train_scaled)):
        X_gru_train.append(X_train_scaled[i-lag_days:i])
        y_gru_train.append(y_train_scaled[i])
    X_gru_test = []
    y_gru_test = []
    for i in range(lag_days, len(X_test_scaled)):
        X_gru_test.append(X_test_scaled[i-lag_days:i])
        y_gru_test.append(y_test_scaled[i])
    X_gru_train, y_gru_train = np.array(X_gru_train), np.array(y_gru_train)
    X_gru_test, y_gru_test = np.array(X_gru_test), np.array(y_gru_test)
    
    print(f"Length of X_gru_train: {len(X_gru_train)}, Length of y_gru_train: {len(y_gru_train)}")
    print(f"Length of X_gru_test: {len(X_gru_test)}, Length of y_gru_test: {len(y_gru_test)}")
    
    # معماری مدل با GRU
    model = Sequential([
        Input(shape=(lag_days, X.shape[1])),
        GRU(100, activation='tanh', return_sequences=True),
        Dropout(0.3),
        GRU(50, activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_gru_train, y_gru_train, epochs=epochs, batch_size=32, validation_split=0.1, 
              callbacks=[early_stopping], verbose=1)
    
    train_predictions = model.predict(X_gru_train)
    test_predictions = model.predict(X_gru_test)
    
    train_predictions = scaler_y.inverse_transform(train_predictions)
    test_predictions = scaler_y.inverse_transform(test_predictions)
    y_train_original = scaler_y.inverse_transform(y_gru_train)
    y_test_original = scaler_y.inverse_transform(y_gru_test)
    
    train_mse = mean_squared_error(y_train_original, train_predictions)
    test_mse = mean_squared_error(y_test_original, test_predictions)
    train_r2 = r2_score(y_train_original, train_predictions)
    test_r2 = r2_score(y_test_original, test_predictions)
    train_mae = mean_absolute_error(y_train_original, train_predictions)
    test_mae = mean_absolute_error(y_test_original, test_predictions)

    print(f"Train MSE (GRU): {train_mse}")
    print(f"Test MSE (GRU): {test_mse}")
    print(f"Train R2 (GRU): {train_r2}")
    print(f"Test R2 (GRU): {test_r2}")
    print(f"Train MAE (GRU): {train_mae}")
    print(f"Test MAE (GRU): {test_mae}")

    # ذخیره مدل
    model.save('gru_model.h5')
    
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[train_size + lag_days:train_size + lag_days + len(y_test_original)], y_test_original, label='Actual Price')
    plt.plot(data.index[train_size + lag_days:train_size + lag_days + len(y_test_original)], test_predictions, label='Predicted Price')
    plt.title('Bitcoin Price Prediction with GRU')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()