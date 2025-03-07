import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def predict_with_lstm(data, lag_days=5, epochs=50):
    data['Target'] = data['Close'].shift(-1)
    for i in range(1, lag_days + 1):
        data[f'Close_Lag_{i}'] = data['Close'].shift(i)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data = data.dropna()
    
    X = data[[f'Close_Lag_{i}' for i in range(1, lag_days + 1)] + ['Volume', 'SMA_50', 'SMA_200']]
    y = data['Target']
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    X_lstm = []
    y_lstm = []
    for i in range(lag_days, len(X_scaled) - 1):
        X_lstm.append(X_scaled[i-lag_days:i])
        y_lstm.append(y_scaled[i])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    
    train_size = int(len(X_lstm) * 0.8)
    X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
    y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(lag_days, X.shape[1]), return_sequences=True))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
    
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_predictions = scaler_y.inverse_transform(train_predictions)
    test_predictions = scaler_y.inverse_transform(test_predictions)
    y_train_original = scaler_y.inverse_transform(y_train)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    train_mse = mean_squared_error(y_train_original, train_predictions)
    test_mse = mean_squared_error(y_test_original, test_predictions)
    print(f"Train MSE (LSTM): {train_mse}")
    print(f"Test MSE (LSTM): {test_mse}")
    
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-len(y_test):], y_test_original, label='Actual Price')
    plt.plot(data.index[-len(y_test):], test_predictions, label='Predicted Price')
    plt.title('Bitcoin Price Prediction with LSTM')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()