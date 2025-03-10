import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def predict_future_with_linear(data, lag_days=10, future_days=10):
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
    
    train_size = int(len(X) * 0.99)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    
    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)
    
    train_predictions = scaler_y.inverse_transform(train_predictions)
    test_predictions = scaler_y.inverse_transform(test_predictions)
    y_train_original = scaler_y.inverse_transform(y_train_scaled)
    y_test_original = scaler_y.inverse_transform(y_test_scaled)
    
    train_mse = mean_squared_error(y_train_original, train_predictions)
    test_mse = mean_squared_error(y_test_original, test_predictions)
    print(f"Train MSE (Linear Regression): {train_mse}")
    print(f"Test MSE (Linear Regression): {test_mse}")
    
    # پیش‌بینی برای روزهای آینده
    last_data = data.iloc[-lag_days:].copy()
    future_predictions = []
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='D')
    
    for i in range(future_days):
        last_X = last_data[[f'Close_Lag_{i}' for i in range(1, lag_days + 1)] + ['Volume', 'SMA_50', 'SMA_200', 'Daily_Change', 'RSI', 'Volatility']].tail(1)
        last_X = last_X.fillna(last_X.mean())
        last_X_scaled = scaler_X.transform(last_X)
        pred_scaled = model.predict(last_X_scaled)
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]
        future_predictions.append(pred)
        
        # به‌روزرسانی داده‌ها با میانگین متحرک ساده
        new_row = last_data.tail(1).copy()
        new_row.index = [last_data.index[-1] + pd.Timedelta(days=1)]
        new_row['Close'] = pred if i == 0 else np.mean([pred, last_data['Close'].iloc[-1]])
        for j in range(lag_days, 1, -1):
            new_row[f'Close_Lag_{j}'] = last_data[f'Close_Lag_{j-1}'].iloc[-1]
        new_row['Close_Lag_1'] = last_data['Close'].iloc[-1]
        new_row['SMA_50'] = last_data['Close'].tail(50).mean() if len(last_data) >= 50 else last_data['Close'].mean()
        new_row['SMA_200'] = last_data['Close'].tail(200).mean() if len(last_data) >= 200 else last_data['Close'].mean()
        new_row['Daily_Change'] = (new_row['Close'] - last_data['Open'].iloc[-1]) / last_data['Open'].iloc[-1] * 100
        new_row['RSI'] = calculate_rsi(pd.concat([last_data, new_row])[-14:]).iloc[-1]
        new_row['Volatility'] = last_data['Close'].tail(14).std() if len(last_data) >= 14 else last_data['Close'].mean()
        last_data = pd.concat([last_data, new_row])
    
    # نمایش نمودار با Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[train_size + lag_days:], y=y_test_original.flatten(), mode='lines', name='Actual Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index[train_size + lag_days:], y=test_predictions.flatten(), mode='lines', name='Predicted Price (Test)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Prediction', line=dict(color='red', dash='dash')))
    fig.update_layout(
        title='Bitcoin Price Prediction with Linear Regression + Future Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white'
    )
    fig.show()