from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def predict_with_random_forest(data, lag_days=5):
    data['Target'] = data['Close'].shift(-1)
    for i in range(1, lag_days + 1):
        data[f'Close_Lag_{i}'] = data['Close'].shift(i)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data = data.dropna()
    
    X = data[[f'Close_Lag_{i}' for i in range(1, lag_days + 1)] + ['Volume', 'SMA_50', 'SMA_200']]
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    print(f"Train MSE (Random Forest): {train_mse}")
    print(f"Test MSE (Random Forest): {test_mse}")
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test, label='Actual Price')
    plt.plot(y_test.index, test_predictions, label='Predicted Price')
    plt.title('Bitcoin Price Prediction with Random Forest')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()