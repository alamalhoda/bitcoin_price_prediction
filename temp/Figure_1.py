import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def create_sequences(data, seq_length):
    """تبدیل داده‌ها به توالی‌های مناسب برای TCN"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def predict_with_tcn(data, epochs=50, batch_size=32, seq_length=60):
    """پیش‌بینی قیمت بیت‌کوین با استفاده از TCN"""
    # آماده‌سازی داده‌ها
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # نرمال‌سازی داده‌ها
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(close_prices)
    
    # ایجاد توالی‌های داده
    X, y = create_sequences(scaled_prices, seq_length)
    
    # تقسیم داده‌ها به مجموعه‌های آموزش و آزمون
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # ساخت مدل TCN با استفاده از Functional API
    input_layer = Input(shape=(seq_length, 1))
    tcn_layer = TCN(
        nb_filters=64,
        kernel_size=2,
        nb_stacks=1,
        dilations=[1, 2, 4, 8, 16, 32],
        padding='causal',
        use_skip_connections=True,
        dropout_rate=0.2,
        return_sequences=False
    )(input_layer)
    output_layer = Dense(1)(tcn_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # آموزش مدل
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    # پیش‌بینی
    y_pred = model.predict(X_test)
    
    # برگرداندن مقیاس پیش‌بینی‌ها به حالت اصلی
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # رسم نتایج
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_inv, label='مقادیر واقعی')
    plt.plot(y_pred_inv, label='پیش‌بینی TCN')
    plt.title('پیش‌بینی قیمت بیت‌کوین با استفاده از TCN')
    plt.xlabel('روز')
    plt.ylabel('قیمت (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # محاسبه MSE
    mse = np.mean((y_test_inv - y_pred_inv) ** 2)
    print(f'میانگین مربعات خطا (MSE): {mse:.2f}')
    
    return model, history 