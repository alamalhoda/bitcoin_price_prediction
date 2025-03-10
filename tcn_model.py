import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def create_sequences(data, seq_length):
    """تبدیل داده‌ها به توالی‌های مناسب برای TCN با پشتیبانی از چند ویژگی"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]  # فقط Close را پیش‌بینی می‌کنیم
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def predict_with_tcn(data, epochs=50, batch_size=32, seq_length=60):
    """پیش‌بینی قیمت بیت‌کوین با استفاده از TCN و چندین ویژگی"""
    # آماده‌سازی داده‌ها با ویژگی‌های بیشتر
    features = data[['Close', 'Open', 'High', 'Low', 'Volume']].values
    
    # نرمال‌سازی همه ویژگی‌ها
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # ایجاد توالی‌های داده
    X, y = create_sequences(scaled_features, seq_length)
    
    # تقسیم داده‌ها به مجموعه‌های آموزش و آزمون
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # ساخت مدل TCN با استفاده از Functional API
    input_layer = Input(shape=(seq_length, 5))  # 5 ویژگی
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
    
    # نمایش خلاصه مدل
    print("\nساختار مدل TCN:")
    model.summary()
    
    # آموزش مدل
    print("\nشروع آموزش مدل...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    # پیش‌بینی
    print("\nانجام پیش‌بینی...")
    y_pred = model.predict(X_test)
    
    # برگرداندن مقیاس پیش‌بینی‌ها به حالت اصلی
    # ایجاد ماتریس موقت برای inverse transform
    temp_array = np.zeros((len(y_test), 5))  # ماتریس 5 ستونی برای همه ویژگی‌ها
    temp_array[:, 0] = y_test  # قرار دادن مقادیر واقعی در ستون Close
    y_test_inv = scaler.inverse_transform(temp_array)[:, 0]  # فقط ستون Close را برمی‌گردانیم
    
    temp_array = np.zeros((len(y_pred), 5))
    temp_array[:, 0] = y_pred.flatten()
    y_pred_inv = scaler.inverse_transform(temp_array)[:, 0]
    
    # رسم نتایج
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_inv, label='مقادیر واقعی', alpha=0.8)
    plt.plot(y_pred_inv, label='پیش‌بینی TCN', alpha=0.8)
    plt.title('پیش‌بینی قیمت بیت‌کوین با استفاده از TCN چند متغیره')
    plt.xlabel('روز')
    plt.ylabel('قیمت (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # رسم نمودار loss
    plt.figure(figsize=(15, 6))
    plt.plot(history.history['loss'], label='خطای آموزش')
    plt.plot(history.history['val_loss'], label='خطای اعتبارسنجی')
    plt.title('روند خطای مدل در طول آموزش')
    plt.xlabel('epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # محاسبه و نمایش معیارهای ارزیابی
    mse = np.mean((y_test_inv - y_pred_inv) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_inv - y_pred_inv))
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    
    print(f'\nنتایج ارزیابی مدل:')
    print(f'میانگین مربعات خطا (MSE): {mse:.2f}')
    print(f'ریشه میانگین مربعات خطا (RMSE): {rmse:.2f}')
    print(f'میانگین قدر مطلق خطا (MAE): {mae:.2f}')
    print(f'درصد میانگین قدر مطلق خطا (MAPE): {mape:.2f}%')
    
    return model, history 