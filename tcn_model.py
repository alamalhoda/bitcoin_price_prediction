import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def create_sequences(data, seq_length):
    """تبدیل داده‌ها به توالی‌های مناسب برای TCN با پشتیبانی از چند ویژگی"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]  # فقط Close را پیش‌بینی می‌کنیم
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def predict_future_days(model, last_sequence, scaler, future_days=30):
    """پیش‌بینی قیمت برای روزهای آینده"""
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(future_days):
        next_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
        future_predictions.append(next_pred[0, 0])
        
        new_row = np.zeros((1, current_sequence.shape[1]))
        new_row[0, 0] = next_pred[0, 0]  # قیمت پیش‌بینی شده
        new_row[0, 1] = next_pred[0, 0]  # Open
        new_row[0, 2] = next_pred[0, 0] * 1.01  # High
        new_row[0, 3] = next_pred[0, 0] * 0.99  # Low
        new_row[0, 4] = current_sequence[-1, 4]  # Volume
        
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    return np.array(future_predictions)

def predict_with_tcn(data, epochs=25, batch_size=32, seq_length=60):
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
        nb_filters=32,
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
    
    # پیش‌بینی روزهای آینده
    print("\nپیش‌بینی قیمت برای روزهای آینده...")
    last_sequence = scaled_features[-seq_length:]
    future_predictions = predict_future_days(model, last_sequence, scaler, future_days=30)
    
    # برگرداندن مقیاس پیش‌بینی‌ها به حالت اصلی
    temp_array = np.zeros((len(y_test), 5))
    temp_array[:, 0] = y_test
    y_test_inv = scaler.inverse_transform(temp_array)[:, 0]
    
    temp_array = np.zeros((len(y_pred), 5))
    temp_array[:, 0] = y_pred.flatten()
    y_pred_inv = scaler.inverse_transform(temp_array)[:, 0]
    
    # برگرداندن مقیاس پیش‌بینی‌های آینده
    temp_array = np.zeros((len(future_predictions), 5))
    temp_array[:, 0] = future_predictions
    future_predictions_inv = scaler.inverse_transform(temp_array)[:, 0]
    
    # ایجاد تاریخ‌های آینده
    last_date = pd.to_datetime(data.index[-1])
    future_dates = [last_date + timedelta(days=x) for x in range(1, 31)]
    
    # رسم نتایج
    plt.figure(figsize=(15, 8))
    plt.plot(data.index[-len(y_test):], y_test_inv, label='مقادیر واقعی', alpha=0.8)
    plt.plot(data.index[-len(y_pred):], y_pred_inv, label='پیش‌بینی مدل', alpha=0.8)
    plt.plot(future_dates, future_predictions_inv, label='پیش‌بینی آینده', linestyle='--', alpha=0.8)
    
    plt.title('پیش‌بینی قیمت بیت‌کوین با استفاده از TCN چند متغیره')
    plt.xlabel('تاریخ')
    plt.ylabel('قیمت (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # نمودار جدید برای پیش‌بینی 30 روز آینده
    plt.figure(figsize=(15, 6))
    
    # محاسبه درصد تغییرات روزانه
    daily_changes = np.diff(future_predictions_inv) / future_predictions_inv[:-1] * 100
    
    # رسم قیمت پیش‌بینی شده
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    
    # نمودار قیمت
    ax1.plot(future_dates, future_predictions_inv, 'b-', label='قیمت پیش‌بینی شده', linewidth=2)
    ax1.fill_between(future_dates, 
                     future_predictions_inv * 0.95,  # حد پایین باند اطمینان
                     future_predictions_inv * 1.05,  # حد بالای باند اطمینان
                     alpha=0.2, color='blue')
    
    # اضافه کردن برچسب قیمت به نقاط مهم
    for i, (date, price) in enumerate(zip(future_dates, future_predictions_inv)):
        if i % 5 == 0:  # نمایش هر 5 روز
            ax1.annotate(f'${price:,.0f}', 
                        (date, price),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=8)
    
    ax1.set_title('پیش‌بینی قیمت بیت‌کوین برای 30 روز آینده')
    ax1.set_xlabel('تاریخ')
    ax1.set_ylabel('قیمت (USD)')
    ax1.grid(True)
    ax1.legend()
    
    # نمودار درصد تغییرات
    bars = ax2.bar(future_dates[1:], daily_changes, 
                   color=['g' if x >= 0 else 'r' for x in daily_changes],
                   alpha=0.6,
                   label='درصد تغییرات روزانه')
    
    # اضافه کردن برچسب درصد به نقاط مهم
    for i, (date, change) in enumerate(zip(future_dates[1:], daily_changes)):
        if abs(change) > 2:  # نمایش تغییرات بیشتر از 2 درصد
            ax2.annotate(f'{change:.1f}%', 
                        (date, change),
                        textcoords="offset points",
                        xytext=(0, 10 if change >= 0 else -15),
                        ha='center',
                        fontsize=8)
    
    ax2.set_title('درصد تغییرات روزانه پیش‌بینی شده')
    ax2.set_xlabel('تاریخ')
    ax2.set_ylabel('درصد تغییرات')
    ax2.grid(True)
    ax2.legend()
    
    # تنظیمات نهایی
    plt.xticks(rotation=45)
    plt.tight_layout()
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
    
    # نمایش پیش‌بینی‌های آینده
    print('\nپیش‌بینی قیمت برای 30 روز آینده:')
    for date, price in zip(future_dates, future_predictions_inv):
        print(f'{date.strftime("%Y-%m-%d")}: ${price:.2f}')
    
    return model, history 