"""
مدل LSTM برای پیش‌بینی قیمت بیت‌کوین
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from datetime import datetime

def create_lstm_model(input_shape, lstm1_units=50, lstm2_units=30, dropout1=0.3, dropout2=0.3):
    """
    ایجاد مدل LSTM برای پیش‌بینی قیمت بیت‌کوین
    
    Args:
        input_shape (tuple): شکل داده‌های ورودی (تعداد گام‌های زمانی، تعداد ویژگی‌ها)
        lstm1_units (int): تعداد واحدهای لایه اول LSTM
        lstm2_units (int): تعداد واحدهای لایه دوم LSTM
        dropout1 (float): نرخ dropout لایه اول
        dropout2 (float): نرخ dropout لایه دوم
        
    Returns:
        keras.Model: مدل LSTM
    """
    model = Sequential()
    
    # لایه اول LSTM
    model.add(LSTM(units=lstm1_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout1))
    
    # لایه دوم LSTM
    model.add(LSTM(units=lstm2_units))
    model.add(Dropout(dropout2))
    
    # لایه خروجی
    model.add(Dense(units=1))
    
    # کامپایل مدل
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_lstm_model(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, 
                     patience=20, sample_weights=None, save_model=True, model_save_dir='output/models'):
    """
    آموزش مدل LSTM
    
    Args:
        X_train (numpy.array): داده‌های آموزش ورودی
        y_train (numpy.array): داده‌های آموزش هدف
        validation_split (float): نسبت داده‌های اعتبارسنجی
        epochs (int): تعداد دوره‌های آموزش
        batch_size (int): اندازه دسته
        patience (int): تعداد دوره‌های صبر برای توقف زودهنگام
        sample_weights (numpy.array, optional): وزن‌های نمونه‌ها
        save_model (bool): ذخیره مدل آموزش دیده
        model_save_dir (str): مسیر ذخیره مدل
        
    Returns:
        tuple: (model, history)
    """
    # ایجاد مدل LSTM
    model = create_lstm_model(input_shape=(X_train.shape[1], 1))
    
    # ایجاد callback برای توقف زودهنگام
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    callbacks = [early_stopping]
    
    # افزودن callback برای ذخیره بهترین مدل اگر درخواست شده باشد
    if save_model:
        # اطمینان از وجود دایرکتوری
        os.makedirs(model_save_dir, exist_ok=True)
        
        # ایجاد نام فایل با تاریخ و زمان
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_save_dir, f"lstm_model_{timestamp}.h5")
        
        # ایجاد callback
        model_checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        callbacks.append(model_checkpoint)
    
    # آموزش مدل
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        sample_weight=sample_weights,
        verbose=1
    )
    
    # چاپ خلاصه مدل
    model.summary()
    
    # اگر مدل ذخیره شده است، مسیر آن را نمایش بده
    if save_model:
        print(f"\nمدل با بهترین عملکرد در مسیر {model_path} ذخیره شد.")
    
    return model, history

def save_model(model, model_path):
    """
    ذخیره مدل در مسیر مشخص شده
    
    Args:
        model: مدل آموزش دیده
        model_path (str): مسیر ذخیره مدل
    """
    # اطمینان از وجود دایرکتوری
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # ذخیره مدل
    model.save(model_path)
    
    print(f"مدل با موفقیت در مسیر {model_path} ذخیره شد.")
    
def load_saved_model(model_path):
    """
    بارگذاری مدل ذخیره شده از مسیر مشخص شده
    
    Args:
        model_path (str): مسیر مدل ذخیره شده
        
    Returns:
        keras.Model: مدل بارگذاری شده
    """
    # بررسی وجود فایل مدل
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"فایل مدل در مسیر {model_path} یافت نشد!")
    
    # بارگذاری مدل
    model = load_model(model_path)
    
    print(f"مدل با موفقیت از مسیر {model_path} بارگذاری شد.")
    
    return model 