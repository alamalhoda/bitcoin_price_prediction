"""
پیاده‌سازی مدل LSTM برای پیش‌بینی قیمت بیت‌کوین
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D
from bitcoin_price_prediction.src.models.base_model import TimeSeriesModel
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

class LSTMModel(TimeSeriesModel):
    """
    پیاده‌سازی مدل LSTM برای پیش‌بینی قیمت بیت‌کوین
    """
    
    def build_model(self, input_shape):
        """
        ساخت معماری مدل LSTM
        
        Args:
            input_shape: شکل ورودی مدل (lag_days, n_features)
        
        Returns:
            model: مدل ساخته شده
        """
        self.model = Sequential([
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(
                self.config.LSTM1_UNITS,
                activation='tanh',
                return_sequences=True
            ),
            Dropout(self.config.DROPOUT1),
            LSTM(
                self.config.LSTM2_UNITS,
                activation='tanh'
            ),
            Dropout(self.config.DROPOUT2),
            Dense(1)
        ])
        
        self.model.summary()
        return self.model
    
    def predict_future_days(self, last_sequence, data, future_days=None):
        """
        پیش‌بینی قیمت برای روزهای آینده با به‌روزرسانی شاخص‌های تکنیکال
        
        Args:
            last_sequence: آخرین توالی داده‌های ورودی
            data: DataFrame اصلی داده‌ها
            future_days: تعداد روزهای آینده برای پیش‌بینی
        
        Returns:
            tuple: پیش‌بینی‌های قیمت، تاریخ‌های آینده
        """
        if future_days is None:
            future_days = self.config.FUTURE_DAYS
        
        if self.scaler_X is None or self.scaler_y is None:
            raise ValueError("اسکیلرها باید قبل از پیش‌بینی تنظیم شوند.")
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        # ایجاد یک کپی از داده‌های اصلی برای به‌روزرسانی شاخص‌ها
        future_data = data.copy()
        
        # آموزش مدل پیش‌بینی حجم معاملات
        volume_model = LinearRegression()
        volume_X = np.arange(len(data)).reshape(-1, 1)
        volume_y = data['Volume'].values
        volume_model.fit(volume_X, volume_y)
        
        # پیش‌بینی حجم معاملات برای روزهای آینده
        future_volumes = volume_model.predict(np.arange(len(data), len(data) + future_days).reshape(-1, 1))
        
        # تعداد ویژگی‌های مورد نیاز برای اسکالر
        n_features = self.scaler_X.n_features_in_
        lag_days = current_sequence.shape[0]  # تعداد روزهای گذشته
        
        # ایجاد لیست تاریخ‌های آینده
        last_date = future_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
        
        for i in range(future_days):
            # پیش‌بینی قیمت روز بعد
            next_pred = self.model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
            future_predictions.append(next_pred[0])
            
            # به‌روزرسانی قیمت در داده‌های آینده
            next_date = future_data.index[-1] + pd.Timedelta(days=1)
            next_price = self.scaler_y.inverse_transform(next_pred)[0][0]
            
            # اضافه کردن ردیف جدید به داده‌های آینده با حجم معاملات پیش‌بینی شده
            new_row = pd.DataFrame({
                'Close': next_price,
                'Open': next_price * 0.99,  # تخمین قیمت باز شدن
                'High': next_price * 1.01,  # تخمین بالاترین قیمت
                'Low': next_price * 0.99,   # تخمین پایین‌ترین قیمت
                'Volume': future_volumes[i],  # استفاده از حجم معاملات پیش‌بینی شده
                'Daily_Change': (next_price - future_data['Close'].iloc[-1]) / future_data['Close'].iloc[-1] * 100
            }, index=[next_date])
            
            future_data = pd.concat([future_data, new_row])
            
            # به‌روزرسانی شاخص‌های تکنیکال
            future_data['SMA_50'] = future_data['Close'].rolling(window=50).mean()
            future_data['SMA_200'] = future_data['Close'].rolling(window=200).mean()
            future_data['RSI'] = self._calculate_rsi(future_data)
            future_data['Volatility'] = future_data['Close'].rolling(window=14).std()
            
            # آماده‌سازی توالی جدید برای پیش‌بینی بعدی
            new_sequence = []
            for j in range(lag_days):
                lag_features = []
                
                # شاخص ردیف داده
                idx = len(future_data) - lag_days + j
                
                # اضافه کردن ویژگی‌های تاخیری
                for k in range(1, lag_days + 1):
                    if idx - k >= 0:
                        lag_features.append(future_data['Close'].iloc[idx - k])
                    else:
                        lag_features.append(future_data['Close'].iloc[0])  # استفاده از اولین داده در صورت عدم وجود داده قدیمی‌تر
                
                # اضافه کردن سایر ویژگی‌ها
                if idx < len(future_data):
                    lag_features.extend([
                        future_data['Volume'].iloc[idx],
                        future_data['SMA_50'].iloc[idx] if not pd.isna(future_data['SMA_50'].iloc[idx]) else 0,
                        future_data['SMA_200'].iloc[idx] if not pd.isna(future_data['SMA_200'].iloc[idx]) else 0,
                        future_data['Daily_Change'].iloc[idx],
                        future_data['RSI'].iloc[idx] if not pd.isna(future_data['RSI'].iloc[idx]) else 50,
                        future_data['Volatility'].iloc[idx] if not pd.isna(future_data['Volatility'].iloc[idx]) else 0
                    ])
                else:
                    # استفاده از داده‌های آخرین ردیف در صورت عدم وجود داده
                    lag_features.extend([
                        future_data['Volume'].iloc[-1],
                        future_data['SMA_50'].iloc[-1] if not pd.isna(future_data['SMA_50'].iloc[-1]) else 0,
                        future_data['SMA_200'].iloc[-1] if not pd.isna(future_data['SMA_200'].iloc[-1]) else 0,
                        future_data['Daily_Change'].iloc[-1],
                        future_data['RSI'].iloc[-1] if not pd.isna(future_data['RSI'].iloc[-1]) else 50,
                        future_data['Volatility'].iloc[-1] if not pd.isna(future_data['Volatility'].iloc[-1]) else 0
                    ])
                
                new_sequence.append(lag_features)
            
            # نرمال‌سازی توالی جدید
            new_sequence = np.array(new_sequence)
            new_sequence = self.scaler_X.transform(new_sequence)
            current_sequence = new_sequence
        
        # تبدیل پیش‌بینی‌ها به مقیاس اصلی
        future_predictions = np.array(future_predictions)
        future_predictions = self.scaler_y.inverse_transform(future_predictions)
        
        return future_predictions, future_dates
    
    def _calculate_rsi(self, data, periods=14):
        """
        محاسبه شاخص قدرت نسبی (RSI)
        
        Args:
            data: DataFrame شامل قیمت‌های Close
            periods: دوره محاسبه RSI (پیش‌فرض: ۱۴)
        
        Returns:
            Series: مقادیر RSI محاسبه شده
        """
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi 