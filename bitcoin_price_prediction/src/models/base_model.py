"""
کلاس پایه برای تمام مدل‌های پیش‌بینی سری زمانی
"""

from abc import ABC, abstractmethod
import os
import joblib
import datetime
import random
import string
import tensorflow as tf
import shutil
import json
import numpy as np

class TimeSeriesModel(ABC):
    """
    کلاس پایه برای تمام مدل‌های پیش‌بینی سری زمانی
    """
    
    def __init__(self, config):
        """
        مقداردهی اولیه کلاس پایه
        
        Args:
            config: تنظیمات مدل
        """
        self.config = config
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.history = None
        self.run_name = self._create_random_run_name()
    
    @abstractmethod
    def build_model(self, input_shape):
        """
        ساخت معماری مدل
        
        Args:
            input_shape: شکل ورودی مدل
        
        Returns:
            model: مدل ساخته شده
        """
        pass
    
    def compile_model(self):
        """
        کامپایل مدل با تنظیمات مشخص شده
        """
        if self.model is None:
            raise ValueError("مدل باید قبل از کامپایل ساخته شود.")
        
        self.model.compile(
            optimizer=self.config.OPTIMIZER,
            loss=self.config.LOSS
        )
        return self.model
    
    def get_callbacks(self):
        """
        ایجاد callbacks برای آموزش مدل
        
        Returns:
            list: لیست callbacks
        """
        callbacks = []
        
        # Early Stopping
        if hasattr(self.config, 'EARLY_STOPPING_PATIENCE'):
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=self.config.RESTORE_BEST_WEIGHTS
            )
            callbacks.append(early_stopping)
        
        # ReduceLROnPlateau
        if hasattr(self.config, 'REDUCE_LR_FACTOR') and hasattr(self.config, 'REDUCE_LR_PATIENCE'):
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.REDUCE_LR_FACTOR,
                patience=self.config.REDUCE_LR_PATIENCE
            )
            callbacks.append(reduce_lr)
        
        return callbacks
    
    def fit(self, X_train, y_train, validation_data=None, sample_weight=None):
        """
        آموزش مدل
        
        Args:
            X_train: داده‌های ورودی آموزش
            y_train: داده‌های هدف آموزش
            validation_data: داده‌های اعتبارسنجی
            sample_weight: وزن نمونه‌ها
        
        Returns:
            history: تاریخچه آموزش
        """
        if self.model is None:
            raise ValueError("مدل باید قبل از آموزش ساخته و کامپایل شود.")
        
        callbacks = self.get_callbacks()
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            validation_data=validation_data,
            callbacks=callbacks,
            sample_weight=sample_weight,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """
        پیش‌بینی با استفاده از مدل
        
        Args:
            X: داده‌های ورودی
        
        Returns:
            y_pred: پیش‌بینی‌ها
        """
        if self.model is None:
            raise ValueError("مدل باید قبل از پیش‌بینی ساخته شود.")
        
        return self.model.predict(X)
    
    def save(self, base_dir='models/saved_models'):
        """
        ذخیره مدل و اسکیلرها
        
        Args:
            base_dir: مسیر پایه برای ذخیره مدل
        
        Returns:
            str: مسیر ذخیره مدل
        """
        # ایجاد دایرکتوری اصلی اگر وجود نداشته باشد
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # ایجاد دایرکتوری برای اجرای فعلی
        run_dir = os.path.join(base_dir, self.run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        
        # ذخیره مدل
        model_path = os.path.join(run_dir, 'model.h5')
        self.model.save(model_path)
        
        # ذخیره اسکیلرها
        if self.scaler_X is not None and self.scaler_y is not None:
            scaler_X_path = os.path.join(run_dir, 'scaler_X.pkl')
            scaler_y_path = os.path.join(run_dir, 'scaler_y.pkl')
            joblib.dump(self.scaler_X, scaler_X_path)
            joblib.dump(self.scaler_y, scaler_y_path)
        
        # ذخیره تنظیمات مدل
        config_path = os.path.join(run_dir, 'config.json')
        config_dict = {k: v for k, v in vars(self.config).items() if not k.startswith('__')}
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        print(f"مدل با موفقیت در مسیر {run_dir} ذخیره شد.")
        return run_dir
    
    @classmethod
    def load(cls, model_path, config=None):
        """
        بارگذاری مدل و اسکیلرها
        
        Args:
            model_path: مسیر مدل ذخیره شده
            config: تنظیمات مدل (اختیاری)
        
        Returns:
            model: نمونه کلاس با مدل بارگذاری شده
        """
        # تعیین مسیرهای فایل
        if os.path.isdir(model_path):
            model_dir = model_path
            model_file = os.path.join(model_dir, 'model.h5')
            scaler_X_file = os.path.join(model_dir, 'scaler_X.pkl')
            scaler_y_file = os.path.join(model_dir, 'scaler_y.pkl')
            config_file = os.path.join(model_dir, 'config.json')
        else:
            model_dir = os.path.dirname(model_path)
            model_file = model_path
            scaler_X_file = os.path.join(model_dir, 'scaler_X.pkl')
            scaler_y_file = os.path.join(model_dir, 'scaler_y.pkl')
            config_file = os.path.join(model_dir, 'config.json')
        
        # بارگذاری تنظیمات اگر ارائه نشده باشد
        if config is None and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
                
            # تبدیل دیکشنری به آبجکت
            class Config:
                pass
            
            config = Config()
            for k, v in config_dict.items():
                setattr(config, k, v)
        
        # ایجاد نمونه کلاس
        instance = cls(config)
        
        # بارگذاری مدل
        instance.model = tf.keras.models.load_model(model_file)
        
        # بارگذاری اسکیلرها
        if os.path.exists(scaler_X_file) and os.path.exists(scaler_y_file):
            instance.scaler_X = joblib.load(scaler_X_file)
            instance.scaler_y = joblib.load(scaler_y_file)
        
        return instance
    
    def set_scalers(self, scaler_X, scaler_y):
        """
        تنظیم اسکیلرها
        
        Args:
            scaler_X: اسکیلر ویژگی‌های ورودی
            scaler_y: اسکیلر متغیر هدف
        """
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
    
    def _create_random_run_name(self):
        """
        ایجاد یک نام تصادفی برای اجرای مدل
        
        Returns:
            str: نام تصادفی
        """
        # ایجاد یک رشته تصادفی 8 کاراکتری
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        # اضافه کردن تاریخ و زمان
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = getattr(self.config, 'MODEL_TYPE', 'model')
        return f"{model_type}_{timestamp}_{random_str}" 