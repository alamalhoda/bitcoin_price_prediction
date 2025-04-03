"""
تنظیمات پایه برای پیش‌بینی قیمت بیت‌کوین

این فایل حاوی تنظیمات پایه برای همه مدل‌های پیش‌بینی است.
"""

# مسیر داده‌های ورودی
DATA_PATH = 'data/bitcoin_data.csv'

# پارامترهای تقسیم داده‌ها
TRAIN_SPLIT_RATIO = 0.80
VALIDATION_SPLIT = 0.15

# پارامترهای پیش‌پردازش
LAG_DAYS = 30  # تعداد روزهای گذشته برای پیش‌بینی

# پارامترهای آموزش
EPOCHS = 50
BATCH_SIZE = 128

# پارامترهای Early Stopping
EARLY_STOPPING_PATIENCE = 10
RESTORE_BEST_WEIGHTS = True

# پارامترهای ReduceLROnPlateau
REDUCE_LR_FACTOR = 0.2
REDUCE_LR_PATIENCE = 5

# پارامترهای مدل
OPTIMIZER = 'adam'
LOSS = 'mse'

# مسیر ذخیره مدل‌ها
MODELS_DIR = 'models/saved_models'

# پارامترهای پیش‌بینی
FUTURE_DAYS = 30  # تعداد روزهای آینده برای پیش‌بینی 