"""
تنظیمات پیش‌فرض برای مدل LSTM
"""

def get_lstm_config():
    """
    تنظیمات پیش‌فرض برای مدل LSTM
    
    Returns:
        dict: تنظیمات پیش‌فرض مدل LSTM
    """
    config = {
        # پارامترهای داده
        'data_path': 'data/bitcoin_data.csv',
        'train_split_ratio': 0.8,
        
        # پارامترهای مدل
        'model_type': 'lstm',
        'lag_days': 30,
        'lstm1_units': 50,
        'lstm2_units': 30,
        'dropout1': 0.3,
        'dropout2': 0.3,
        
        # پارامترهای آموزش
        'exponential_weighting': False,
        'weight_decay': 0.5,
        'epochs': 100,
        'batch_size': 32,
        'patience': 20,
        'validation_split': 0.2,
        
        # پارامترهای پیش‌بینی
        'future_days': 30,
        
        # پارامترهای خروجی
        'output_dir': 'output',
        'save_model': True,
        'model_save_dir': 'output/models',
        'figures_dir': 'output/figures',
        'predictions_dir': 'output/predictions',
    }
    
    return config

def get_lstm_model_args(args=None):
    """
    تبدیل آرگومان‌های خط فرمان به پارامترهای مدل LSTM
    
    Args:
        args (argparse.Namespace, optional): آرگومان‌های خط فرمان
        
    Returns:
        dict: پارامترهای مدل LSTM
    """
    # دریافت تنظیمات پیش‌فرض
    config = get_lstm_config()
    
    # جایگزینی پارامترهای تعیین شده توسط کاربر
    if args:
        for key, value in vars(args).items():
            if key in config and value is not None:
                config[key] = value
    
    return config 