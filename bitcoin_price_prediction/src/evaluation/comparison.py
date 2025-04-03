"""
مقایسه عملکرد مدل‌های مختلف پیش‌بینی
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bitcoin_price_prediction.src.evaluation.metrics import calculate_all_metrics, print_metrics

class ModelComparison:
    """
    کلاس مقایسه عملکرد مدل‌های مختلف برای پیش‌بینی قیمت بیت‌کوین
    """
    
    def __init__(self, model_names=None):
        """
        مقداردهی اولیه کلاس
        
        Args:
            model_names: لیست نام‌های مدل‌ها (اختیاری)
        """
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.model_names = model_names if model_names else []
    
    def add_model(self, model_name, model, predictions=None, true_values=None):
        """
        اضافه کردن مدل به لیست مقایسه
        
        Args:
            model_name: نام مدل
            model: شیء مدل
            predictions: پیش‌بینی‌های مدل (اختیاری)
            true_values: مقادیر واقعی (اختیاری)
        """
        self.models[model_name] = model
        
        if model_name not in self.model_names:
            self.model_names.append(model_name)
        
        if predictions is not None and true_values is not None:
            self.add_predictions(model_name, predictions, true_values)
    
    def add_predictions(self, model_name, predictions, true_values):
        """
        اضافه کردن پیش‌بینی‌های مدل
        
        Args:
            model_name: نام مدل
            predictions: پیش‌بینی‌های مدل
            true_values: مقادیر واقعی
        """
        if model_name not in self.models:
            raise ValueError(f"مدل {model_name} در لیست مدل‌ها وجود ندارد.")
        
        self.predictions[model_name] = {
            'predictions': predictions,
            'true_values': true_values
        }
        
        # محاسبه معیارهای ارزیابی
        self.metrics[model_name] = calculate_all_metrics(true_values, predictions)
    
    def get_metrics(self, model_name=None):
        """
        دریافت معیارهای ارزیابی
        
        Args:
            model_name: نام مدل (اختیاری)
        
        Returns:
            dict: معیارهای ارزیابی
        """
        if model_name:
            if model_name not in self.metrics:
                raise ValueError(f"معیارهای ارزیابی برای مدل {model_name} محاسبه نشده است.")
            return self.metrics[model_name]
        
        return self.metrics
    
    def compare_metrics(self, metrics=None):
        """
        مقایسه معیارهای ارزیابی مدل‌ها
        
        Args:
            metrics: لیست معیارهای مورد نظر برای مقایسه
        
        Returns:
            DataFrame: جدول مقایسه معیارها
        """
        if not self.metrics:
            raise ValueError("هیچ معیار ارزیابی‌ای محاسبه نشده است. ابتدا از متد add_predictions استفاده کنید.")
        
        if metrics is None:
            # استفاده از معیارهای پیش‌فرض
            metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'Direction_Accuracy']
        
        # ایجاد دیکشنری برای تبدیل به DataFrame
        comparison_dict = {}
        for model_name in self.metrics:
            comparison_dict[model_name] = {metric: self.metrics[model_name][metric] for metric in metrics}
        
        # تبدیل به DataFrame
        comparison_df = pd.DataFrame(comparison_dict)
        
        return comparison_df
    
    def print_metrics(self, model_name=None):
        """
        چاپ معیارهای ارزیابی
        
        Args:
            model_name: نام مدل (اختیاری)
        """
        if model_name:
            if model_name not in self.metrics:
                raise ValueError(f"معیارهای ارزیابی برای مدل {model_name} محاسبه نشده است.")
            print_metrics(self.metrics[model_name], title=f"معیارهای ارزیابی {model_name}")
        else:
            for model_name in self.metrics:
                print_metrics(self.metrics[model_name], title=f"معیارهای ارزیابی {model_name}")
    
    def plot_comparison(self, metric, title=None):
        """
        رسم نمودار مقایسه مدل‌ها براساس یک معیار
        
        Args:
            metric: نام معیار
            title: عنوان نمودار (اختیاری)
        """
        if not self.metrics:
            raise ValueError("هیچ معیار ارزیابی‌ای محاسبه نشده است. ابتدا از متد add_predictions استفاده کنید.")
        
        values = [self.metrics[model_name][metric] for model_name in self.model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.model_names, values, width=0.5)
        
        # اضافه کردن مقادیر بالای نمودار
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom')
        
        if title:
            plt.title(title)
        else:
            plt.title(f'مقایسه {metric} بین مدل‌های مختلف')
        
        plt.xlabel('مدل')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_all_metrics(self, metrics=None, save_path=None):
        """
        رسم نمودار مقایسه مدل‌ها براساس چندین معیار
        
        Args:
            metrics: لیست معیارهای مورد نظر برای مقایسه
            save_path: مسیر ذخیره نمودار (اختیاری)
        """
        if not self.metrics:
            raise ValueError("هیچ معیار ارزیابی‌ای محاسبه نشده است. ابتدا از متد add_predictions استفاده کنید.")
        
        if metrics is None:
            # استفاده از معیارهای پیش‌فرض
            metrics = ['RMSE', 'MAE', 'R2', 'Direction_Accuracy', 'Win_Rate', 'Profit_Factor']
        
        # تعداد ردیف و ستون برای نمودارها
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                ax = axes[i]
                values = [self.metrics[model_name][metric] for model_name in self.model_names]
                
                bars = ax.bar(self.model_names, values, width=0.5)
                
                # اضافه کردن مقادیر بالای نمودار
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.4f}', ha='center', va='bottom')
                
                ax.set_title(f'مقایسه {metric}')
                ax.set_xlabel('مدل')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
        
        # حذف محورهای اضافی
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"نمودار در مسیر {save_path} ذخیره شد.")
        
        return fig
    
    def plot_predictions(self, dates, actual_values, model_names=None, future_dates=None, future_predictions=None, title=None, save_path=None):
        """
        رسم نمودار مقایسه پیش‌بینی‌های مدل‌های مختلف
        
        Args:
            dates: تاریخ‌های پیش‌بینی
            actual_values: مقادیر واقعی
            model_names: لیست نام‌های مدل‌ها (اختیاری)
            future_dates: تاریخ‌های آینده (اختیاری)
            future_predictions: دیکشنری پیش‌بینی‌های آینده برای هر مدل (اختیاری)
            title: عنوان نمودار (اختیاری)
            save_path: مسیر ذخیره نمودار (اختیاری)
        """
        if not self.predictions:
            raise ValueError("هیچ پیش‌بینی‌ای ثبت نشده است. ابتدا از متد add_predictions استفاده کنید.")
        
        if model_names is None:
            model_names = list(self.predictions.keys())
        
        plt.figure(figsize=(15, 8))
        
        # رسم مقادیر واقعی
        plt.plot(dates, actual_values, label='مقادیر واقعی', color='blue', linewidth=2)
        
        # رسم پیش‌بینی‌های مدل‌های مختلف
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, model_name in enumerate(model_names):
            if model_name in self.predictions:
                color_idx = i % len(colors)
                plt.plot(dates, self.predictions[model_name]['predictions'], 
                        label=f'پیش‌بینی {model_name}', 
                        color=colors[color_idx], 
                        linestyle='--')
        
        # رسم پیش‌بینی‌های آینده
        if future_dates is not None and future_predictions is not None:
            for i, model_name in enumerate(model_names):
                if model_name in future_predictions:
                    color_idx = i % len(colors)
                    plt.plot(future_dates, future_predictions[model_name], 
                            label=f'پیش‌بینی آینده {model_name}', 
                            color=colors[color_idx], 
                            linestyle=':')
        
        if title:
            plt.title(title)
        else:
            plt.title('مقایسه پیش‌بینی‌های مدل‌های مختلف')
        
        plt.xlabel('تاریخ')
        plt.ylabel('قیمت')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"نمودار در مسیر {save_path} ذخیره شد.")
        
        return plt.gcf()
    
    def find_best_model(self, metric=None, higher_is_better=True):
        """
        یافتن بهترین مدل براساس یک معیار
        
        Args:
            metric: نام معیار (اختیاری)
            higher_is_better: آیا مقدار بیشتر بهتر است؟ (اختیاری)
        
        Returns:
            tuple: (نام بهترین مدل، مقدار معیار)
        """
        if not self.metrics:
            raise ValueError("هیچ معیار ارزیابی‌ای محاسبه نشده است. ابتدا از متد add_predictions استفاده کنید.")
        
        if metric is None:
            # استفاده از معیار پیش‌فرض
            metric = 'R2'
            higher_is_better = True
        
        # معیارهایی که کمتر بودن آنها بهتر است
        if metric in ['MSE', 'RMSE', 'MAE', 'MAPE', 'Max_Error', 'Median_Error']:
            higher_is_better = False
        
        values = {}
        for model_name in self.metrics:
            if metric in self.metrics[model_name]:
                values[model_name] = self.metrics[model_name][metric]
        
        if higher_is_better:
            best_model = max(values.items(), key=lambda x: x[1])
        else:
            best_model = min(values.items(), key=lambda x: x[1])
        
        return best_model 