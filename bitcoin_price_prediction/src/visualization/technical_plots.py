"""
نمودارهای مربوط به شاخص‌های تکنیکال
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_price_with_sma(data, title=None, save_path=None):
    """
    رسم نمودار قیمت به همراه میانگین‌های متحرک
    
    Args:
        data: DataFrame شامل ستون‌های Close، SMA_50 و SMA_200
        title: عنوان نمودار (اختیاری)
        save_path: مسیر ذخیره نمودار (اختیاری)
    
    Returns:
        fig: شیء نمودار
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # رسم قیمت و میانگین‌های متحرک
    ax.plot(data.index, data['Close'], label='قیمت', color='blue')
    ax.plot(data.index, data['SMA_50'], label='میانگین متحرک ۵۰ روزه', color='red')
    ax.plot(data.index, data['SMA_200'], label='میانگین متحرک ۲۰۰ روزه', color='green')
    
    # تنظیم عنوان و برچسب‌ها
    if title:
        ax.set_title(title)
    else:
        ax.set_title('قیمت بیت‌کوین و میانگین‌های متحرک')
    
    ax.set_xlabel('تاریخ')
    ax.set_ylabel('قیمت (دلار)')
    ax.grid(True)
    ax.legend()
    
    # تنظیم فرمت تاریخ در محور x
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"نمودار در مسیر {save_path} ذخیره شد.")
    
    return fig

def plot_rsi(data, title=None, save_path=None):
    """
    رسم نمودار شاخص قدرت نسبی (RSI)
    
    Args:
        data: DataFrame شامل ستون RSI
        title: عنوان نمودار (اختیاری)
        save_path: مسیر ذخیره نمودار (اختیاری)
    
    Returns:
        fig: شیء نمودار
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # رسم RSI
    ax.plot(data.index, data['RSI'], label='RSI', color='purple')
    
    # اضافه کردن خطوط مرزی
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='مرز اشباع خرید (۷۰)')
    ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='مرز اشباع فروش (۳۰)')
    ax.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
    
    # تنظیم محدوده محور y
    ax.set_ylim(0, 100)
    
    # تنظیم عنوان و برچسب‌ها
    if title:
        ax.set_title(title)
    else:
        ax.set_title('شاخص قدرت نسبی (RSI)')
    
    ax.set_xlabel('تاریخ')
    ax.set_ylabel('RSI')
    ax.grid(True)
    ax.legend()
    
    # تنظیم فرمت تاریخ در محور x
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"نمودار در مسیر {save_path} ذخیره شد.")
    
    return fig

def plot_volume(data, title=None, save_path=None):
    """
    رسم نمودار حجم معاملات
    
    Args:
        data: DataFrame شامل ستون Volume
        title: عنوان نمودار (اختیاری)
        save_path: مسیر ذخیره نمودار (اختیاری)
    
    Returns:
        fig: شیء نمودار
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # رسم حجم معاملات
    ax.bar(data.index, data['Volume'], alpha=0.7, width=2, color='blue', label='حجم معاملات')
    
    # محاسبه و رسم میانگین متحرک حجم معاملات
    volume_ma = data['Volume'].rolling(window=20).mean()
    ax.plot(data.index, volume_ma, color='red', label='میانگین متحرک ۲۰ روزه حجم', linewidth=2)
    
    # تنظیم عنوان و برچسب‌ها
    if title:
        ax.set_title(title)
    else:
        ax.set_title('حجم معاملات بیت‌کوین')
    
    ax.set_xlabel('تاریخ')
    ax.set_ylabel('حجم معاملات')
    ax.grid(True)
    ax.legend()
    
    # تنظیم فرمت تاریخ در محور x
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"نمودار در مسیر {save_path} ذخیره شد.")
    
    return fig

def plot_volatility(data, title=None, save_path=None):
    """
    رسم نمودار نوسانات قیمت
    
    Args:
        data: DataFrame شامل ستون Volatility
        title: عنوان نمودار (اختیاری)
        save_path: مسیر ذخیره نمودار (اختیاری)
    
    Returns:
        fig: شیء نمودار
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # رسم نوسانات قیمت
    ax.plot(data.index, data['Volatility'], label='نوسانات قیمت', color='orange')
    
    # محاسبه و رسم میانگین متحرک نوسانات قیمت
    volatility_ma = data['Volatility'].rolling(window=20).mean()
    ax.plot(data.index, volatility_ma, color='red', label='میانگین متحرک ۲۰ روزه نوسانات', linewidth=2)
    
    # تنظیم عنوان و برچسب‌ها
    if title:
        ax.set_title(title)
    else:
        ax.set_title('نوسانات قیمت بیت‌کوین')
    
    ax.set_xlabel('تاریخ')
    ax.set_ylabel('نوسانات (انحراف معیار)')
    ax.grid(True)
    ax.legend()
    
    # تنظیم فرمت تاریخ در محور x
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"نمودار در مسیر {save_path} ذخیره شد.")
    
    return fig

def plot_price_with_volume(data, title=None, save_path=None):
    """
    رسم نمودار ترکیبی قیمت و حجم معاملات
    
    Args:
        data: DataFrame شامل ستون‌های Close و Volume
        title: عنوان نمودار (اختیاری)
        save_path: مسیر ذخیره نمودار (اختیاری)
    
    Returns:
        fig: شیء نمودار
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # رسم قیمت در محور بالایی
    ax1.plot(data.index, data['Close'], label='قیمت', color='blue')
    
    # تنظیم محور بالایی
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title('قیمت و حجم معاملات بیت‌کوین')
    
    ax1.set_ylabel('قیمت (دلار)')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # رسم حجم معاملات در محور پایینی
    ax2.bar(data.index, data['Volume'], alpha=0.7, width=2, color='green', label='حجم معاملات')
    
    # محاسبه و رسم میانگین متحرک حجم معاملات
    volume_ma = data['Volume'].rolling(window=20).mean()
    ax2.plot(data.index, volume_ma, color='red', label='میانگین متحرک ۲۰ روزه حجم', linewidth=2)
    
    # تنظیم محور پایینی
    ax2.set_xlabel('تاریخ')
    ax2.set_ylabel('حجم معاملات')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # تنظیم فرمت تاریخ در محور x
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"نمودار در مسیر {save_path} ذخیره شد.")
    
    return fig

def plot_complete_technical_analysis(data, title=None, save_path=None):
    """
    رسم نمودار کامل تحلیل تکنیکال
    
    Args:
        data: DataFrame شامل تمام شاخص‌های تکنیکال
        title: عنوان نمودار (اختیاری)
        save_path: مسیر ذخیره نمودار (اختیاری)
    
    Returns:
        fig: شیء نمودار
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # رسم قیمت و میانگین‌های متحرک در محور بالایی
    ax1.plot(data.index, data['Close'], label='قیمت', color='blue')
    ax1.plot(data.index, data['SMA_50'], label='میانگین متحرک ۵۰ روزه', color='red')
    ax1.plot(data.index, data['SMA_200'], label='میانگین متحرک ۲۰۰ روزه', color='green')
    
    # تنظیم محور بالایی
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title('تحلیل تکنیکال جامع بیت‌کوین')
    
    ax1.set_ylabel('قیمت (دلار)')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # رسم حجم معاملات در محور میانی
    ax2.bar(data.index, data['Volume'], alpha=0.7, width=2, color='green', label='حجم معاملات')
    
    # محاسبه و رسم میانگین متحرک حجم معاملات
    volume_ma = data['Volume'].rolling(window=20).mean()
    ax2.plot(data.index, volume_ma, color='red', label='میانگین متحرک ۲۰ روزه حجم', linewidth=2)
    
    # تنظیم محور میانی
    ax2.set_ylabel('حجم معاملات')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # رسم RSI در محور پایینی
    ax3.plot(data.index, data['RSI'], label='RSI', color='purple')
    
    # اضافه کردن خطوط مرزی به نمودار RSI
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='مرز اشباع خرید (۷۰)')
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='مرز اشباع فروش (۳۰)')
    ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
    
    # تنظیم محور پایینی
    ax3.set_ylim(0, 100)
    ax3.set_xlabel('تاریخ')
    ax3.set_ylabel('RSI')
    ax3.grid(True)
    ax3.legend(loc='upper left')
    
    # تنظیم فرمت تاریخ در محور x
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"نمودار در مسیر {save_path} ذخیره شد.")
    
    return fig

def create_all_technical_plots(data, output_dir=None):
    """
    ایجاد و ذخیره تمام نمودارهای تحلیل تکنیکال در یک دایرکتوری
    
    Args:
        data: DataFrame شامل تمام شاخص‌های تکنیکال
        output_dir: مسیر دایرکتوری خروجی (اختیاری)
    """
    if output_dir is None:
        output_dir = 'technical_plots'
    
    # ایجاد دایرکتوری خروجی اگر وجود نداشته باشد
    os.makedirs(output_dir, exist_ok=True)
    
    # نمودار قیمت و میانگین‌های متحرک
    plot_price_with_sma(
        data,
        save_path=os.path.join(output_dir, '01_price_with_sma.png')
    )
    
    # نمودار RSI
    plot_rsi(
        data,
        save_path=os.path.join(output_dir, '02_rsi.png')
    )
    
    # نمودار حجم معاملات
    plot_volume(
        data,
        save_path=os.path.join(output_dir, '03_volume.png')
    )
    
    # نمودار نوسانات قیمت
    plot_volatility(
        data,
        save_path=os.path.join(output_dir, '04_volatility.png')
    )
    
    # نمودار ترکیبی قیمت و حجم معاملات
    plot_price_with_volume(
        data,
        save_path=os.path.join(output_dir, '05_price_with_volume.png')
    )
    
    # نمودار کامل تحلیل تکنیکال
    plot_complete_technical_analysis(
        data,
        save_path=os.path.join(output_dir, '06_complete_technical_analysis.png')
    )
    
    print(f"تمام نمودارهای تحلیل تکنیکال در مسیر {output_dir} ذخیره شدند.") 