import os
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bitcoin_price_prediction",
    version="0.1.0",
    author="استاد و دانشجو",
    author_email="example@example.com",
    description="سیستم پیش‌بینی قیمت بیت‌کوین با استفاده از یادگیری عمیق",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/bitcoin_price_prediction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.5.0",
        "keras>=2.5.0",
        "pandas>=1.3.0",
        "numpy>=1.19.5",
        "matplotlib>=3.4.2",
        "scikit-learn>=0.24.2",
        "yfinance>=0.1.70",
        "seaborn>=0.11.1",
    ],
) 