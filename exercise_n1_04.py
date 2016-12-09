# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:59:06 2016

@author: Sergey Kazanin
"""

import pandas as pd
import urllib.request
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
import datetime, sys

import numpy as np
from scipy.optimize import curve_fit

def linear_func(x, a, b):
    return a+b*x
def exponenial_func(x, a, b, c):
    return a*np.exp(-c*(x-b))
    #return a*np.e**(b*x)
def log_func(x, a, b):
    return a+b*np.log(x)
# Read data
data = pd.read_excel("course_1.xlsx", header=0, sheetname=1)
data.columns =['ID', 'week', 'sales', 'average price',
       'seasonality', 'trend impact', 'price impact',
       'trend forecast', 'price forecast']

def load_currency_exchange_rates(date_indx):
    exch_rates = pd.Series(index = date_indx)
    last_rate = 7.9898
    for rdate in date_indx:
        response = urllib.request.urlopen('https://api.privatbank.ua/p24api/exchange_rates?date=' + rdate.strftime('%d.%m.%Y'))
        data_xml = response.read().decode("utf-8")
        root = ET.fromstring(str(data_xml))
        for el in range(len(root)):
            if root[el].attrib['currency'] == 'USD':
                last_rate = float(list(root[el].attrib.values())[5])
        exch_rates.ix[rdate] = last_rate
    return exch_rates

def make_forecast(f_data, week_data):
    res = sm.tsa.seasonal_decompose(f_data)
    trend_ = pd.Series(res.trend[res.trend.notnull()], index=week_data.index)
    seasonal_ = pd.Series(res.seasonal.notnull(), index=week_data.index)
    for i in range(54):
        seasonal_[week_data==i] = res.seasonal[i]

    reg = LinearRegression()
    reg.fit(f_data.index.values.reshape(-1,1), f_data)
    trend_forecast_ = reg.predict(week_data.index.values.astype(float).reshape(-1,1))
    price_forecast_ = trend_forecast_ + seasonal_

    return trend_, seasonal_, trend_forecast_, price_forecast_

def show_graphs(title_='', data_=None, trend_=None, seasonal_=None, trend_forecast_=None, price_forecast_=None, sizeX=12, sizeY=8):
    if data_ is not None:
        data_.plot(figsize=(sizeX, sizeY), title=title_, legend=True)
    if trend_ is not None:
        trend_.plot(figsize=(sizeX, sizeY), title=title_, legend=True)
    if seasonal_ is not None:
        seasonal_.plot(figsize=(sizeX, sizeY), title=title_, legend=True)
    if trend_forecast_ is not None:
        trend_forecast_.plot(figsize=(sizeX, sizeY), title=title_, legend=True)
    if price_forecast_ is not None:
        price_forecast_.plot(figsize=(sizeX, sizeY), title=title_, legend=True)
    plt.show()

# Feature design
data['wdate'] = data['week'].astype(str).str[:4]+'-W'+data['week'].astype(str).str[4:]+'-1'
data['year'] = data['week'].astype(str).str[:4].astype(int)
data['week'] = data['week'].astype(str).str[4:].astype(int)
data.index = pd.to_datetime(data['wdate'], format='%Y-W%W-%w')

# Get USD exchange rates & set USD prices
data['usd rate'] = load_currency_exchange_rates(data.index)
data['usd rate'].plot(title='USD exchange rate', figsize=(12,8))
plt.show()

data['average price usd'] = data['average price']/data['usd rate']

# 1. Посчитать линейный тренд по цене (инфляцию) и продлить значения цены до конца 2016-го года
# UAH
price_data = data['average price'].dropna()
price_data.index = pd.date_range(price_data.index.min(), price_data.index.max(), freq='W-MON')[:200]

data['price trend'], data['price seasonal'], data['price trend forecast'], data['price forecast'] = make_forecast(price_data, data['week'])

show_graphs('Average price (UAH): actual data, trend & seasonality, trend prediction',
            data['average price'], data['price trend'], data['price seasonal'], data['price trend forecast'], data['price forecast'])

# USD
price_usd_data = data['average price usd'].dropna()
price_usd_data.index = pd.date_range(price_usd_data.index.min(), price_usd_data.index.max(), freq='W-MON')[:200]

data['price trend usd'], data['price seasonal usd'], data['price trend forecast usd'], data['price forecast usd'] = make_forecast(price_usd_data, data['week'])

show_graphs('Average price (USD): actual data, trend & seasonality, trend prediction',
            data['average price usd'], data['price trend usd'], data['price seasonal usd'], data['price trend forecast usd'], data['price forecast usd'])

# 2. Посчитать сезонность и линейный тренд так, как было в предыдущей задаче
sales_data = data['sales'].dropna()
sales_data.index = pd.date_range(sales_data.index.min(), sales_data.index.max(), freq='W-MON')[:200]

data['sales trend'], data['sales seasonal'], data['sales trend forecast'], data['sales forecast'] = make_forecast(sales_data, data['week'])

show_graphs('Average price (USD): actual data, trend & seasonality, trend prediction',
            data['sales'], data['sales trend'], data['sales seasonal'], data['sales trend forecast'], data['sales forecast'], 16, 8)

sales_volume = data['sales'].dropna()
sales_volume.index = data['average price'].dropna()
sales_volume.sort_index(inplace=True)
sales_volume.drop_duplicates(inplace=True)

# 4. Расчитать линейную регрессию по цене (Y = A + B*X) с помощью метода наименьших квадратов, где X - цена, Y - продажи в штуках
popt, pcov = curve_fit(linear_func, sales_volume.index.values, sales_volume.values)
print('Linear model parameters', popt)
sales_volume_lin = pd.Series(linear_func(sales_volume.index.values, popt[0], popt[1]), index=sales_volume.index)
sales_volume_lin.plot(figsize=(10,8))
sales_volume.plot(figsize=(10,8))
plt.show()

# 5. Расчитать экспоненциальную регрессию по цене (Y = A * exp(B*X) или Y = A * e^BX), с помощью метода наименьших квадратов, где X - цена, Y - продажи в штуках
popt, pcov = curve_fit(exponenial_func, np.log(sales_volume.index.values), sales_volume.values)
print('Exponenial model parameters', popt)
sales_volume_exp = pd.Series(exponenial_func(np.log(sales_volume.index.values), popt[0], popt[1], popt[2]), index=sales_volume.index)
sales_volume_exp.plot(figsize=(10,8))
sales_volume.plot()
plt.show()

# 6. Расчитать логарифмическую регрессию по цене (Y = A + B * ln(X)), с помощью метода наименьших квадратов, где X - цена, Y - продажи в штуках
popt, pcov = curve_fit(log_func, np.exp(sales_volume.index.values), sales_volume.values)
print('Logarithmic model parameters', popt)
sales_volume_log = pd.Series(log_func(np.exp(sales_volume.index.values), popt[0], popt[1]), index=sales_volume.index)
sales_volume_log.plot(figsize=(10,8))
sales_volume.plot()
plt.show()

# Все результаты на одном графике
sales_volume_lin.plot(figsize=(10,8))
sales_volume_exp.plot(figsize=(10,8))
sales_volume_log.plot(figsize=(10,8))
plt.show()

# 7. По всем ценовым регрессиям проверить направленность, она должна быть не прямой, а обратной, т.е. при увеличении значения X значение Y должно уменьшаться
# Таки ДА: зависимость объема продаж от цены обратная!

# Повторяем все для долларовых цен
sales_volume = data['sales'].dropna()
sales_volume.index = data['average price usd'].dropna()
sales_volume.sort_index(inplace=True)
sales_volume.drop_duplicates(inplace=True)

# 4. Расчитать линейную регрессию по цене (Y = A + B*X) с помощью метода наименьших квадратов, где X - цена, Y - продажи в штуках
popt, pcov = curve_fit(linear_func, sales_volume.index.values, sales_volume.values)
print('Linear model parameters', popt)
sales_volume_lin = pd.Series(linear_func(sales_volume.index.values, popt[0], popt[1]), index=sales_volume.index)
sales_volume_lin.plot(figsize=(10,8))
sales_volume.plot(figsize=(10,8))
plt.show()

# 5. Расчитать экспоненциальную регрессию по цене (Y = A * exp(B*X) или Y = A * e^BX), с помощью метода наименьших квадратов, где X - цена, Y - продажи в штуках
popt, pcov = curve_fit(exponenial_func, np.log(sales_volume.index.values), sales_volume.values)
print('Exponenial model parameters', popt)
sales_volume_exp = pd.Series(exponenial_func(np.log(sales_volume.index.values), popt[0], popt[1], popt[2]), index=sales_volume.index)
sales_volume_exp.plot(figsize=(10,8))
sales_volume.plot()
plt.show()

# 6. Расчитать логарифмическую регрессию по цене (Y = A + B * ln(X)), с помощью метода наименьших квадратов, где X - цена, Y - продажи в штуках
popt, pcov = curve_fit(log_func, np.exp(sales_volume.index.values), sales_volume.values)
print('Logarithmic model parameters', popt)
sales_volume_log = pd.Series(log_func(np.exp(sales_volume.index.values), popt[0], popt[1]), index=sales_volume.index)
sales_volume_log.plot(figsize=(10,8))
sales_volume.plot()
plt.show()

# Все результаты на одном графике
sales_volume_lin.plot(figsize=(10,8))
sales_volume_exp.plot(figsize=(10,8))
sales_volume_log.plot(figsize=(10,8))
plt.show()

