# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:26:22 2024

@author: akshi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import lag_plot
from scipy.stats import skew, kurtosis
from feature_engine.outliers import Winsorizer
import scipy.stats as stats
import pylab
from sklearn.impute import SimpleImputer


data=pd.read_excel('C:/Users/akshi/Downloads/My prj/Data/Data.xlsx')
data['Billing date']=pd.to_datetime(data['Billing date'])
data=data.sort_values(by='Billing date')
#data.set_index('Billing date',inplace=True)
data.isna().sum()
data['monthly']=data['Billing date'].dt.to_period('M')

monthly_data=data.groupby(['monthly','Variant','Seasonality Factor']).agg({
    'Economic Index':'sum', 
    'Industry Growth Rate (%)':'sum'}).reset_index()

monthly_data['Variant Count'] = monthly_data.groupby(['monthly', 'Variant'])['Variant'].transform('count')

monthly_data.dtypes
duplicates = monthly_data.duplicated()
duplicate_rows = monthly_data[monthly_data.duplicated()]
print(duplicate_rows)

monthly_data=monthly_data.drop_duplicates()
#outliers 
sns.boxplot(monthly_data['Economic Index'])
sns.boxplot(monthly_data['Industry Growth Rate (%)'])

#missing values
monthly_data['Economic Index'].isnull().sum()
monthly_data['Industry Growth Rate (%)'].isnull().sum()
monthly_data['Variant'].isna().sum()

stats.probplot(monthly_data['Economic Index'], dist="norm", plot=pylab)
monthly_data['Economic Index'].skew()
monthly_data['Economic Index']=np.log(monthly_data['Economic Index'])
stats.probplot(monthly_data['Economic Index'], dist="norm", plot=pylab)
monthly_data['Economic Index'].skew()

stats.probplot(monthly_data['Industry Growth Rate (%)'],dist='norm',plot=pylab)
monthly_data['Industry Growth Rate (%)'].skew()
monthly_data['Industry Growth Rate (%)'] = np.sqrt(monthly_data['Industry Growth Rate (%)'])
monthly_data['Industry Growth Rate (%)'].isna().sum()
stats.probplot(monthly_data['Industry Growth Rate (%)'], dist="norm", plot=plt)
monthly_data['Industry Growth Rate (%)'].skew()

mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
monthly_data["Industry Growth Rate (%)"] = pd.DataFrame(mean_imputer.fit_transform(data[['Industry Growth Rate (%)']]))
monthly_data['Industry Growth Rate (%)'].isnull().sum()

#line plot
 
# Plot the original time series data
plt.figure(figsize=(7, 5))
plt.plot(monthly_data['Economic Index'], label='Time Series')
plt.title('Economic index Time Series')
plt.xlabel('monthly')
plt.ylabel('economic index')
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(monthly_data['Industry Growth Rate (%)'], label='Time Series')
plt.title('growth rate Time Series')
plt.xlabel('monthly')
plt.ylabel('growth rate')
plt.legend()
plt.show()

#seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(monthly_data['Economic Index'], model='additive',period=12)
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(monthly_data['Economic Index'])
plt.title('Time Series')
plt.subplot(4, 1, 2)
plt.plot(decomposition.trend.dropna())
plt.title('Trend Component')
plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal.dropna())
plt.title('Seasonal Component')
plt.subplot(4, 1, 4)
plt.plot(decomposition.resid.dropna())
plt.title('Residual Component')
plt.tight_layout()
plt.show()


decomposition = seasonal_decompose(monthly_data['Industry Growth Rate (%)'], model='additive',period=12)
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(data['Industry Growth Rate (%)'])
plt.title(' Time Series')
plt.subplot(4, 1, 2)
plt.plot(decomposition.trend.dropna())
plt.title('Trend Component')
plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal.dropna())
plt.title('Seasonal Component')
plt.subplot(4, 1, 4)
plt.plot(decomposition.resid.dropna())
plt.title('Residual Component')
plt.tight_layout()
plt.show()

monthly_data.to_csv('data_final.csv', index=False)
from IPython.display import FileLink

# Provide a download link for the CSV file
FileLink('data_final.csv')
