# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 21:26:21 2024

@author: akshi
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

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
# for economic index

# Create a dictionary to store the results for each variant
sarima_results = {}

# Loop through each unique variant
variants = monthly_data['Variant'].unique()
for variant in variants:
    variant_data = monthly_data[monthly_data['Variant'] == variant].copy()
    
    # Sort data by date if not already sorted
    variant_data = variant_data.sort_values('monthly')
    
    # Define the target variable 'Economic Index'
    y = variant_data['Economic Index']
    
    # Split the data into train and test sets
    train_size = int(len(y) * 0.8)
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Define the SARIMA model order (adjust p, d, q, and seasonal order based on your data)
    sarima_order = (1, 1, 1)        # Example non-seasonal order
    seasonal_order = (1, 1, 1, 12)  # Example seasonal order (12 for monthly data)
    
    # Fit the SARIMA model
    model = SARIMAX(y_train, order=sarima_order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    sarima_model = model.fit(disp=False)
    
    # Forecast on test set
    y_pred = sarima_model.predict(start=len(y_train), end=len(y)-1, dynamic=False)
    
    # Calculate error metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape=np.mean(np.abs((np.array(y_test)-np.array(y_pred))/np.array(y_pred)))*100
    
    # Store the results
    sarima_results[variant] = {'mape': mape,'mae': mae, 'mse': mse }
    print(f'Variant {variant} - MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%')

sarima_results

sarima_results2 = {}

# Loop through each unique variant
variants = monthly_data['Variant'].unique()
for variant in variants:
    variant_data = monthly_data[monthly_data['Variant'] == variant].copy()
    
    # Sort data by date if not already sorted
    variant_data = variant_data.sort_values('monthly')
    
    # Define the target variable 'Economic Index'
    y = variant_data['Industry Growth Rate (%)']
    
    # Split the data into train and test sets
    train_size = int(len(y) * 0.8)
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Define the SARIMA model order (adjust p, d, q, and seasonal order based on your data)
    sarima_order = (1, 1, 1)        # Example non-seasonal order
    seasonal_order = (1, 1, 1, 12)  # Example seasonal order (12 for monthly data)
    
    # Fit the SARIMA model
    model = SARIMAX(y_train, order=sarima_order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    sarima_model = model.fit(disp=False)
    
    # Forecast on test set
    y_pred = sarima_model.predict(start=len(y_train), end=len(y)-1, dynamic=False)
    
    # Calculate error metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape=np.mean(np.abs((np.array(y_test)-np.array(y_pred))/np.array(y_pred)))*100
    
    # Store the results
    sarima_results2[variant] = {'mape': mape,'mae': mae, 'mse': mse }
    print(f'Variant {variant} - MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%')
    
sarima_results2