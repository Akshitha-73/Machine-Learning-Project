# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:35:42 2024

@author: akshi
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
# Create a dictionary to store MA model results
se_model_results = {}
variants = data['Variant'].unique()
# Loop through each variant

for variant in variants:
    # Filter data for the current variant
    variant_data=monthly_data[monthly_data['Variant']==variant].copy()
    tar=variant_data['Economic Index'].values
    
    split=int(len(tar)*0.8)
    train=tar[:split]
    test=tar[split:]
    
    model=SimpleExpSmoothing(train).fit(smoothing_level=0.2)
    pred=model.predict(len(train),len(tar)-1)
    
    mae=mean_absolute_error(test,pred)
    mse=mean_squared_error(test,pred)
    mape=np.mean(np.abs((np.array(test)-np.array(pred))/np.array(pred)))*100
    
    se_model_results[variant]={'mape':mape,'mae':mae,'mse':mse}
    print(f'variant {variant}, mape : {mape}')

se_model_results

#for industry growth rate

se_model_results2 = {}
variants = data['Variant'].unique()
# Loop through each variant

for variant in variants:
    # Filter data for the current variant
    variant_data=monthly_data[monthly_data['Variant']==variant].copy()
    tar=variant_data['Industry Growth Rate (%)'].values
    
    split=int(len(tar)*0.8)
    train=tar[:split]
    test=tar[split:]
    
    model=SimpleExpSmoothing(train).fit(smoothing_level=0.2)
    pred=model.predict(len(train),len(tar)-1)
    
    mae=mean_absolute_error(test,pred)
    mse=mean_squared_error(test,pred)
    mape=np.mean(np.abs((np.array(test)-np.array(pred))/np.array(pred)))*100
    
    se_model_results2[variant]={'mape':mape,'mae':mae,'mse':mse}
    print(f'variant {variant}, mape : {mape}')
    
se_model_results2
