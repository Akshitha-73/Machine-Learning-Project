# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:04:12 2024

@author: akshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import  Holt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
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
variant=monthly_data['Variant'].unique()
# Create a dictionary to store model results for each variant
mv_re1={}
for variant in variant:
    variant_data=monthly_data[monthly_data['Variant']==variant].copy()
    tar=variant_data['Economic Index'].values
    
    split=int(len(tar)*0.8)
    train=tar[:split]
    test=tar[split:]
    
    model=Holt(train).fit(smoothing_level=0.2,smoothing_slope=0.1)
    pred=model.predict(len(train),len(tar)-1)
    
    mae=mean_absolute_error(test,pred)
    mse=mean_squared_error(test,pred)
    mape=np.mean(np.abs((np.array(test)-np.array(pred))/np.array(pred)))*100
    
    mv_re1[variant]={'mape':mape,'mae':mae,'mse':mse}
    print(f'variant {variant}, mape : {mape}')   
    
mv_re1

    
# for industry growth rate

variant=monthly_data['Variant'].unique()
# Create a dictionary to store model results for each variant
mv_re2={}
for variant in variant:
    variant_data=monthly_data[monthly_data['Variant']==variant].copy()
    tar=variant_data['Industry Growth Rate (%)'].values
    
    split=int(len(tar)*0.8)
    train=tar[:split]
    test=tar[split:]
    
    model=Holt(train).fit(smoothing_level=0.2,smoothing_slope=0.1)
    pred=model.predict(len(train),len(tar)-1)
    
    mae=mean_absolute_error(test,pred)
    mse=mean_squared_error(test,pred)
    mape=np.mean(np.abs((np.array(test)-np.array(pred))/np.array(pred)))*100
    
    mv_re2[variant]={'mape':mape,'mae':mae,'mse':mse}
    print(f'variant {variant}, mape : {mape}')   

mv_re2