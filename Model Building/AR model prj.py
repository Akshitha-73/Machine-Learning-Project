# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:39:14 2024

@author: akshi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score


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

# unique variants in the data
variant=monthly_data['Variant'].unique()
ar_res1={}
for Variant in variant:
    variant_data=monthly_data[monthly_data['Variant']==Variant].copy()
    tar=variant_data['Economic Index'].values
    
    split=int(len(tar)*0.8)
    train=tar[:split]
    test=tar[split:]  
    
    # Set lags to one less than the length of train
    model = AutoReg(train, lags=1).fit()  
    pre=model.predict(len(train),len(tar)-1)    
    mse=mean_squared_error(test, pre)
    mae=mean_absolute_error(test,pre)
    mape=np.mean(np.abs((np.array(test)-np.array(pre))/np.array(pre)))*100
    ar_res1[Variant]={'mape':mape,
        'mse':mse,'mae':mae}
                      
    print(f' variant:{Variant}, mape: {mape}')

fig, axes = plt.subplots(3, 4, figsize=(15, 10))  # 3 rows, 4 columns of subplots
axes = axes.flatten()      
for i, Variant in enumerate(variant):  # Use enumerate to get an index `i`
    variant_data = monthly_data[monthly_data['Variant'] == Variant].copy()
    tar = variant_data['Economic Index'].values
    axes[i].plot(test, label='Actual')  # Plot test data
    axes[i].plot(pre, color='red', label='Predicted')  # Plot predicted data
    axes[i].set_title(f'Variant {Variant}')
    axes[i].set_xlabel('X-axis')
    axes[i].set_ylabel('Economic Index')
    axes[i].legend()
# Adjust layout to prevent overlap
plt.tight_layout()
# Show the final figure with all subplots
plt.show()


#FOR industry growth rate
# unique variants in the data
variant=monthly_data['Variant'].unique()
# Create a dictionary to store model results for each variant
ar_res2={}
# Loop over each variant, build a model, and evaluate it
for variant in variant:
    variant_data=monthly_data[monthly_data['Variant']==variant].copy()
    tar=variant_data['Industry Growth Rate (%)'].values
    split=int(len(tar)*0.08)
    train=tar[:split]
    test=tar[split:]
    model=AutoReg(train, lags=1).fit()  
    pre=model.predict(len(train),len(tar)-1)
    mae=mean_absolute_error(test,pre)
    mse=mean_squared_error(test, pre)
    mape=np.mean(np.abs((np.array(test)-np.array(pre))/np.array(test)))*100 
    ar_res2[variant]={'mae':mae,'mape':mape}
    print(f'variant  {variant}, mape :{mape}')
   
    
fig,axes=plt.subplot(3,4,figsize=(15,10))
axes=axes.flatten()
for i, variant in enumerate(variant):
 # Use enumerate to get an index `i`
      variant_data = monthly_data[monthly_data['Variant'] == Variant].copy()
      tar = variant_data['Industry Growth Rate'].values
      axes[i].plot(test, label='Actual')  # Plot test data
      axes[i].plot(pre, color='red', label='Predicted')  # Plot predicted data
      axes[i].set_title(f'Variant {Variant}')
      axes[i].set_xlabel('X-axis')
      axes[i].set_ylabel('Industry Growth Rate (%)')
      axes[i].legend()

  # Adjust layout to prevent overlap
plt.tight_layout()
  # Show the final figure with all subplots
plt.show()  

