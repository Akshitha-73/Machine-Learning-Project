# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 23:07:21 2024

@author: akshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

#load data
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

#for economic index
variant=monthly_data['Variant'].unique()
mv_re1={}
for variant in variant:
    variant_data=monthly_data[monthly_data['Variant']==variant].copy()
    X=variant_data.drop(columns=['Economic Index'])
    y=variant_data['Economic Index']
    X = X.select_dtypes(include=[np.number])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model=DecisionTreeRegressor()
    model.fit(X_train,y_train)    
    y_pre=model.predict(X_test)
    
    mae=mean_absolute_error(y_test,y_pre)
    mse=mean_squared_error(y_test,y_pre)
    mape=np.mean(np.abs((np.array(y_test)-np.array(y_pre))/np.array(y_pre)))*100
    
    mv_re1[variant]={'mape':mape,'mae':mae,'mse':mse}
    print(f'variant {variant}, mape : {mape}')
mv_re1


# for industry growth rate
variant=monthly_data['Variant'].unique()
mv_re2={}
for variant in variant:
    variant_data=monthly_data[monthly_data['Variant']==variant].copy()
    X=variant_data.drop(columns=['Industry Growth Rate (%)'])
    y=variant_data['Industry Growth Rate (%)']
    X = X.select_dtypes(include=[np.number])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model=DecisionTreeRegressor()
    model.fit(X_train,y_train)    
    y_pre=model.predict(X_test)
    
    mae=mean_absolute_error(y_test,y_pre)
    mse=mean_squared_error(y_test,y_pre)
    mape=np.mean(np.abs((np.array(y_test)-np.array(y_pre))/np.array(y_pre)))*100
    
    mv_re2[variant]={'mape':mape,'mae':mae,'mse':mse}
    print(f'variant {variant}, mape : {mape}')
mv_re2