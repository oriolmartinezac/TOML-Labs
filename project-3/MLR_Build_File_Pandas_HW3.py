# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:03:54 2022

@author: jose m barcelo ordinas

"""

#%%

import pandas as pd

#%%

########### INDICATE THE PATH IN WHICH YOU HAVE THE DATA FILE datos_TOML-17001.csv

###### SUBSTITUTE THE PATH HERE BY YOUR OWN PATH WHERE YOU HAVE COPIED THE DATA FILES
dir_path = "/home/oriol/Escritorio/TOML/project1/project-3/"

###### DATA FILES TO BE LOADED
data_NO2 = dir_path + "NO2_Manlleu.csv"
data_NO = dir_path + "NO_Manlleu.csv"
data_SO2 = dir_path + "SO2_Manlleu.csv"
data_PM10 = dir_path + "PM10_Manlleu.csv"
data_O3 = dir_path + "captor17013-sensor1.csv"

#%%
##### UPLOAD in a PANDAS FRAME the O3/environmental data 
#####. i.e., (O3 sensor data, reference data, temperature and Relative humidity) 

data_PR_O3 = pd.read_csv(data_O3, delimiter=';')
data_PR_O3.isnull().values.any()
data_PR_O3 = data_PR_O3.dropna()
data_PR_O3['date'] = pd.to_datetime(data_PR_O3['date']).dt.strftime('%Y-%m-%dT%H:%M')
data_PR_O3.head()
print('O3', data_PR_O3.shape)

#%%
##### UPLOAD NO2 data
data_PR_NO2 = pd.read_csv(data_NO2, delimiter=';')
data_PR_NO2.isnull().values.any()
data_PR_NO2 = data_PR_NO2.dropna()
data_PR_NO2.head()
print('NO2', data_PR_NO2.shape)

#%%
##### UPLOAD PM10 data
data_PR_PM10 = pd.read_csv(data_PM10, delimiter=';')
data_PR_PM10.isnull().values.any()
data_PR_PM10 = data_PR_PM10.dropna()
data_PR_PM10.head()
print('PM10', data_PR_PM10.shape)

#%%
##### UPLOAD NO data 
data_PR_NO = pd.read_csv(data_NO, delimiter=';')
data_PR_NO.isnull().values.any()
data_PR_NO = data_PR_NO.dropna()
data_PR_NO.head()
print('NO', data_PR_NO.shape)

#%%
##### UPLOAD SO2 data 
data_PR_SO2 = pd.read_csv(data_SO2, delimiter=';')
data_PR_SO2.isnull().values.any()
data_PR_SO2 = data_PR_SO2.dropna()
data_PR_SO2.head()
print('SO2', data_PR_SO2.shape)

#%%
#### MERGE O3 and environmental data withe the NO2 data
new_PR_data_inner = pd.merge(data_PR_O3, data_PR_NO2, how='inner', left_on='date', right_on='date')
new_PR_data_inner.head()
print('O3+NO2', new_PR_data_inner.shape)

#%%
#### MERGE the previous data with NO data
new_PR_data_inner = pd.merge(new_PR_data_inner, data_PR_NO, how='inner', left_on='date',right_on='date')
new_PR_data_inner.head()
print('O3+NO2+NO', new_PR_data_inner.shape)

#%%
#### MERGE the previous data with SO2 data
new_PR_data_inner = pd.merge(new_PR_data_inner, data_PR_SO2, how='inner', left_on='date', right_on='date')
new_PR_data_inner.head()

print('O3+NO2+NO+SO2', new_PR_data_inner.shape)

#%%
##### If you add PM10 the dataset reduces from 1089 data points to 584, 
##### so maybe don't do it since you will get a very reduced data set

#new_PR_data_inner = pd.merge(new_PR_data_inner, data_PR_PM10,how='inner', left_on='date',right_on='date')
#new_PR_data_inner.head()
#new_PR_data_inner.shape
#print('O3+NO2+NO+SO2+PM10',new_PR_data_inner.shape)

#%%
##### PRINT the HEADERS in the PANDAS data frame
print('headers of the pandas object new_PR_data_inner :\n', new_PR_data_inner.columns)


