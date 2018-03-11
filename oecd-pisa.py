import pandas as pd
import csv
import numpy as np
import os
from functools import reduce
from pandas import DataFrame, Series 



GDP_Per_Capita=pd.read_csv('DATA/GDP-Per-Capita-all.csv',header=None,dtype=object,skiprows=[0,2,3],usecols=[1,*range(50,60)])

GDP_Per_Capita=GDP_Per_Capita.dropna()
print(GDP_Per_Capita.head())

with open('1-GDP_Per_Capita.csv', 'w') as f:
    
    GDP_Per_Capita.to_csv(f, header=False,index=0,float_format='%.3f')
f.close()



Population_by_country=pd.read_csv('DATA/Population-by-country-all.csv',header=None,dtype=object,skiprows=[0,2,3],usecols=[1,*range(50,60)])

Population_by_country=Population_by_country.dropna()
print(Population_by_country.head())

with open('2-Population_by_country.csv', 'w') as f:
    
    Population_by_country.to_csv(f, header=False,index=0)
f.close()



