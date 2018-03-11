import pandas as pd
import csv
import numpy as np
import os
from functools import reduce
from pandas import DataFrame, Series 



GDP_Per_Capita=pd.read_csv('DATA/GDP-Per-Capita-all.csv',dtype=object,skiprows=[0,2,3],usecols=[1,*range(50,60)])

GDP_Per_Capita=GDP_Per_Capita.dropna()
GDP_Per_Capita.rename(columns={'Country Code': 'Country_code'}, inplace=True)
print(GDP_Per_Capita.head())

with open('1-GDP_Per_Capita.csv', 'w') as f:
    
    GDP_Per_Capita.to_csv(f, header=True,index=0,float_format='%.3f')
f.close()

################################################

Population_by_country=pd.read_csv('DATA/Population-by-country-all.csv',dtype=object,skiprows=[0,2,3],usecols=[1,*range(50,60)])

Population_by_country=Population_by_country.dropna()
Population_by_country.rename(columns={'Country Code': 'Country_code'}, inplace=True)
print(Population_by_country.head())

with open('2-Population_by_country.csv', 'w') as f:
    
    Population_by_country.to_csv(f, header=True,index=0)
f.close()
###################################################


colnames=['Country Code','Indicator','Year','Value'] 

Maths_Performance_2015=pd.read_csv('DATA/MathsPerformance/PISA_Math_all.csv', names=colnames,skiprows=[0], header=None,dtype=object,usecols=[0,1,5,6])
print(Maths_Performance_2015.head())

Maths_Performance_2015=Maths_Performance_2015[(Maths_Performance_2015['Year']>='2015')]

Maths_Performance_2015.rename(columns={'Country Code': 'Country_code'}, inplace=True)

with open('3-Maths_Performance_2015.csv', 'w') as f:
    
    Maths_Performance_2015.to_csv(f, header=True,index=0)
f.close()