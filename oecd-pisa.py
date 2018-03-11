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


Maths_Performance_2015=pd.read_csv('DATA/MathsPerformance/PISA_Maths_all.csv',dtype=object,usecols=[0,1,2,5,6])
Maths_Performance_2015.rename(columns={'LOCATION': 'Country_code','TIME':'Year'}, inplace=True)
print(Maths_Performance_2015.head())

Maths_Performance_2015=Maths_Performance_2015[(Maths_Performance_2015['Year']>='2015') & (Maths_Performance_2015['SUBJECT']=='TOT')]

Maths_Performance_2015.rename(columns={'Country Code': 'Country_code'}, inplace=True)

with open('3-Maths_Performance_2015.csv', 'w') as f:
    
    Maths_Performance_2015.to_csv(f, header=True,index=0)
f.close()



Reading_Performance_2015=pd.read_csv('DATA/ReadingPerformance/PISA_Reading_all.csv',dtype=object,usecols=[0,1,2,5,6])
Reading_Performance_2015.rename(columns={'LOCATION': 'Country_code','TIME':'Year'}, inplace=True)
print(Maths_Performance_2015.head())

Reading_Performance_2015=Reading_Performance_2015[(Reading_Performance_2015['Year']>='2015') & (Reading_Performance_2015['SUBJECT']=='TOT')]

Reading_Performance_2015.rename(columns={'Country Code': 'Country_code'}, inplace=True)

with open('4-Reading_Performance_2015.csv', 'w') as f:
    
    Reading_Performance_2015.to_csv(f, header=True,index=0)
f.close()



Science_Performance_2015=pd.read_csv('DATA/SciencePerformance/PISA_Science_all.csv',dtype=object,usecols=[0,1,2,5,6])
Science_Performance_2015.rename(columns={'LOCATION': 'Country_code','TIME':'Year'}, inplace=True)
print(Science_Performance_2015.head())

Science_Performance_2015=Science_Performance_2015[(Science_Performance_2015['Year']>='2015') & (Science_Performance_2015['SUBJECT']=='TOT')]

Science_Performance_2015.rename(columns={'Country Code': 'Country_code'}, inplace=True)

with open('5-Science_Performance_2015.csv', 'w') as f:
    
    Science_Performance_2015.to_csv(f, header=True,index=0)
f.close()

Pisa_df_2015=[Maths_Performance_2015,Reading_Performance_2015,Science_Performance_2015]
Pisa_df_2015= reduce(lambda  left,right: pd.merge(left,right,on='Country_code', how='left'), Pisa_df_2015)
with open('6-Pisa_df_2015.csv', 'w') as f:
    Pisa_df_2015.to_csv(f, header=True,index=0)