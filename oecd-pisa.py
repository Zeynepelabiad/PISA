import pandas as pd
import csv
import numpy as np
import os
from functools import reduce
from pandas import DataFrame, Series 
import requests
import matplotlib
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


#GDP_Per_Capita=pd.read_csv('DATA/GDP-Per-Capita-all.csv',dtype=object,skiprows=[0,2,3],usecols=[1,*range(50,60)])
GDP_Per_Capita=pd.read_csv('DATA/GDP-Per-Capita-all.csv',skiprows=[0,2,3],usecols=[1,59])
GDP_Per_Capita=GDP_Per_Capita.dropna()
GDP_Per_Capita.rename(columns={'Country Code': 'Country_code','2015':'GDP_2015'}, inplace=True)
print(GDP_Per_Capita.head())

with open('1-GDP_Per_Capita.csv', 'w') as f:
    
    GDP_Per_Capita.to_csv(f, header=True,index=0,float_format='%.3f')
f.close()



################################################

#Population_by_country=pd.read_csv('DATA/Population-by-country-all.csv',dtype=object,skiprows=[0,2,3],usecols=[1,*range(50,60)])
Population_by_country=pd.read_csv('DATA/Population-by-country-all.csv',skiprows=[0,2,3],usecols=[1,59])
Population_by_country=Population_by_country.dropna()
Population_by_country.rename(columns={'Country Code': 'Country_code','2015':'Population_2015'}, inplace=True)
print(Population_by_country.head())

with open('2-Population_by_country.csv', 'w') as f:
    
    Population_by_country.to_csv(f, header=True,index=0)
f.close()
###################################################

#Life_expectancyy=pd.read_csv('DATA/Life_expectancy-all.csv',dtype=object,skiprows=[0,2,3],usecols=[1,*range(50,60)])
Life_expectancy=pd.read_csv('DATA/Life_expectancy_all.csv',skiprows=[0,2,3],usecols=[1,59])
Life_expectancy=Life_expectancy.dropna()
Life_expectancy.rename(columns={'Country Code': 'Country_code','2015':'Life_E_2015'}, inplace=True)
print(Life_expectancy.head())

with open('3-Life_expectancy.csv', 'w') as f:
    
    Life_expectancy.to_csv(f, header=True,index=0,float_format='%.3f')
f.close()

###################################################

Math_Performance_2015=pd.read_csv('DATA/MathPerformance/PISA_Math_all.csv',usecols=[0,1,2,5,6])
Math_Performance_2015.rename(columns={'LOCATION': 'Country_code','TIME':'Year','Value':'MATH2015'}, inplace=True)
print(Math_Performance_2015.head())

Math_Performance_2015=Math_Performance_2015[(Math_Performance_2015['Year']==2015) & (Math_Performance_2015['SUBJECT']=='TOT')]

Math_Performance_2015.rename(columns={'Country Code': 'Country_code'}, inplace=True)

with open('4-Math_Performance_2015.csv', 'w') as f:
    
    Math_Performance_2015.to_csv(f, header=True,index=0)
f.close()


Reading_Performance_2015=pd.read_csv('DATA/ReadingPerformance/PISA_Reading_all.csv',usecols=[0,1,2,5,6])
Reading_Performance_2015.rename(columns={'LOCATION': 'Country_code','TIME':'Year','Value':'READ2015'}, inplace=True)
print(Math_Performance_2015.head())

Reading_Performance_2015=Reading_Performance_2015[(Reading_Performance_2015['Year']==2015) & (Reading_Performance_2015['SUBJECT']=='TOT')]

Reading_Performance_2015.rename(columns={'Country Code': 'Country_code'}, inplace=True)

with open('5-Reading_Performance_2015.csv', 'w') as f:
    
    Reading_Performance_2015.to_csv(f, header=True,index=0)
f.close()


Science_Performance_2015=pd.read_csv('DATA/SciencePerformance/PISA_Science_all.csv',usecols=[0,1,2,5,6])
Science_Performance_2015.rename(columns={'LOCATION': 'Country_code','TIME':'Year','Value':'SCIENCE2015'}, inplace=True)
print(Science_Performance_2015.head())

Science_Performance_2015=Science_Performance_2015[(Science_Performance_2015['Year']==2015) & (Science_Performance_2015['SUBJECT']=='TOT')]

Science_Performance_2015.rename(columns={'Country Code': 'Country_code'}, inplace=True)

with open('6-Science_Performance_2015.csv', 'w') as f:
    
    Science_Performance_2015.to_csv(f, header=True,index=0)
f.close()



Pisa_df_2015=[Math_Performance_2015,Reading_Performance_2015,Science_Performance_2015,GDP_Per_Capita,Population_by_country,Life_expectancy]
Pisa_df_2015= reduce(lambda  left,right: pd.merge(left,right,on='Country_code', how='left'), Pisa_df_2015)
Pisa_df_2015=Pisa_df_2015.dropna()

with open('7-Pisa_df_2015.csv', 'w') as f:
    Pisa_df_2015.to_csv(f, header=True,index=0,columns=['Country_code','MATH2015','READ2015','SCIENCE2015','GDP_2015','Population_2015','Life_E_2015'])

f.close()


with sns.axes_style('white'):
    g = sns.jointplot("SCIENCE2015", "Life_E_2015", Pisa_df_2015, kind='hex')
    g.ax_joint.plot(np.linspace(0, 500),
                    np.linspace(50, 100), ':k')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
 
# create data
x = Pisa_df_2015['Country_code']
y = Pisa_df_2015['MATH2015']
z = Pisa_df_2015['Life_E_2015']
 
# use the scatter function
plt.scatter(x, y, s=z*5, alpha=0.5)
plt.show()


plt.style.use('ggplot')
Pisa_df_2015.plot(kind='barh', y="GDP_2015", x="Country_code")
plt.show()

plt.style.use('ggplot')
Pisa_df_2015.plot(kind='barh', y="Population_2015", x="Country_code")
plt.show()

plt.style.use('ggplot')
Pisa_df_2015.plot(kind='barh', y="Life_E_2015", x="Country_code")
plt.show()

plt.style.use('ggplot')
Pisa_df_2015.plot(kind='barh', y="MATH2015", x="Country_code")
plt.show()

plt.style.use('ggplot')
Pisa_df_2015.plot(kind='barh', y="READ2015", x="Country_code")
plt.show()

plt.style.use('ggplot')
Pisa_df_2015.plot(kind='barh', y="SCIENCE2015", x="Country_code")
plt.show()



with sns.axes_style('white'):
    sns.jointplot("READ2015", "Life_E_2015", data=Pisa_df_2015, kind='hex')
    plt.show()



sns.jointplot("GDP_2015", "Life_E_2015", data=Pisa_df_2015, kind='reg');
plt.show()

sns.jointplot("MATH2015", "GDP_2015", data=Pisa_df_2015, kind='reg');
plt.show()

sns.jointplot("MATH2015", "Life_E_2015", data=Pisa_df_2015, kind='reg');
plt.show()

sns.jointplot("SCIENCE2015", "Life_E_2015", data=Pisa_df_2015, kind='reg');
plt.show()

Pisa_df_2015.plot(kind='scatter', x='GDP_2015', y='Life_E_2015', c='Population_2015',figsize=[20,10])
plt.show()


from pandas.tools.plotting import scatter_matrix
areas = Pisa_df_2015[['Country_code','MATH2015','READ2015','SCIENCE2015','GDP_2015','Population_2015','Life_E_2015']]
scatter_matrix(areas, alpha=0.2, figsize=(18,18), diagonal='kde')
plt.show()


g = sns.pairplot(data=Pisa_df_2015[['Country_code','MATH2015','READ2015','SCIENCE2015','GDP_2015','Population_2015','Life_E_2015']], hue='Country_code', dropna=True)
plt.show()
"""