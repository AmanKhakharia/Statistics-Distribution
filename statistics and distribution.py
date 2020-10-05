import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

path="D:\\Machine Learning\\house_prices.csv"
df=pd.read_csv(path)
print(df.head())
#df.info()

saleprice=df['SalePrice']

mean=saleprice.mean()
median=saleprice.median()
mode=saleprice.mode()

print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode)
plt.figure(figsize=(10,5))

plt.hist(saleprice,bins=100,color='skyblue')
plt.axvline(mean,color='red',label='Mean')
plt.axvline(median,color='yellow',label='Median')
plt.axvline(mode[0],color='green',label='Mode')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print("Start Value",saleprice.cumsum().head())

print("Minimum Value",saleprice.min())
print("Maximum Value",saleprice.max())
#Range
print("Range",saleprice.max()-saleprice.min())
#Variance
print("Variance",saleprice.var())

from math import sqrt

std=sqrt(saleprice.var())
print("Standard Deviation",std)

#skewness
print("Skewness",saleprice.skew())
#kurtosis
print("Kurtosis",saleprice.kurt())

#convert pandas DataFrame object to numpy array and sort
h = np.asarray(df['SalePrice'])
h = sorted(h) 
#use the scipy stats module to fit a normal distirbution with same mean and standard deviation
fit = stats.norm.pdf(h, np.mean(h), np.std(h))
#plot both series on the histogram
plt.plot(h,fit,'--',linewidth = 2,label="Normal distribution with same mean and var")
plt.hist(h,normed=True,bins = 100,label="Actual distribution")
plt.legend()
plt.show() 

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
corelation=df[['LotArea','GrLivArea','GarageArea','SalePrice']].corr()
print(corelation)
sns.heatmap(corelation)

#covariance
print("Covariance", df[['LotArea','GrLivArea','GarageArea','SalePrice']].cov().head())

print("Median q2",saleprice.quantile(0.5))
q3=saleprice.quantile(0.75)
print("q3",q3)
q1=saleprice.quantile(0.25)
print("q1",q1)
IQR=q3-q1
print("IQR",IQR)
plt.boxplot(saleprice)
plt.show()







