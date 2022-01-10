import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

data = pd.read_csv('pa0(train-only).csv')

print(data.shape)

data = data.iloc[:,1:]
print(data.head())
print(data.shape)

# convert date to year, month, day
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data = data.iloc[:,1:]
print(data.shape)

#unique values of bedrooms, bathrooms, floors
bedrooms_uniq = data.bedrooms.unique()
bathrooms_uniq = data.bathrooms.unique()
floors_uniq = data.floors.unique()

#plot boxplot
data.boxplot(column='price',by=['bedrooms'])
plt.savefig('bedrooms_boxplot.png')
data.boxplot(column='price',by=['bathrooms'])
plt.savefig('bathrooms_boxplot.png')
data.boxplot(column='price',by=['floors'])
plt.savefig('floors_boxplot.png')

#co-varience matrix
cov = data[['sqft_living','sqft_living15','sqft_lot','sqft_lot15']].cov()
print(cov)

#scatter plot
data.plot.scatter(x='sqft_living',y='sqft_living15')
plt.savefig('sqft_living_scatter.png')
data.plot.scatter(x='sqft_lot',y='sqft_lot15')
plt.savefig('sqft_lot_scatter.png')