import pandas as pd
import numpy as np
df=pd.read_csv("dataset.csv")
print("shape:",df.shape)
print(df.head())
df.info()
print(df.columns)
print(df.dtypes)
print(df.isnull().sum())
df['total_bedrooms']=df['total_bedrooms'].fillna(df['total_bedrooms'].median())
df.drop_duplicates(inplace=True)
df=df[df['households']>0]
df=df[df['total_rooms']>0]
df=df[df['population']>0]
print(df.isnull().sum())
print(df.describe())
print(df['ocean_proximity'].value_counts())
print(df[df['median_house_value']>300000].head())
print(df[df['median_income']>5].head())
print(df[df['housing_median_age']>30].head())
print(df[(df['median_income']>5) & (df['median_house_value']>200000)].head())
print(df[(df['total_rooms']>2000) & (df['population']<2000)].head())
print("avg price:",df['median_house_value'].mean())
print("median price:",df['median_house_value'].median())
print("max price:",df['median_house_value'].max())
print("min price:",df['median_house_value'].min())
print("avg income:",df['median_income'].mean())
print(df.groupby('ocean_proximity')['median_house_value'].mean())
print(df.groupby('ocean_proximity')['median_income'].mean())
print(df.groupby('ocean_proximity')['total_rooms'].mean())
df['rooms_per_household']=df['total_rooms']/df['households']
df['bedrooms_per_room']=df['total_bedrooms']/df['total_rooms']
df['pop_per_household']=df['population']/df['households']
df['income_per_person']=df['median_income']/df['population']
df=df[df['rooms_per_household']<20]
df=df[df['bedrooms_per_room']<1]
df=df[df['pop_per_household']<10]
print(df[['rooms_per_household','bedrooms_per_room','pop_per_household']].head())
print(df[df['rooms_per_household']>5].head())
print(df[df['bedrooms_per_room']>0.5].head())
print(df[(df['median_income']>6) & (df['rooms_per_household']>4)].head())
print(df.corr(numeric_only=True)['median_house_value'].sort_values(ascending=False))
high_price=df[df['median_house_value']>400000]
low_price=df[df['median_house_value']<150000]
print("high price count:",high_price.shape[0])
print("low price count:",low_price.shape[0])
print(high_price[['median_income','total_rooms']].head())
print(low_price[['median_income','total_rooms']].head())
print(df.groupby(pd.cut(df['median_income'],bins=5))['median_house_value'].mean())
print(df.groupby(pd.cut(df['housing_median_age'],bins=5))['median_house_value'].mean())
df['price_per_room']=df['median_house_value']/df['total_rooms']
print(df[['price_per_room']].head())
df=df[df['price_per_room']<500]
print(df.corr(numeric_only=True)['median_house_value'].sort_values(ascending=False))
print(df.sample(10))
print(df.head())