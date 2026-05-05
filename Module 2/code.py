"""Module 2"""
import pandas as pd
import numpy as np

URL="automobile/imports-85.data"
df=pd.read_csv(URL,header=None)
headers=["symboling", "normalized-losses", "make", "fuel-type", \
"aspiration", "num-of-doors", "body-style", "drive-wheels", \
"engine-location", "wheel-base", "length", "width", "height", \
"curb-weight", "engine-type", "num-of-cylinders", "engine-size", \
"fuel-system", "bore", "stroke", "compression-ratio", "horsepower", \
"peak-rpm", "city-mpg", "highway-mpg", "price"]
df.columns=headers
df["normalized-losses"].replace("?", np.nan, inplace=True)
#if inplace is false we dont change the df
df["normalized-losses"]=df["normalized-losses"].astype("float")
print(df)
df.dropna(subset=['price'], axis=0,inplace=True)
#axis=0 is to drop row and axis=1 is to drop column
print(df)
mean=df["normalized-losses"].mean()
print(mean)
df['normalized-losses'].replace(np.nan,mean,inplace=True)
print(df)

#Data formating
df["city-mpg"]=235/df["city-mpg"] #from gasoline to litter per kilometr
df.rename(columns={"city-mpg": 'city - L/100 km' }, inplace=True)
print(df)

#Bins
df["price"].replace("?", 0, inplace=True)
df["price"]=df["price"].astype("float")
bins=np.linspace(min(df['price']),max(df['price']),4)
categories=['Low','Medium','High']
df['price-binned']=pd.cut(df["price"], bins,labels=categories,include_lowest=True)
print(df)
print(pd.get_dummies(df['fuel-type']))
