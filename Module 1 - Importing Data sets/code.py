"""Module cod to Module 1"""
import pandas as pd
URL="automobile/imports-85.data"
df=pd.read_csv(URL,header=None)
print(df)
print(df.head(5)) #5-number of first 2 row to print
print(df.tail(2)) #last 2 rows
headers=["symboling", "normalized-losses", "make", "fuel-type", \
"aspiration", "num-of-doors", "body-style", "drive-wheels", \
"engine-location", "wheel-base", "length", "width", "height", \
"curb-weight", "engine-type", "num-of-cylinders", "engine-size", \
"fuel-system", "bore", "stroke", "compression-ratio", "horsepower", \
"peak-rpm", "city-mpg", "highway-mpg", "price"]
df.columns=headers
print(df.head(5))
print(df.dtypes) #table of types in each colum
print(df.describe()) #specific math features
print(df.describe(include='all')) #not only numbers
