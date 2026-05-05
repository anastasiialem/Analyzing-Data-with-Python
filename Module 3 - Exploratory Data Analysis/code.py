"""Module 3"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

URL="automobile/imports-85.data"
df=pd.read_csv(URL,header=None)
headers=["symboling", "normalized-losses", "make", "fuel-type", \
"aspiration", "num-of-doors", "body-style", "drive-wheels", \
"engine-location", "wheel-base", "length", "width", "height", \
"curb-weight", "engine-type", "num-of-cylinders", "engine-size", \
"fuel-system", "bore", "stroke", "compression-ratio", "horsepower", \
"peak-rpm", "city-mpg", "highway-mpg", "price"]
df.columns=headers

#print(df.describe())
#print(df.value_counts())
dr_counts=df["drive-wheels"].value_counts()
#print(dr_counts)
print(sns.boxplot(x='drive-wheels', y='price',data=df))
x=df['price']
y=df['engine-type']
plt.scatter(x,y)

#GroupBy
#print(df['drive-wheels'])
df["price"].replace("?", 0, inplace=True)
df["price"]=df["price"].astype("float")
df1=df[['drive-wheels','body-style','price']]

df2=df1.groupby(['drive-wheels','body-style'],as_index=False).mean(numeric_only=True)
print(df2)
df_p=df2.pivot(index='drive-wheels',columns='body-style',values='price')
print(df_p)
plt.pcolor(df_p,cmap='RdBu')
plt.colorbar()
plt.show()

#Correlation
sns.regplot(x='engine-size',y='price',data=df)
plt.ylim(0,)
df['horsepower']=pd.to_numeric(df['horsepower'], errors='coerce')
df['price']=pd.to_numeric(df['price'], errors='coerce')
coef,p=stats.pearsonr(df["horsepower"],df["price"])
print((coef,p))
