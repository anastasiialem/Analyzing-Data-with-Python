"""Module 4"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
lr=LinearRegression()

URL="automobile/imports-85.data"
df=pd.read_csv(URL,header=None)
headers=["symboling", "normalized-losses", "make", "fuel-type", \
"aspiration", "num-of-doors", "body-style", "drive-wheels", \
"engine-location", "wheel-base", "length", "width", "height", \
"curb-weight", "engine-type", "num-of-cylinders", "engine-size", \
"fuel-system", "bore", "stroke", "compression-ratio", "horsepower", \
"peak-rpm", "city-mpg", "highway-mpg", "price"]
df.columns=headers
df["price"].replace("?",0, inplace=True)
df["price"]=df["price"].astype("float")
df["curb-weight"].replace("?",0, inplace=True)
df["curb-weight"]=df["curb-weight"].astype("float")
df["horsepower"].replace("?",0, inplace=True)
df["horsepower"]=df["horsepower"].astype("float")


x=df[['highway-mpg']]
y=df['price']
lr.fit(x,y)
yhat=lr.predict(x)
#print(yhat,lr.intercept_,lr.coef_)

lm=LinearRegression()
z=df[['horsepower', 'curb-weight','highway-mpg']]
dy=df['price']
lm.fit(z,dy)
pred=lm.predict(z)
#print(pred,lm.intercept_,lm.coef_)

sns.regplot(x='highway-mpg',y='price',data=df)
plt.ylim(0,)

axl=sns.displot(df['price'],color='r',label='Actual Value')
#sns.displot(yhat,color='r',label='Fitted Value',ax=axl)

pr=PolynomialFeatures(degree=2,include_bias=False)
print(pr)

inputi=[('polonomial',PolynomialFeatures(degree=2)),('scale',StandardScaler()),\
        ('module',LinearRegression())]
pipe=Pipeline(inputi)
