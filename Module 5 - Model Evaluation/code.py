"""Module 5"""
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
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
df["price"].replace("?",0, inplace=True)
df["price"]=df["price"].astype("float")
df["horsepower"].replace("?",0, inplace=True)
df["horsepower"]=df["horsepower"].astype("float")


x_data=df['horsepower']
y_data=df['price']
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.3,random_state=0)
print(x_train)
print()
print(x_test)
lr=LinearRegression()
x=df[['horsepower']]
y=df['price']
lr.fit(x,y)
scores=cross_val_score(lr,x_data,y_data,cv=3)
print(np.mean(scores))

test=[]
order=[1,2,3,4]
for n in order:
    pr=PolynomialFeatures(degree=n)
    x_train_pr=pr.fit_transform(x_train[['horsepower']])
    x_test_pr=pr.fit_transform(x_test[['horsepower']])
    lr.fit(x_train,y_train)
    test.append(lr.score(x_test_pr,y_test))
print(test)

RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_data,y_data)
yhat=RidgeModel.predict(x_data)
