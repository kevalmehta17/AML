import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

df = pd.read_csv(r"C:\Users\keval\Downloads\housing.csv")
df.head(10)
df.isnull().sum()
df.isnull().sum()
df = df.drop_duplicates()
df.boxplot(column = ["total_rooms","total_bedrooms","population","households","median_income"])
plt.show()
Q1 = df["total_rooms"].quantile(0.25)
Q3 = df["total_rooms"].quantile(0.75)
IQR = Q3-Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

df["total_rooms"] = df["total_rooms"].where((df["total_rooms"]>lower_bound) & (df["total_rooms"]<upper_bound))
df["total_rooms"].fillna(df["total_rooms"].median())
Q1 = df["total_bedrooms"].quantile(0.25)
Q3 = df["total_bedrooms"].quantile(0.75)
IQR = Q3-Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5* IQR

df["total_bedrooms"] = df["total_bedrooms"].where((df["total_bedrooms"]>lower_bound) & (df["total_bedrooms"] < upper_bound))
df["total_bedrooms"].fillna(df["total_bedrooms"].median())
Q1 = df["population"].quantile(0.25)
Q3 = df["population"].quantile(0.75)
IQR = Q3-Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5* IQR

df["population"] = df["population"].where((df["population"]>lower_bound) & (df["population"] < upper_bound))
df["population"].fillna(df["population"].mean())
if "ocean_proximity" in df.columns:
    le = LabelEncoder()
    df["ocean_proximity_encoded"] = le.fit_transform(df["ocean_proximity"])
    df = df.drop(columns=["ocean_proximity"])

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso

df = df.fillna(df.ffill())
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
lr = LinearRegression()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=None)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

mae = metrics.mean_absolute_error(y_test,y_pred)
mse = metrics.mean_squared_error(y_test,y_pred)
print("mae:- ", mae)
print("mse:- ", mse)

# Ridge Regression

r = Ridge(alpha=0.1)
r.fit(x_train,y_train)
y_pred = r.predict(x_test)
mae = metrics.mean_absolute_error(y_test,y_pred)
mse = metrics.mean_squared_error(y_test,y_pred)
print("mae:- ", mae)
print("mse:- ", mse)

# Lasso Regression
l=Lasso(alpha=0.1)
l.fit(x_train,y_train)
y_pred = l.predict(x_test)
mae = metrics.mean_absolute_error(y_test,y_pred)
mse = metrics.mean_squared_error(y_test,y_pred)
print("mae:- ", mae)
print("mse:- ", mse)