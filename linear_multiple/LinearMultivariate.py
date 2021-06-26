# price = x1*f1 + x2*f2 + x3*f3 + b
import pandas as pd
import numpy as mp
import math
from sklearn import linear_model

train = pd.read_csv("linear_multivariate_train.csv")
# there is a NaN in the bedrooms column
median_beds = math.floor(train.bedrooms.median())
train.bedrooms.fillna(median_beds, inplace=True)
reg = linear_model.LinearRegression().fit(train[["area", "bedrooms", "age"]], train.price)
print(reg.predict([[3000, 3, 40]]))
print(reg.coef_)

test = pd.read_csv("linear_mulivariate_test.csv")
median_beds_test = math.floor(test.bedrooms.median())
test.bedrooms.fillna(median_beds_test, inplace=True)
predictions = reg.predict(test)
print(predictions)
