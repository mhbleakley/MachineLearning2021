import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import style
from sklearn import linear_model
style.use("dark_background")
# fit the data to a linear regression
train = pd.read_csv("linear_train.csv")
reg = linear_model.LinearRegression()
reg.fit(train[["area"]], train.price)
train_pred = reg.predict([[3300]])
# test the data on a different set
test = pd.read_csv("linear_test.csv")
test_pred = reg.predict(test)
test["price"] = test_pred
test.to_csv("linear_test_result.csv")
# plot data
plt.title("Price VS. Area")
plt.xlabel("Area (SQFT)")
plt.ylabel("Price ($USD)")
train_line, = plt.plot(train.area, reg.coef_*train.area + reg.intercept_, "y--")
# extrapolation, = plt.plot(range(max(train.area)), reg.coef_*range(max(train.area)) + reg.intercept_, "y-")
train_data = plt.scatter(train.area, train.price, color="green")
test_data = plt.scatter(test.area, test.price, color="red")
plt.legend([train_data, test_data, train_line], ["Training Set", "Testing Results", "Best Fit"])
plt.tight_layout()
plt.show()
