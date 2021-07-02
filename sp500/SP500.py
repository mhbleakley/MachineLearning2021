import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from matplotlib import style
from sklearn import linear_model
from SP500_toolbox import *

reg = linear_model.LinearRegression()
style.use("ggplot")  # style
yf.pdr_override()  # yahoo finance override

aapl = pd.read_csv("stock_dfs/AAPL.csv")


def prep_data(df, future=1, return_date=False):
    ndf = df.copy()
    ndf.drop(["Open", "High", "Low", "Close", "Volume"], 1, inplace=True)  # leaving date and adj close
    ndf = simple_moving_average(ndf, 5, False)  # column 2 (0 index)
    ndf = exponential_moving_average(ndf, 5, False)  # column 3
    ndf = macd(ndf, 1, False)  # column 4
    for day in range(-1, -future - 1, -1):
        ndf["D{} Close".format(-day)] = ndf["Adj Close"].shift(periods=day)
    ndf.drop(ndf.tail(future).index, inplace=True)
    ndf.fillna(0, inplace=True)
    if return_date:
        return ndf.iloc[:, 0], ndf.iloc[:, 1:5], ndf.iloc[:, 5:]
    else:
        return ndf.iloc[:, 1:5], ndf.iloc[:, 5:]


X_train, y_train = prep_data(aapl.iloc[:5000], 5)
date, X_test, y_test = prep_data(aapl[5000:], 5, True)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
predictions = pd.DataFrame().from_records(y_pred)
predictions.transpose()
predictions.columns = ["D{} Predicted".format(i) for i in range(1, 6)]
y_test = y_test.shift(periods=5)
# predictions = predictions.shift(periods=5)
y_test = y_test.reset_index()
y_test = y_test.join(predictions)
results = pd.DataFrame()
results["Date"] = date
results = results.reset_index()
results = results.drop(columns=["index"])
results["D5 Actual"] = y_test["D5 Close"]
results["D5 Pred."] = y_test["D5 Predicted"]
results.drop(results.head(5).index, inplace=True)


# print(results.tail(20))
print(X_test.tail(10))
print(y_test["D5 Close"].tail(10))
print(predictions.tail(10))

plt.plot(results["Date"], results["D5 Actual"])
plt.plot(results["Date"], results["D5 Pred."])


# plt.plot(aapl["Date"][5005:], aapl["Adj Close"][5005:])
# plt.plot(aapl["Date"][5005:], y_test["D5 Close"])
# plt.grid()
plt.tight_layout()
plt.show()
