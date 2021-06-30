import pandas as pd
import yfinance as yf
from matplotlib import style
from sklearn import linear_model
from SP500_toolbox import *

reg = linear_model.LinearRegression()
style.use("ggplot")  # style
yf.pdr_override()  # yahoo finance override

aapl = pd.read_csv("sp500/stock_dfs/AAPL.csv")


def prep_data(df, future=1):
    ndf = df.copy()
    ndf.drop(["Open", "High", "Low", "Close", "Volume"], 1, inplace=True)  # leaving date and adj close
    ndf = simple_moving_average(ndf, 5, False)  # column 2 (0 index)
    ndf = exponential_moving_average(ndf, 5, False)  # column 3
    ndf = macd(ndf, 1, False)  # column 4
    for day in range(-1, -future - 1, -1):
        ndf["D{} Close".format(-day)] = ndf["Adj Close"].shift(periods=day)
    ndf.drop(ndf.tail(future).index, inplace=True)
    ndf.fillna(0, inplace=True)
    return ndf.iloc[:, 1:5], ndf.iloc[:, 5:]
    # else:
    #     ndf.fillna(0, inplace=True)
    #     return ndf.iloc[:, 1:5], ndf.iloc[:, 5:]


def isolate_result(prediction, actual, day):
    adf = actual.copy()
    pdf = prediction.copy()
    new_df = pd.DataFrame()
    new_df["Actual"] = adf["D{} Close".format(day)]
    new_df["Predicted"] = pdf[day-1][:]
    return new_df


X_train, y_train = prep_data(aapl.iloc[:5000], 5)
X_test, y_test = prep_data(aapl[5000:], 5)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(y_pred)
print(y_test)
# print(X_test.index)
print(isolate_result(y_pred, y_test, 4).head())


# plt.plot(aapl["Date"][5005:], aapl["Adj Close"][5005:])
# plt.plot(aapl["Date"][5005:], y_test["D5 Close"])
# plt.grid()
# plt.tight_layout()
# plt.show()
