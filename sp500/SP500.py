import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from matplotlib import style
from sklearn import linear_model
from SP500_toolbox import *


style.use("ggplot")  # style
yf.pdr_override()  # yahoo finance override


def train_data_linear(ticker, future=1, split=0.75):
    model = linear_model.LinearRegression()
    stock = pd.read_csv("sp500/stock_dfs/{}.csv".format(ticker))
    n = len(stock.iloc[:, 0])
    X_train, y_train = prep_data(stock.iloc[:5000], future)
    date, X_test, y_test = prep_data(stock[5000:], future, True)
    model.fit(X_train, y_train)
    return model


def test_data_linear(ticker, model, return_date=True, future=1):
    stock = pd.read_csv("sp500/stock_dfs/{}.csv".format(ticker))
    date, X_test, y_test = prep_data(stock[5000:], future, return_date=True)
    y_pred = model.predict(X_test)
    print(model.score(X_test, y_test))
    predictions = pd.DataFrame().from_records(y_pred)
    predictions.transpose()
    predictions.columns = ["D{} Predicted".format(i) for i in range(1, future + 1)]
    y_test = y_test.shift(periods=future).reset_index()
    # y_test = y_test.reset_index()
    y_test = y_test.join(predictions)
    if return_date:
        return y_test, date
    return y_test


def isolate_result(data, day, future=1):
    results = pd.DataFrame()
    results["Date"] = date
    results = results.reset_index()
    results = results.drop(columns=["index"])
    results["D{} Actual".format(day)] = data["D{} Close".format(day)]
    results["D{} Pred.".format(day)] = data["D{} Predicted".format(day)]
    results.drop(results.head(future).index, inplace=True)
    print(results.tail(20))
    return results


mod = train_data_linear("AAPL", 5, .75)
result, date = test_data_linear("AAPL", mod, return_date=True, future=5)
isolated = isolate_result(result, 5, 5)
# print(X_test.tail(10))
# print(y_test["D5 Close"].tail(10))
# print(predictions.tail(10))

plt.plot(isolated["Date"], isolated["D5 Actual"])
plt.plot(isolated["Date"], isolated["D5 Pred."])


# plt.plot(aapl["Date"][5005:], aapl["Adj Close"][5005:])
# plt.plot(aapl["Date"][5005:], y_test["D5 Close"])
# plt.grid()
plt.tight_layout()
plt.show()
