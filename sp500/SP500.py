import yfinance as yf
from matplotlib import style
from sklearn import linear_model
from SP500_toolbox import *

reg = linear_model.LinearRegression()
style.use("ggplot")  # style
yf.pdr_override()  # yahoo finance override

aapl = pd.read_csv("sp500/stock_dfs/AAPL.csv")

# sma = simple_moving_average(aapl, 50)["50DSMA"][5000:]
# ema = exponential_moving_average(aapl, 50)["50DEMA"][5000:]
# MACD = macd(aapl)["26-12MACD"][5000:]
# adj_close = aapl["Adj Close"][5000:]
# time = aapl["Date"][5000:]

# plt.plot(time, sma)
# plt.plot(time, ema)
# plt.plot(time, MACD)
# plt.plot(time, adj_close)
# plt.grid()
# plt.tight_layout()
# plt.show()


def prep_data(df, future=1, training=False):
    df.drop(["Open", "High", "Low", "Close", "Volume"], 1, inplace=True)
    df = simple_moving_average(df, 20)
    df = exponential_moving_average(df, 20)
    df = macd(df)
    if training:
        for day in range(-1, -future - 1, -1):
            df["D{} Close".format(-day)] = df["Adj Close"].shift(periods=day)
        df.drop(df.tail(future).index, inplace=True)
    df.fillna(0, inplace=True)
    return df


print(prep_data(aapl, 5, True))
# aapl = pd.read_csv("sp500/stock_dfs/AAPL.csv")
# train = aapl.iloc[4050:]
# train = prep_data(train, 5, True)
# aapl = pd.read_csv("sp500/stock_dfs/AAPL.csv")
# test = aapl.iloc[:4050]
# test = prep_data(test, 5, False)
# aapl = pd.read_csv("sp500/stock_dfs/AAPL.csv")
# aapl = aapl.iloc[:4050]
# aapl = prep_data(aapl, 5, True)
# test.drop(test.tail(5).index, inplace=True)
# reg.fit(train[["Adj Close", "20DSMA", "20DEMA", "12-26MACD"]], train[["D1 Close", "D2 Close", "D3 Close", "D4 Close", "D5 Close"]])

# test_pred = reg.predict(test[["Adj Close", "20DSMA", "20DEMA", "12-26MACD"]])
# score = reg.score(test[["Open"]], aapl[["Adj Close"]][4050:])
# score = reg.score(test[["Adj Close", "20DSMA", "20DEMA", "12-26MACD"]], aapl[["D1 Close", "D2 Close", "D3 Close", "D4 Close", "D5 Close"]])
# print(score)
# plt.plot(test["Date"], aapl["Open"].iloc[4050:], "r")
# plt.plot(test["Date"], aapl["High"].iloc[4050:], "b")
# plt.plot(test["Date"], test["High"], "g")
# plt.grid()
# plt.tight_layout()
# plt.show()
