import bs4 as bs
import datetime as dt
import os
import pandas as pd
from pandas_datareader import data as pdr
import pickle
import requests
import matplotlib.pyplot as plt
import numpy as np


# returns a list of tickers (of the S&P500) from wikpedia
def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.', '-')
        ticker = ticker[:-1]
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers


# collects tickers information from yahoo finance and saves them as dataframes
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2021, 6, 21)
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


# compiles all the adjusted closes of all stocks into a csv
def compile_joined_closes():
    with open("sp500tickers.pickle", "rb") as f:  # read bytes
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv("sp500/stock_dfs/{}.csv".format(ticker))
        df.set_index("Date", inplace=True)

        df.rename(columns={"Adj Close": ticker}, inplace=True)
        df.drop(["Open", "High", "Low", "Close", "Volume"], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how="outer")

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv("sp500_joined_closes.csv")


# compiles all the opens of all stocks into a csv
def compile_joined_opens():
    with open("sp500tickers.pickle", "rb") as f:  # read bytes
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv("sp500/stock_dfs/{}.csv".format(ticker))
        df.set_index("Date", inplace=True)

        df.rename(columns={"Open": ticker, "High": ticker}, inplace=True)
        df.drop(["Adj Close", "Low", "Close", "Volume"], 1, inplace=True)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how="outer")

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv("sp500_joined_opens.csv")


# compiles all the daily changes of all stocks into a csv
def compile_percent_deltas():
    with open("sp500tickers.pickle", "rb") as f:  # read bytes
        tickers = pickle.load(f)
    print(tickers)
    main_df = pd.DataFrame()
    for count, ticker in enumerate(tickers):
        df = pd.read_csv("stock_dfs/{}.csv".format(ticker))
        df.set_index("Date", inplace=True)
        df["Percent Delta"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100
        df.rename(columns={"Percent Delta": ticker}, inplace=True)
        df.drop(["High", "Low", "Close", "Volume", "Open", "Adj Close"], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how="outer")

        if count % 10 == 0:
            print(count)
    print(main_df)
    print(main_df.describe())
    main_df.to_csv("sp500_percent_deltas_today.csv")


# compiles percent change of all stocks from day to day
def compile_daily_percent_deltas():
    with open("sp500tickers.pickle", "rb") as f:  # read bytes
        tickers = pickle.load(f)
    print(tickers)
    main_df = pd.DataFrame()
    for count, ticker in enumerate(tickers):
        df = pd.read_csv("stock_dfs/{}.csv".format(ticker))
        df.set_index("Date", inplace=True)
        df["Percent Delta Today"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100
        df["Percent Delta Tomorrow"] = df["Percent Delta Today"]
        df.shift(periods=1, fill_value=0)
        df.rename(columns={"Percent Delta": ticker}, inplace=True)
        df.drop(["High", "Low", "Close", "Volume", "Open", "Adj Close"], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how="outer")

        if count % 10 == 0:
            print(count)
    print(main_df)
    print(main_df.describe())
    main_df.to_csv("sp500_percent_deltas_today.csv")


# visualizes day to day stock correlations
def visualize_data():
    df = pd.read_csv("sp500_percent_deltas.csv")
    df_corr = df.corr()

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # nrows ncols index, also 111

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)  # color map
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + .5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + .5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(labels=column_labels)
    ax.set_yticklabels(labels=row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)  # color limit
    plt.tight_layout()  # "clean things up"
    # plt.show()
    print(df_corr.head())


# correlation of yesterday's change in price to today's for all stocks
def delta_correlations():
    df = pd.read_csv("sp500/sp500_percent_deltas.csv")
    main_df = pd.DataFrame()
    corrs = []
    tickers = save_sp500_tickers()
    for ticker in save_sp500_tickers():
        stock = df[ticker].to_frame()
        stock_yesterday = stock.shift(periods=1)
        stock.drop([0], inplace=True)
        stock_yesterday.drop([0], inplace=True)
        stock = stock.join(stock_yesterday, lsuffix="_TODAY", rsuffix="_YESTERDAY")
        corrs.append(float(stock.corr().iloc[1][0]))
    # main_df.set_index(tickers)
    main_df = main_df.join(corrs)
    print(main_df.head())


# takes a stock df and a period in days
# averages the adjusted close over all those days
# returns df with SMA column
def simple_moving_average(df, days, series=True):
    ndf = df.copy()
    ndf["{}DSMA".format(days)] = ndf["Adj Close"].rolling(days).mean()
    if series:
        return ndf["{}DSMA".format(days)]
    else:
        return ndf


# takes a stock df and a period in days
# averages the adjusted close over all those days, giving more weight to recent ones
# returns df with EMA column
def exponential_moving_average(df, days, series=True):
    ndf = df.copy()
    ndf["{}DEMA".format(days)] = ndf["Adj Close"].ewm(span=days, adjust=False).mean()
    if series:
        return ndf["{}DEMA".format(days)]
    else:
        return ndf


def macd(df, period=1, series=True):
    ndf = df.copy()
    short, long = period * 12, period * 26
    ndf["{}-{}MACD".format(short, long)] = exponential_moving_average(ndf, short)\
                                           - exponential_moving_average(ndf, long)
    if series:
        return ndf["{}-{}MACD".format(short, long)]
    else:
        return ndf
