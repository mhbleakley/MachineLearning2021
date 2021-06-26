import bs4 as bs
import datetime as dt
import os
import pandas as pd
from pandas_datareader import data as pdr
import pickle
import requests
import matplotlib.pyplot as plt
import numpy as np


def save_sp500_tickers():  # get a list of tickers
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


def get_data_from_yahoo(reload_sp500=False):  # get data corresponding to those tickers
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


def simple_moving_average(df, days):
    df["{}DSMA".format(days)] = df["Adj Close"].rolling(days).mean()
    return df


def exponential_moving_average(df, days):
    df["{}DEMA".format(days)] = df["Adj Close"].ewm(span=days, adjust=False).mean()
    return df


def macd(df, period=1):
    short, long = period * 12, period * 26
    df["{}-{}MACD".format(short, long)] = exponential_moving_average(df, short)["{}DEMA".format(short)]\
                                          - exponential_moving_average(df, long)["{}DEMA".format(long)]
    return df
