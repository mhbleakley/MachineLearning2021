import matplotlib
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression as linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib import style
from SP500_toolbox import *
import yfinance as yf


style.use("ggplot")  # style
yf.pdr_override()  # yahoo finance override

# sp500/stock_dfs/AAPL.csv WINDOWS
# stock_dfs/AAPL.csv APPLE
accuracies = []

# update_stocks()


def ml_train(ticker, future=10, threshold=0.02):
    df = pd.read_csv("sp500/stock_dfs/{}.csv".format(ticker))
    dates = df["Date"]

    emas = [i for i in range(5, 11)]
    smas = [i for i in range(5, 11)]
    e = [5, 10, 15, 50, 200]
    s = [5, 10, 15, 50, 200]
    df = prep_columns(df, ema=e, sma=s, future=future)

    X = df.iloc[:, :len(df.columns) - 1]
    y = df.iloc[:, len(df.columns) - 1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=False)
    print(X_train)
    models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()),
              ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()), ('SVM', SVC(gamma='auto'))]  # , ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()), ('SVM', SVC(gamma='auto'))
    results = []
    names = []
    fitted = []
    bsh = extract_row(df, dates, "2021-06-02")

    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        model.fit(X_train, Y_train)
        ans = model.predict(bsh)
        print("{}: Accuracy {}, Std. ({}) {}".format(str(name), str(round(cv_results.mean(), 5)), str(round(cv_results.std(), 5)), str(ans[0])))


ml_train("MMM", future=20, threshold=0.03)
