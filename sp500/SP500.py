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
from sklearn.linear_model import LinearRegression as linear_model
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


# def train_data_linear(ticker, future=1, split=0.75):
#     model = linear_model.LinearRegression()
#     stock = pd.read_csv("sp500/stock_dfs/{}.csv".format(ticker))
#     n = len(stock.iloc[:, 0])
#     X_train, y_train = prep_data(stock.iloc[5000:10000], future)
#     # date, X_test, y_test = prep_data(stock[10000:], future, True)
#     model.fit(X_train, y_train)
#     return model
#
#
# def test_data_linear(ticker, model, return_date=True, future=1):
#     stock = pd.read_csv("sp500/stock_dfs/{}.csv".format(ticker))
#     date, X_test, y_test = prep_data(stock[10000:], future, return_date=True)
#     y_pred = model.predict(X_test)
#     print(model.score(X_test, y_test))
#     predictions = pd.DataFrame().from_records(y_pred)
#     predictions.transpose()
#     predictions.columns = ["D{} Predicted".format(i) for i in range(1, future + 1)]
#     y_test = y_test.shift(periods=future).reset_index()
#     # y_test = y_test.reset_index()
#     y_test = y_test.join(predictions)
#     if return_date:
#         return y_test, date
#     return y_test
#
#
# def isolate_result(data, day, date, future=1):
#     results = pd.DataFrame()
#     results["Date"] = date
#     results = results.reset_index()
#     results = results.drop(columns=["index"])
#     results["D{} Actual".format(day)] = data["D{} Close".format(day)]
#     results["D{} Pred.".format(day)] = data["D{} Predicted".format(day)]
#     results.drop(results.head(future).index, inplace=True)
#     print(results.tail(20))
#     return results

df = pd.read_csv("sp500/stock_dfs/AAPL.csv")
emas = [i for i in range(5, 11)]
smas = [i for i in range(5, 11)]
df = prep_columns(df, future=1)

print(df.tail(30))

# X = df.iloc[:, :len(df.columns) - 2]
# y = df.iloc[:, len(df.columns) - 1]
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#
# models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()),
#           ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()),
#           ('SVM', SVC(gamma='auto'))]
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
#     results.append(cv_results)
#     names.append(name)
#     print("{}: Accuracy {}, Std. ({})".format(str(name), str(round(cv_results.mean(), 5)),
#                                               str(round(cv_results.std(), 5))))
