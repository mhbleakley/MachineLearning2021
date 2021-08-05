from matplotlib import pyplot as plt
from matplotlib import style
import pandas as pd

style.use("dark_background")

data = pd.read_csv("data.csv")

males = data.loc[data["Gender"] == "M"]
females = data.loc[data["Gender"] == "F"]

c = (150, 70)

# def knn(k, c, data):


plt.scatter(males["Weight"], males["Height"], marker="d", color="red", label="Males")
plt.scatter(females["Weight"], females["Height"], marker="d", color="green", label="Females")
plt.title("Male and Female Height Vs. Weight")
plt.ylabel("Height (in.)")
plt.xlabel("Weight (lbs.)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
