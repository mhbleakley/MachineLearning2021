import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import pandas as pd

style.use("fivethirtyeight")

df = pd.DataFrame({"M": [0], "B": [0]})
df.to_csv("equation.csv")

xs = np.array([92, 56, 88, 70, 80, 49, 65, 35, 66, 67])
ys = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])


def animate(i):
    equation = pd.read_csv("equation.csv")
    m = equation.iloc[0][0]
    b = equation.iloc[0][1]
    x = max(xs)
    y = m * max(xs) + b
    plt.clf()
    plt.plot([0, x], [b, y])


def gradient_descent(x, y, iterations=1000, learning_rate=0.001, show=False):
    m = b = 0
    plt.scatter(x, y)
    y_pred = [0, 0]
    plt.tight_layout()
    plt.show()
    for i in range(iterations):
        y_pred = m * x + b
        cost = (1/len(x)) * sum([val**2 for val in (y - y_pred)])
        dm = -(2/len(x)) * sum(x*(y - y_pred))
        db = -(2 / len(x)) * sum(y - y_pred)
        m = m - learning_rate * dm
        b = b - learning_rate * db
        ndf = pd.DataFrame({"M": [m], "B": [b]})
        ndf.to_csv("equation.csv")
        print("m {} b {} cost {} iteration {}".format(m, b, cost, i))
        ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)


gradient_descent(xs, ys, iterations=10000, learning_rate=0.00001)


def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, list(points), learning_rate)
    return [b, m]


def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_m = 0  # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
