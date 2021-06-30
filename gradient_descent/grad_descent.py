import numpy as np


def gradient_descent(x, y, iterations=1000, learning_rate=0.001):
    m = b = 0
    for i in range(iterations):
        y_pred = m * x + b
        cost = (1/len(x)) * sum([val**2 for val in (y - y_pred)])
        dm = -(2/len(x)) * sum(x*(y - y_pred))
        db = -(2 / len(x)) * sum(y - y_pred)
        m = m - learning_rate * dm
        b = b - learning_rate * db
        print("m {} b {} cost {} iteration {}".format(m, b, cost, i))


xs = np.array([92, 56, 88, 70, 80, 49, 65, 35, 66, 67])
ys = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])

gradient_descent(xs, ys, iterations=10000, learning_rate=0.00001)
