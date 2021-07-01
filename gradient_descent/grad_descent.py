import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import pandas as pd

style.use("fivethirtyeight")

x1 = np.array([92, 56, 88, 70, 80, 49, 65, 35, 66, 67])
y1 = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])
m = [0.0]
b = [0.0]
iteration = [0]


def update_line(it):
    m0 = m[0]
    b0 = b[0]
    m.pop()
    b.pop()
    learning_rate = 0.00001
    n = len(x1)
    y_pred = m0 * x1 + b0
    dm = -(2 / n) * sum(x1 * (y1 - y_pred))
    db = -(2 / n) * sum(y1 - y_pred)
    m.append(float(m0 - learning_rate * dm))
    b.append(float(b0 - learning_rate * db))
    data = open("mb.txt", "w")
    data.write(str(m[0]) + "," + str(b[0]))
    data.close()


def animate(i):
    update_line(iteration[0])
    iteration.append(iteration[0] + 1)
    iteration.pop(0)
    data = open("mb.txt", "r")
    m_curr = b_curr = 0
    for line in data:
        if len(line) > 0:
            m_curr = float(line.split(",")[0])
            b_curr = float(line.split(",")[1])

    y = (m_curr * 92) + b_curr
    xs = [0, max(x1) + (0.1 * max(x1))]
    ys = [b_curr, y]
    # print(m_curr)
    # print(b_curr)
    # print(y)
    # print()
    plt.clf()
    plt.annotate("Iteration #" + str(iteration[0]), xy=(0.1 * min(x1), 0.9 * max(y1)))
    plt.annotate(("m = " + str(round(m_curr, 4))), xy=(0.1 * min(x1), 0.8 * max(y1)))
    plt.annotate(("b = " + str(round(b_curr, 4))), xy=(0.1 * min(x1), 0.7 * max(y1)))
    plt.plot(xs, ys)
    plt.scatter(x1, y1)
    plt.tight_layout()


ani = animation.FuncAnimation(plt.gcf(), animate, interval=(1/(iteration[0] + 1)))
plt.show()
