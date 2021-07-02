import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np

style.use("dark_background")

x1 = np.array([92, 56, 88, 70, 80, 49, 65, 35, 66, 67])
y1 = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])
m = [0.0]
b = [0.0]
iteration = [0]


def update_line(x, y):
    m0 = m[0]
    b0 = b[0]
    m.pop()
    b.pop()
    learning_rate = 0.00001
    n = len(x)
    y_pred = m0 * x + b0  # x1 and y1 must be np arrays
    dm = -(2 / n) * sum(x * (y - y_pred))
    db = -(2 / n) * sum(y - y_pred)
    m.append(float(m0 - learning_rate * dm))
    b.append(float(b0 - learning_rate * db))
    data = open("mb.txt", "w")
    data.write(str(m[0]) + "," + str(b[0]))
    data.close()


def animate_line(i, x, y, it):
    if iteration[0] < it:
        update_line(x, y)
        iteration.append(iteration[0] + 1)
        iteration.pop(0)
        data = open("mb.txt", "r")
        m_curr = b_curr = 0
        for line in data:
            if len(line) > 0:
                m_curr = float(line.split(",")[0])
                b_curr = float(line.split(",")[1])

        y_pred = (m_curr * max(x)) + b_curr
        xs = [0, max(x) + (0.1 * max(x))]
        ys = [b_curr, y_pred]
        # print(m_curr)
        # print(b_curr)
        # print(y)
        # print()
        plt.clf()
        plt.annotate("Iteration #" + str(iteration[0]), xy=(0.1 * min(x), 0.9 * max(y)))
        plt.annotate(("m = " + str(round(m_curr, 4))), xy=(0.1 * min(x), 0.8 * max(y)))
        plt.annotate(("b = " + str(round(b_curr, 4))), xy=(0.1 * min(x), 0.7 * max(y)))
        plt.grid(color="#333", alpha=.5)
        plt.plot(xs, ys, "--", color="#1b1")
        plt.scatter(x, y, marker="o", linewidths=1, color="#42f2f5")
        plt.tight_layout()


ani = animation.FuncAnimation(plt.gcf(), animate_line, fargs=[x1, y1, 200], interval=20)
plt.show()
