import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


style.use("ggplot")

x1 = [92, 56, 88, 70, 80, 49, 65, 35, 66, 67]
y1 = [98, 68, 81, 80, 83, 52, 66, 30, 68, 73]
m = [0.0]
b = [0.0]
iteration = [0]


def update_line(it):
    m.pop()
    b.pop()
    m.append(float(it) * .1)
    b.append(float(it * .01))
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
    xs = [0, 100]
    ys = [b_curr, y]
    print(m_curr)
    print(b_curr)
    print(y)
    print()
    plt.clf()
    plt.plot(xs, ys)
    plt.scatter(x1, y1)
    plt.tight_layout()


ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)
plt.show()
