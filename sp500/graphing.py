# import matplotlib.pyplot as plt
# import pandas as pd
# import yfinance as yf
# from matplotlib import style
# import matplotlib.animation as animation
# from sklearn import linear_model
# from SP500_toolbox import *
# import random
#
# style.use("dark_background")
#
# fig, axs = plt.subplots(1, 1)
#
#
# def random_data_generator(rows=2, cols=10, r=(0, 100)):
#     data = []
#     for row in range(rows):
#         line = []
#         for col in range(cols):
#             line.append(random.randint(r[0], r[1]))
#         data.append(line)
#     return data
#
#
# def fractal(z, c):
#
#
#
# def animate_line(i):
#     x, y = random_data_generator(cols=50)
#     axs.clear()
#     # plt.grid(color="#333", alpha=.5)
#     axs.scatter(x, y, marker=".", color="green")
#     # axs.minorticks_off()
#     # axs.tight_layout()
#
#
# ani = animation.FuncAnimation(fig, animate_line, interval=500)
# plt.show()
