import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# linear regression model with square cost function and gradiant descent
# https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction

# define functions
def cost_function(slope, intercept, size, x_points, y_points):
    # function of slope
    total_cost = 0
    for i in range(size):
        f_wb = slope * x_points[i] + intercept
        y_i = y_points[i]
        total_cost += ((f_wb - y_i) ** 2) / (2 * size)


# def gradient_descent(l_rate, x_points, y_points, slope, intercept, size):
#     # batch gradient descent
#     for i in range(size):
#         x_i = x_points[i]
#         y_i = y_points[i]
#         f_wb = slope * x_points[i] + intercept


# read file data

data = pd.read_csv('Housing.csv')

areas = np.array(data['area'])
prices = np.array(data['price'])

plt.plot(areas, prices, 'o')
# plt.xlim(0, 12000)
plt.show()
