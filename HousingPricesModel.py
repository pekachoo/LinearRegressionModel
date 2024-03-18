import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# linear regression model with square cost function and gradiant descent
# https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction

# define functions
def cost_function(cost_slope, cost_intercept, x_points, y_points):
    # function of slope
    total_cost = 0
    for e in range(x_points.shape[0]):
        f_wb = cost_slope * x_points[e] + cost_intercept
        y_i = y_points[e]
        total_cost += ((f_wb - y_i) ** 2)
    return total_cost / (2 * x_points.shape[0])


def calculate_derivatives(x_points, y_points, derivative_slope, derivative_intercept):
    # batch gradient descent
    size = x_points.shape[0]
    dJ_dw = 0
    dJ_db = 0

    for h in range(size):
        x_i = x_points[h]
        y_i = y_points[h]
        f_wb = derivative_slope * x_i + derivative_intercept
        temp_dJ_dw = (f_wb - y_i) * x_i
        temp_dJ_db = (f_wb - y_i)
        dJ_dw += temp_dJ_dw
        dJ_db += temp_dJ_db
    dJ_dw = dJ_dw / size
    dJ_db = dJ_db / size
    return dJ_dw, dJ_db


def gradiant_descent(x_points, y_points, slope_i, intercept_i, l_rate, num_iterations):
    cost_arr = [cost_function(slope_i, intercept_i, x_points, y_points)]
    slope_arr = [slope_i]
    intercept_arr = [intercept_i]
    iteration_arr = [0]

    current_slope = slope_i
    current_intercept = intercept_i

    for j in range(num_iterations):
        dJ_dw, dJ_db = calculate_derivatives(x_points, y_points, current_slope, current_intercept)
        current_slope = current_slope - l_rate * dJ_dw
        current_intercept = current_intercept - l_rate * dJ_db

        cost_arr.append(cost_function(current_slope, current_intercept, x_points, y_points))
        slope_arr.append(current_slope)
        intercept_arr.append(current_intercept)
        iteration_arr.append(j + 1)
    return cost_arr, slope_arr, current_slope, current_intercept, iteration_arr


# read file data

data = pd.read_csv('Housing.csv')

areas = np.array(data['area'])
prices = np.array(data['price'])

# for i in range(prices.shape[0]):
#     prices[i] = prices[i]/1000

# print(areas)
# print(prices)
costs, slopes, slope, intercept, iterations = gradiant_descent(areas, prices, 0, 0, 1.0e-9, 200)
costs = np.array(costs)
slopes = np.array(slopes)
# plt.plot(slopes, costs, 'o')
plt.plot(slopes, costs)
# for i in range(slopes.shape[0]):
#     plt.plot()

# # plt.xlim(0, 12000)
plt.show()
