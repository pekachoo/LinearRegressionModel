import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def cost_function(x, y, w, intercept):
    # for one specific feature
    size = x.shape[0]
    total_cost = 0
    for i in range(size):
        y_i = y[0]
        x_i = x[i]
        f_wb = w * x_i + intercept
        total_cost += (f_wb - y_i) ** 2
    total_cost = total_cost / (2 * size)
    return total_cost


def predict_fwb(x, w, b):
    # takes the current slope and makes a prediction prices based off of it
    return np.dot(x, w) + b


def cost_function_multiple(x, y, w, intercept):
    # with multiple w, f_wb = w_0*x_0 + w_1*x_1 ... w_n-1*x_n-1 + b--> dot product the parameter 'w' in this function
    # is actually a vector containing w_0, w_1, w_2, etc. each w represents a slope for each feature for a given
    # output price the parameter 'x' contains the 2d array of the training samples
    size = x.shape[0]
    total_cost = 0
    for i in range(size):
        x_i = x[i]  # individual training set
        y_i = y[i]
        f_wb = np.dot(w, x_i) + intercept
        total_cost += (f_wb - y_i) ** 2
    total_cost = total_cost / (2 * size)
    return total_cost


def calculate_derivatives_multiple(x, y, w, b):
    size = x.shape[0]
    num_features = x.shape[1]
    dj_dw = np.zeros((size,))  # dj_dw is now an array with the derivatives of each w
    dj_db = 0
    for i in range(size):
        x_i = x[i]  # array of x's in each feature i
        y_i = y[i]  # corresponding price
        f_wb = np.dot(w, x_i) + b
        for j in range(num_features):
            dj_dw[i][j] += (f_wb - y_i) * x_i[j]
        dj_db += f_wb - y_i
    dj_dw = dj_dw / size
    dj_db = dj_db / size
    return dj_dw, dj_db


def gradient_descent_multiple(x, y, w_i, b_i, l_rate, num_iterations):
    cost_arr = [cost_function(x, y, w_i, b_i)]
    slope_arr = [w_i]
    intercept_arr = [b_i]
    iteration_arr = [0]

    size = x.shape[0]
    num_features = x.shape[1]

    w = w_i
    b = b_i
    for i in range(num_iterations):
        dJ_dw, dJ_db = calculate_derivatives_multiple(x, y, w, b)
        b = b - l_rate * dJ_db
        for j in range(num_features):
            w[j] = w[j] - l_rate * dJ_dw[j]
        cost_arr.append(cost_function(x, y, w, b))
        slope_arr.append(w)
        intercept_arr.append(b)
        iteration_arr.append(i + 1)

    return cost_arr, slope_arr, intercept_arr, iteration_arr, w, b


data = pd.read_csv('Housing.csv')

prices = np.array(data['price'])
areas = np.array(data['area'])
stories = np.array(data['stories'])
bathrooms = np.array(data['bathrooms'])
bedrooms = np.array(data['bedrooms'])
# converts it into a 2d array with each index being a training set
features = np.column_stack((areas, stories, bathrooms, bedrooms))

costs, slopes, intercepts, iterations, w, b = gradient_descent_multiple(features, prices, np.zeros(len(features[0])))
