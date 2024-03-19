import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Housing.csv')

prices = np.array(data['price'])
areas = np.array(data['area'])
stories = np.array(data['stories'])
bathrooms = np.array(data['bathrooms'])
bedrooms = np.array(data['bedrooms'])
# converts it into a 2d array with each index being a training set
features = np.column_stack((areas, stories, bathrooms, bedrooms))
print(features)


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
    dj_dw = 0
    dj_db = 0
    for i in range(size):
        x_i = x[i] # array of x's in each feature i
        y_i = y[i] # corresponding price
        f_wb = np.dot(w, x_i) + intercept
        dj_dw += (f_wb - y_i)*x_i
        dj_db += f_wb - y_i
    dj_dw = dj_dw/size
    dj_db = dj_db/size
    return dj_dw, dj_db
