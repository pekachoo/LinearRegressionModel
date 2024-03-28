import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(X, y, w, b):
    # size of data set X, containing multiple features
    m, n = X.shape
    z = 0
    for i in range(m):
        x = X[i]
        z += np.dot(x, w) + b
    f_wbx = sigmoid(z)
    total_cost = (-1 / m) * np.sum(y * np.log(f_wbx) + (1 - y) * np.log(1 - f_wbx))
    return total_cost


def calculate_derivatives(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wbx = sigmoid(z)
    dw = 0
    for i in range(m):
        dw += (f_wbx[i] - y[i]) * X[i]
    db = (1 / m) * np.sum(h - y)
    return dw, db


def gradient_descent(X, y, w_i, b_i, l_rate, num_iterations):
    m, n = X.shape
    cost_arr = [cost_function(X, y, w_i, b_i)]
    slope_arr = [w_i]
    intercept_arr = [b_i]
    iteration_arr = [0]

    w = w_i
    b = b_i
    for i in range(m):
        dJ_dw, dJ_db = calculate_derivatives(X, y, w, b)


data = pd.read_csv('bmd.csv')

bmd_list = data['bmd']
fracture_list = data['fracture']

# plot data circle for true, x for false

plt.scatter(bmd_list, fracture_list, color='red')
plt.show()
