import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(X, y, w, b):
    # size of data set X, containing multiple features
    m = X.shape[0]
    z = np.dot(X, w) + b
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


data = pd.read_csv('bmd.csv')

bmd_list = data['bmd']
fracture_list = data['fracture']

# plot data circle for true, x for false

plt.scatter(bmd_list, fracture_list, color='red')
plt.show()
