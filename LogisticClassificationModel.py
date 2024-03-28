import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(X, y, w, b):
    # size of data set X, containing multiple features
    m, n = X.shape
    z = 0
    total_cost = 0
    for i in range(m):
        x_i = X[i]
        y_i = y[i]
        z_wb = np.dot(x_i, w) + b
        f_wb = sigmoid(z_wb)
        loss_value = (-y_i * np.log(f_wb)) - (1 - y_i) * np.log(1 - f_wb)
        total_cost += loss_value
    return total_cost


def calculate_derivatives(X, y, w, b):
    m, n = X.shape
    dw = np.zeros(n)
    db = 0
    for i in range(m):
        x = X[i]
        z = np.dot(x, w) + b
        f_wbx = sigmoid(z)
        for j in range(n):
            dw[j] += (f_wbx - y[i]) * x[j]
        db += f_wbx - y[i]
    dw = dw / m
    db = db / m
    return dw, db


def gradient_descent(X, y, w_i, b_i, l_rate, num_iterations):
    m, n = X.shape
    cost_arr = [cost_function(X, y, w_i, b_i)]
    slope_arr = [w_i]
    intercept_arr = [b_i]
    iteration_arr = [0]

    w = w_i
    b = b_i
    for i in range(num_iterations):
        dJ_dw, dJ_db = calculate_derivatives(X, y, w, b)
        b = b - l_rate * dJ_db
        for j in range(n):
            w[j] = w[j] - l_rate * dJ_dw[j]
        cost_arr.append(cost_function(X, y, w, b))
        slope_arr.append(w)
        intercept_arr.append(b)
        iteration_arr.append(i + 1)
    return cost_arr, slope_arr, intercept_arr, iteration_arr, w, b


data = pd.read_csv('bmd.csv')

bmd_list = np.array(data['bmd'])
fracture_list = np.array(data['fracture'])
# convert fracture list where every "fracture" becomes a 1 and every nonfracture becomes a 0
# reshapes dimension to 1 x n matrix
fracture_list = np.array([1 if fracture == 'fracture' else 0 for fracture in fracture_list]).reshape(len(fracture_list), 1)
bmd_list = bmd_list.reshape(len(bmd_list), 1)
# print(bmd_list)
# print(fracture_list)
# print(cost_function(bmd_list, fracture_list, np.zeros(len(bmd_list[0])), 0))
# plot data circle for true, x for false
costs, slopes, intercepts, iterations, final_w, final_b = gradient_descent(bmd_list, fracture_list, np.zeros(len(bmd_list[0])), 0, 0.005, 3000)
#
print(costs)
plt.scatter(iterations, costs, color='red')
plt.show()
# print(slopes)
# print(intercepts)
# plt.scatter(iterations, slopes, color='blue')
# plt.show()
# # Create a range of x values
x_values = np.linspace(min(bmd_list), max(bmd_list), 100)

# Calculate the corresponding y values
y_values = sigmoid(x_values * final_w + final_b)

# Plot the classification function
plt.plot(x_values, y_values, label='Classification Function')

# Plot the original data
plt.scatter(bmd_list, fracture_list, color='red', label='Original Data')

# Add labels and a legend
plt.xlabel('BMD')
plt.ylabel('Fracture')
plt.legend()

# Show the plot
plt.show()