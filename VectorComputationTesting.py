import numpy as np

# arr1 = np.array([1, 2, 3, 4, 5])
# arr2 = np.array([6, 7, 8, 9, 10])
#
# print(arr1 - arr2)
# print(arr1**2)
#

# Set K
X = np.array([[1.84207953, 4.6075716]
                 , [5.65858312, 4.79996405]
                 , [6.35257892, 3.2908545]
                 , [2.90401653, 4.61220411]
                 , [3.23197916, 4.93989405]])
centroids = np.array([[3, 3], [6, 2], [8, 5]])

K = centroids.shape[0]

# You need to return the following variables correctly
idx = np.zeros(X.shape[0], dtype=int)

for i in range(X.shape[0]):
    # go through every input

    # compute distance to each centroid
    centroid_num = 0
    min_dist = np.sum((X[i] - centroids[0])**2)
    print(i, min_dist)
    for e in range(K - 1):
        curr_dist = np.sum((X[i] - centroids[e + 1])**2)
        print(i, curr_dist)
        if curr_dist < min_dist:
            centroid_num = e + 1
            min_dist = curr_dist
    idx[i] = centroid_num

print(idx)
