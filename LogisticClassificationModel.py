import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('bmd.csv')

bmd_list = data['bmd']
fracture_list = data['fracture']

# plot data circle for true, x for false

plt.scatter(bmd_list, fracture_list, color='red')
plt.show()