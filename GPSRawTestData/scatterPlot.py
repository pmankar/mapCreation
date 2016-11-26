import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('20160505_gpsOnly.csv', delimiter=',', skip_header=1)
x = data[:,2]
y = data[:,1]

plt.scatter(x, y, marker='x', color = '#000000')
plt.show()