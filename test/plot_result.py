import matplotlib.pyplot as plt

import numpy as np

fps = 20
result = np.load('output.npy')
total = len(result)
t = np.linspace(0, total / 20, total)
plt.plot(t, result)
plt.show()
