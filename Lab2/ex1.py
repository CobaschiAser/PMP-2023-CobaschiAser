import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

# np.random.seed(1)

x = stats.expon.rvs(scale=1/4, size=10000)
y = stats.expon.rvs(scale=1/6, size=10000)
z = 0.4*x + 0.6*y

az.plot_posterior({'x': x, 'y': y, 'z':z})
plt.show()
