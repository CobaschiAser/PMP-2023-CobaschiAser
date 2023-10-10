import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

# np.random.seed(1)

server1 = stats.gamma.rvs(4, scale=1/3, size = 10000)
server2 = stats.gamma.rvs(4, scale=1/2, size = 10000)
server3 = stats.gamma.rvs(5, scale=1/2, size = 10000)
server4 = stats.gamma.rvs(5, scale=1/3, size = 10000)
lat = stats.expon.rvs(scale=1/4, size = 10000)

y = 0.25 * server1 + 0.25 * server2 + 0.3 * server3 + 0.2 * server4 + lat

az.plot_posterior({'S1': server1, 'S2': server2, 'S3': server3, 'S4': server4, 'y':y})
plt.show()
