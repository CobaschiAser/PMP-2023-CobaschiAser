import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

SS = []
SB = []
BS = []
BB = []

for i in range(100):
    ss = 0
    sb = 0
    bs = 0
    bb = 0
    for j in range(10):
        a = np.random.random()
        b = np.random.random()
        if a < 0.3 and b < 0.5:
            ss = ss + 1
        elif a < 0.3 and b >= 0.5:
            sb = sb + 1
        elif a >= 0.3 and b < 0.5:
            bs = bs + 1
        elif a >= 0.3 and b >= 0.5:
            bb = bb + 1
    SS.append(ss)
    SB.append(sb)
    BS.append(bs)
    BB.append(bb)

az.plot_posterior({'SS': SS, 'SB': SB, 'BS': BS, 'BB': BB})
plt.show()
