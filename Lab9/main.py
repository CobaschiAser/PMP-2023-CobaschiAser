import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd

# a)

data = pd.read_csv("Admission.csv")
admission = data['Admission'].values
x_n = ['GPA', 'GRE']
x_1 = data[x_n].values

with pm.Model() as model:
  alpha = pm.Normal('alpha', mu=0, sigma=10)
  beta = pm.Normal('beta', mu=0, sigma=2, shape=len(x_n))
  mu = alpha + pm.math.dot(x_1, beta)
  teta = pm.Deterministic('teta', 1 / (1 + pm.math.exp(-mu)))
  bd = pm.Deterministic('bd', -alpha/beta[1] - beta[0]/beta[1] * x_1[:,0])
  yl = pm.Bernoulli('yl', p=teta, observed=admission)
  idata_1 = pm.sample(2000, return_inferencedata=True)

# az.plot_posterior(idata_1, var_names=['beta'])

# b)

idx = np.argsort(x_1[:,0])
bd = idata_1.posterior['bd'].mean(("chain", "draw"))[idx]
plt.scatter(x_1[:,0], x_1[:,1], c=[f'C{x}' for x in admission])
plt.plot(x_1[:,0][idx], bd, color='k');
az.plot_hdi(x_1[:,0], idata_1.posterior['bd'], color='k', hdi_prob=0.94)
plt.xlabel(x_n[0])
plt.ylabel(x_n[1])

plt.show()