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

"""
# c)

# Valorile studentului
new_data = np.array([3.5, 550])

# Calculul probabilității de admitere pentru studentul nou
with model:
    mu_new = idata_1.posterior['alpha'].mean() + np.dot(new_data, idata_1.posterior['beta'].mean(axis=(0, 1)))
    prob_admission = 1 / (1 + np.exp(-mu_new))
    hdi_90 = az.hdi(prob_admission, hdi_prob=0.9)

# Plotarea intervalului de 90% HDI pentru probabilitatea de admitere
#plt.figure(figsize=(8, 6))
plt.scatter(x_1[:, 0], x_1[:, 1], c=[f'C{x}' for x in admission])
plt.plot(x_1[:, 0][idx], bd, color='k')
plt.scatter(new_data[0], new_data[1], marker='x', color='red', s=100, label='Student nou (3.5, 550)')
plt.xlabel(x_n[0])
plt.ylabel(x_n[1])
plt.legend()

# Adăugarea intervalului HDI pe grafic
plt.fill_betweenx([plt.ylim()[0], plt.ylim()[1]], new_data[0], hdi_90, color='blue', alpha=0.3, label='Interval HDI 90%')
plt.legend()
plt.show()


"""