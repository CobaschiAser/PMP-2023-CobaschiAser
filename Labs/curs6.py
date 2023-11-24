import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
N = 100
alpha_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)
x = np.random.normal(10, 1, N)
y_real = alpha_real + beta_real * x
y = y_real + eps_real

# folosim pymc pt a calibra modelul

with pm.Model() as model_g:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=1)
    eps = pm.HalfCauchy('eps', 5)
    miu = pm.Deterministic('miu', alpha + beta * x)
    y_pred = pm.Normal('y_pred', mu=miu, sigma=eps, observed=y)
    idata_g = pm.sample(100, tune=100, return_inferencedata=True, cores=1)

az.plot_trace(idata_g, var_names=["alpha", 'beta', 'eps'])
plt.show()
