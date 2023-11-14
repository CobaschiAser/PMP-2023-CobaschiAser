import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

# pct a)
raw_data = pd.read_csv("./auto-mpg.csv")
data = raw_data[['mpg', 'horsepower']]
data = data.drop(data[data.horsepower == '?'].index)
# print(data)
y = data['mpg'].values
x = data['horsepower'].values

y=np.array(y,dtype=np.float64)
x=np.array(x,dtype=np.float64)

plt.scatter(x, y)
plt.xlabel('horsepower')
plt.ylabel('mpg', rotation=0)
plt.show()

# pct b)

with pm.Model() as linear_model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=1)
    eps = pm.HalfCauchy('eps', 5)
    mu = pm.Deterministic('mu', alpha + beta * x)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y)
    idata_poly = pm.sample(500,return_inferencedata=True)

# az.plot_trace(idata_poly, var_names=["alpha", "beta", "eps"])
# plt.show()


# pct c)

plt.plot(x, y, 'C0.')
posterior_g = idata_poly.posterior.stack(samples={"chain", "draw"})
alpha_m = posterior_g['alpha'].mean().item()
beta_m = posterior_g['beta'].mean().item()
draws = range(0, posterior_g.samples.size, 10)
plt.plot(x, posterior_g['alpha'][draws].values + posterior_g['beta'][draws].values * x[:,None], c='gray', alpha=0.5)
plt.plot(x, alpha_m + beta_m * x, c='k',
label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()
plt.show()

# pct d)

w = pm.sample_posterior_predictive(idata_poly, model = linear_model)
plt.plot(x, y, 'b.')
plt.plot(x, alpha_m + beta_m * x, c='k',label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
az.plot_hdi(x, w.posterior_predictive['y_pred'], hdi_prob = 0.95, color = 'red')
#az.plot_hdi(x, w.posterior_predictive['y_pred'], color='red')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()
plt.show()

# Atunci cand creste puterea masinii (horsepower) observam ca eficienta combustibilului scade, deoarece va creste si consumul masinii. Deci cu acelasi combustibil, masina va merge mai putine mile.



