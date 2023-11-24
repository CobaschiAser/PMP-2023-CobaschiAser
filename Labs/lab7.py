import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import csv

#A)

# preprocesarea datelor
# citire cu pandas
total_data = pd.read_csv("auto-mpg.csv")
# pastram doar coloanele 'mpg' si 'horsepower'
data = total_data[['mpg', 'horsepower']]
# eliminam liniile unde horsepower == ?
data = data.drop(data[data.horsepower == '?'].index)
# print(data)

# ni se spune ca CP este independenta( axa OX), iar MPG este dependenta( axa OY)

y = data['mpg'].values
x = data['horsepower'].values

y=np.array(y,dtype=np.float64)
x=np.array(x,dtype=np.float64)

plt.scatter(x, y)
plt.xlabel("CP")
plt.ylabel("MPG")
plt.show()

#B)

# calibram modelul

with pm.Model() as cars_model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=1)
    eps = pm.HalfCauchy('eps', 5)
    miu = pm.Deterministic('miu', alpha + beta * x)
    y_pred = pm.Normal('y_pred', mu=miu, sigma=eps, observed=y)
    idata_poly = pm.sample(500, return_inferencedata=True)

#az.plot_trace(idata_poly, var_names=['alpha', 'beta', 'eps'])
#plt.show()

# C) Dreapta care se potriveste cel mai bine cu regresia liniara

plt.plot(x, y, 'C0.')
posterior_g = idata_poly.posterior.stack(samples={"chain", "draw"})
alpha_m = posterior_g['alpha'].mean().item()
beta_m = posterior_g['beta'].mean().item()
draws = range(0, posterior_g.samples.size, 10)
plt.plot(x, posterior_g['alpha'][draws].values + posterior_g['beta'][draws].values * x[:,None], c='gray', alpha=0.5)
plt.plot(x, alpha_m + beta_m * x, c='k',label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()
plt.show()


# D)
ppc = pm.sample_posterior_predictive(idata_poly, model=cars_model)
plt.plot(x, y, 'b.')
plt.plot(x, alpha_m + beta_m * x, c='k', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
az.plot_hdi(x, ppc.posterior_predictive['y_pred'], hdi_prob=0.95, color='gray'), #smooth=False)
az.plot_hdi(x, ppc.posterior_predictive['y_pred'], color='gray') #smooth=False)
plt.xlabel('x')
plt.ylabel('y', rotation=0)