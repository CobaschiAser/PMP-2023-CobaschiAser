import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

# preprocesarea datelor
data_from_csv = pd.read_csv("Prices.csv")
price = data_from_csv['Price'].values
speed = data_from_csv['Speed'].values
hard_drive = data_from_csv['HardDrive'].values

price = np.array(price, dtype=np.float64)
speed = np.array(speed, dtype=np.float64)
hard_drive = np.array(hard_drive, dtype=np.float64)
hard_drive = np.log(hard_drive)

# a)
with pm.Model() as model_mlr:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta_1 = pm.Normal('beta_1', mu=0, sigma=1)
    beta_2 = pm.Normal('beta_2', mu=0, sigma=1)
    eps = pm.HalfCauchy('eps', 5)
    mu = pm.Deterministic('mu', alpha + beta_1 * speed + beta_2 * hard_drive)
    price_pred = pm.Normal('price_pred', mu=mu, sigma=eps, observed=price)
    idata_g = pm.sample(50, tune=50, cores=1, return_inferencedata=True)

"""az.plot_trace(idata_g, var_names=['alpha', 'beta_1', 'beta_2'])
plt.show()"""

# b)

az.plot_posterior(idata_g, var_names='beta_1', hdi_prob=0.95)
plt.show()


az.plot_posterior(idata_g, var_names='beta_2', hdi_prob=0.95)
plt.show()

