import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt


### EX 1
# a) Incarcarea datelor
BostonHousing = pd.read_csv('BostonHousing.csv')
medv = BostonHousing['medv'].values
rm = BostonHousing['rm'].values
crim = BostonHousing['crim'].values
indus = BostonHousing['indus'].values

# b)
# Ne folosim de variabilele independente rm, crim, indus pt a prezice medv

X = np.column_stack((rm, crim, indus))
Y = medv

# definim modelul in PYMC pentru a prezice Y, conform unei regresii liniare multiple de forma Y = alpha + X[0] * Beta[0] + X[1] * Beta[1] + X[2] * Beta[2]
with pm.Model() as model_mlr:
    alpha = pm.Normal('alpha', mu=0, sigma=1000)
    beta = pm.Normal('beta', mu=0, sigma=1000, shape=3)
    eps = pm.HalfCauchy('eps', 5000)
    nu = pm.Exponential('niu', 1 / 30)
    X_shared = pm.MutableData('x_shared', X)
    mu = pm.Deterministic('μ', alpha + pm.math.dot(X_shared, beta))

    y_pred = pm.StudentT('y_pred', mu=mu, sigma=eps, nu=nu, observed=Y)

    idata_mlr = pm.sample(1250, return_inferencedata=True)

az.plot_trace(idata_mlr, var_names=["alpha", "beta", "eps"])
plt.show()

# c)

# Obtinem estimari de 95% pentru HDI ale parametrilor (beta[0], beta[1], beta[2])
az.plot_forest(idata_mlr, hdi_prob=0.95, var_names=['beta'])
az.summary(idata_mlr, hdi_prob=0.95, var_names=['beta'])

# RASPUNS
# Variabila care influenteaza cel mai mult rezultatul este "rm" (X[0]) deoarece beta[0] este mult
# mai mare in comparatie cu beta[1] si cu beta[2]

# d)

# setam valori pentru x_shared conform cu mediile datelor din csv
pm.set_data({"x_shared":[[rm.mean(), crim.mean(), indus.mean()]]}, model=model_mlr)

# simulam extrageri din distributia predictiva posterioara
ppc = pm.sample_posterior_predictive(idata_mlr, model=model_mlr)

# selectam valoarea locuintelor
y_ppc = ppc.posterior_predictive['y_pred'].stack(sample=("chain", "draw")).values

# cautam intervalul de predictie de 50% HDI pentru valoarea locuintelor
az.plot_posterior(y_ppc,hdi_prob=0.5)

# asa cum se vede din grafic, intervalul ar fi [19-25] cu media 22.


### EX 2

# a)
def posterior_grid(grid_points=50, first_success = 5):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1 / grid_points, grid_points)  # uniform prior
    # likelihood este probabilitatea ca sa obtinem stema abia la a "fisrt_success" incercare
    likelihood = stats.geom.pmf(first_success, 0.5)
    print(likelihood)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

# cateva exemple ale faptului ca functioneaza bine
# print(posterior_grid(first_success = 1)[1]) # 0.5
# print(posterior_grid(first_success = 3)[1]) # 0.2

# b)


