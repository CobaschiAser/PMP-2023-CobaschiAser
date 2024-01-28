#  import

import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt

Prices = pd.read_csv('Prices.csv')
x_1 = Prices['Speed'].values
x_2 = np.log(Prices['HardDrive'].values)
y = Prices['Price'].values
X = np.column_stack((x_1, x_2))
X_mean = X.mean(axis=0, keepdims=True)

# pentru a ne face o idee asupra mediilor si dev. standard:

print(X_mean)
print(y.mean())
print(X.std(axis=0, keepdims=True))
print(y.std())


# si o idee despre date:
def scatter_plot(x, y):
    plt.figure(figsize=(15, 5))
    for idx, x_i in enumerate(x.T):
        plt.subplot(1, 3, idx + 1)
        plt.scatter(x_i, y)
        plt.xlabel(f'x_{idx + 1}')
        plt.ylabel(f'y', rotation=0)

    plt.subplot(1, 3, idx + 2)
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlabel(f'x_{idx}')
    plt.ylabel(f'x_{idx + 1}', rotation=0)


scatter_plot(X, y)

# A
# Folosind distribuţii a priori slab informative asupra parametrilor
# α, β1, β2 şi σ, folosiţi PyMC pentru a
# simula un eşantion suficient de mare din distribuţia a posteriori.

with pm.Model() as model_mlr:
    alpha = pm.Normal('alpha', mu=0, sigma=1000)
    # am luat sigma=1000 deoarece nu am standardizat datele, iar dev. standard pentru y este f. mare
    beta = pm.Normal('beta', mu=0, sigma=1000, shape=2)
    eps = pm.HalfCauchy('eps', 5000)
    nu = pm.Exponential('niu', 1 / 30)
    X_shared = pm.MutableData('x_shared', X)  # pentru pct. 5
    mu = pm.Deterministic('μ', alpha + pm.math.dot(X_shared, beta))

    y_pred = pm.StudentT('y_pred', mu=mu, sigma=eps, nu=nu, observed=y)

    idata_mlr = pm.sample(1250, return_inferencedata=True)

# 2
#  Obţineţi estimări de 95% HDI ale parametrilor β1 şi β2.

az.plot_forest(idata_mlr, hdi_prob=0.95, var_names=['β'])
az.summary(idata_mlr, hdi_prob=0.95, var_names=['β'])

# 3
# Pe baza rezultatelor obţinute, sunt frecvenţa procesorului
# şi mărimea hard diskului predictori utili ai preţului de vânzare?

# R: Se vede ca x1(frecventa) are o mica influenta asupra pretului(y),
# deoarece beta[0] este mic in comparatie cu beta[1]. Totusi, 0 nu este
# in intervalul HDI pentru beta[0] (si nici pentru beta[1]), deci putem
# lua acest parametru in calcul


# 4
# Să presupunem acum că un consumator este interesat de un computer
# cu o frecvenţă de 33 MHz şi un hard disk de 540 MB.
# Simulaţi 5000 de extrageri din preţul de vânzare aşteptat (μ)
# şi construiţi un interval de 90% HDI pentru acest preţ.

posterior_g = idata_mlr.posterior.stack(samples={"chain", "draw"}) #avem 5000 de extrageri in esantion (nr. draws x nr. chains)
mu = posterior_g['α']+33*posterior_g['β'][0]+np.log(540)*posterior_g['β'][1]
az.plot_posterior(mu.values,hdi_prob=0.9)


# 5
# În schimb, să presupunem că acest consumator doreşte să prezică preţul de vânzare
# al unui computer cu această frecvenţă şi mărime a hard disk-ului.
# Simulaţi 5000 de extrageri din distribuţia predictivă posterioară şi utilizaţi aceste extrageri
# simulate pentru a găsi un interval de predicţie de 90% HDI.

pm.set_data({"x_shared":[[33,np.log(540)]]}, model=model_mlr)
ppc = pm.sample_posterior_predictive(idata_mlr, model=model_mlr)
y_ppc = ppc.posterior_predictive['y_pred'].stack(sample=("chain", "draw")).values
az.plot_posterior(y_ppc,hdi_prob=0.9)