import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# Ex 1

# Generaţi 500 de date dintr-o mixtură de trei distribuţii Gaussiene.

clusters = 3  # avem un cluster cu 3 modele in mixt
n_cluster = [250, 150, 100]  # cate date aduce fiecare model
n_total = sum(n_cluster)  # cate date avem in total = 500
means = [-2, 2, 5]  # media celor 3 modele
std_devs = [1, 1, 1]  # dev std pt fiecare model
mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster)) # mixtul
az.plot_kde(np.array(mix))

# Ex 2

# Calibraţi pe acest set de date un model de mixtură de distribuţii Gaussiene cu 2, 3, respectiv 4 componente

clusters = [2, 3, 4] # avem 3 clustere cu cate 2, 3, 4 componente
models = []  # aici adaugam cele 3 modele
idatas = []  # aici adaugam cele 3 idata generate
for cluster in clusters:
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.ones(cluster))
        means = pm.Normal('means', mu=np.linspace(mix.min(), mix.max(), cluster),sigma=10, shape=cluster, transform=pm.distributions.transforms.ordered)
        sd = pm.HalfNormal('sd', sigma=10)
        y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=mix)
        idata = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)
    idatas.append(idata)
    models.append(model)

# 3
# Comparaţi cele 3 modele folosind metodele WAIC şi LOO. Care este concluzia?
[pm.compute_log_likelihood(idatas[i], model=models[i]) for i in range(3)]
# important sa nu uit asta calculează log-likelihood (probabilitatea logaritmică) pentru un set de date și un model specific.


# calc waic si compararea
cmp_waic = az.compare({'model_2':idatas[0], 'model_3':idatas[1], 'model_4':idatas[2]},method='BB-pseudo-BMA', ic="waic", scale="deviance")
print(cmp_waic)
az.plot_compare(cmp_waic)

# calc loo si compararea
cmp_loo = az.compare({'model_2': idatas[0], 'model_3': idatas[1], 'model_4': idatas[2]},method='BB-pseudo-BMA', ic="loo", scale="deviance")
print(cmp_loo)
az.plot_compare(cmp_loo)

# Var 2:

comp_waic = az.compare(dict(zip([str(c) for c in clusters], idatas)),method='BB-pseudo-BMA', ic="waic", scale="deviance")
print(comp_waic)
az.plot_compare(comp_waic)


cmp_loo = az.compare(dict(zip([str(c) for c in clusters], idatas)),method='BB-pseudo-BMA', ic="loo", scale="deviance")
print(cmp_loo)
az.plot_compare(cmp_loo)
