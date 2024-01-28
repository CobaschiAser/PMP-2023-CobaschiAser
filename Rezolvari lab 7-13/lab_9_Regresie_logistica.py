import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

# O idee despre date
admissions = pd.read_csv("Admission.csv")

y = admissions["Admission"]
print(len(y[y==1]),len(y[y==0])) #date nebalansate
Index = np.random.choice(np.flatnonzero(y==0), size=len(y[y==0])-len(y[y==1]), replace=False) #pentru a balansa datele, alegem la intamplare indici pentru a fi stersi
admissions = admissions.drop(labels=Index)
y = admissions["Admission"]
x_GRE = admissions["GRE"].values
x_GPA = admissions["GPA"].values
x_GRE_mean = x_GRE.mean()
x_GRE_std = x_GRE.std()
x_GPA_mean = x_GPA.mean()
x_GPA_std = x_GPA.std()
#standardizam datele:
x_GRE = (x_GRE-x_GRE_mean)/x_GRE_std
x_GPA = (x_GPA-x_GPA_mean)/x_GPA_std
X = np.column_stack((x_GRE,x_GPA))

# 1
# Folosind distribuţii a priori slab informative asupra parametrilor β0, β1 şi β2,
# folosiţi PyMC pentru a simula un eşantion suficient de mare
# (construi modelul) din distribuţia a posteriori.

with pm.Model() as adm_model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=1, shape = 2)
    X_shared = pm.MutableData('x_shared',X) #pentru pct. 3 si 4
    mu = pm.Deterministic('μ',alpha + pm.math.dot(X_shared, beta))
    theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
    bd = pm.Deterministic("bd", -alpha/beta[1] - beta[0]/beta[1] * x_GRE)
    y_pred = pm.Bernoulli("y_pred", p=theta, observed=y)
    idata = pm.sample(2000, return_inferencedata=True)

# 2
# Care este, în medie, graniţa de decizie pentru acest model?
# Reprezentaţi de asemenea grafic o zonă în jurul acestei grafic
# care să reprezinte un interval 94% HDI.

idx = np.argsort(x_GRE)
bd = idata.posterior["bd"].mean(("chain", "draw"))[idx]
plt.scatter(x_GRE, x_GPA, c=[f"C{x}" for x in y])
plt.xlabel("GRE")
plt.ylabel("GPA")

idx = np.argsort(x_GRE)
bd = idata.posterior["bd"].mean(("chain", "draw"))[idx]
plt.scatter(x_GRE, x_GPA, c=[f"C{x}" for x in y])
plt.plot(x_GRE[idx], bd, color = 'k')
az.plot_hdi(x_GRE, idata.posterior["bd"], color ='k')
plt.xlabel("GRE")
plt.ylabel("GPA")

# 3

# Să presupunem că un student are un scor GRE de 550 şi un GPA de 3.5.
# Construiţi un interval de 90% HDI pentru probabilitatea ca acest student să fie admis.

obs_std1 = [(550-x_GRE_mean)/x_GRE_std,(3.5-x_GPA_mean)/x_GPA_std]
sigmoid = lambda x: 1 / (1 + np.exp(-x))
posterior_g = idata.posterior.stack(samples={"chain", "draw"})
mu = posterior_g['alpha'] + posterior_g['beta'][0]*obs_std1[0] + posterior_g['beta'][1]*obs_std1[1]
theta = sigmoid(mu)
az.plot_posterior(theta.values, hdi_prob=0.9)

# 4

# Dar dacă studentul are un scor GRE de 500 şi un GPA de 3.2?
# (refaceţi exerciţiul anterior cu aceste date)
# Cum justificaţi diferenţa?

obs_std1 = [(500-x_GRE_mean)/x_GRE_std,(3.2-x_GPA_mean)/x_GPA_std]
sigmoid = lambda x: 1 / (1 + np.exp(-x))
posterior_g = idata.posterior.stack(samples={"chain", "draw"})
mu = posterior_g['alpha'] + posterior_g['beta'][0]*obs_std1[0] + posterior_g['beta'][1]*obs_std1[1]
theta = sigmoid(mu)
az.plot_posterior(theta.values, hdi_prob=0.9)

# Observam ca obs_1 este mai apropiat de decizie decat obs_2 ceea ce indica gradul de incertitudine mai mic asociat

