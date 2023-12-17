import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import scipy.stats as stats

# ex1

az.style.use('arviz-darkgrid')
dummy_data = np.loadtxt('./dummy.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = ((x_1p - x_1p.mean(axis=1, keepdims=True)) /
        x_1p.std(axis=1, keepdims=True))
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')

# a)
with pm.Model() as model_p:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

"""alpha_l_post = idata_l.posterior['alpha'].mean(("chain", "draw")).values
beta_l_post = idata_l.posterior['beta'].mean(("chain", "draw")).values
y_l_post = alpha_l_post + beta_l_post * x_new
plt.plot(x_new, y_l_post, 'C1', label='linear model')
"""
alpha_p_post = idata_p.posterior['alpha'].mean(("chain", "draw")).values
beta_p_post = idata_p.posterior['beta'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)
plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order} sd = 10')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()

# b)
with pm.Model() as model_p2:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=100, shape=order)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_1s)
    idata_p2 = pm.sample(2000, return_inferencedata=True)

alpha_p2_post = idata_p2.posterior['alpha'].mean(("chain", "draw")).values
beta_p2_post = idata_p2.posterior['beta'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p2_post = alpha_p2_post + np.dot(beta_p2_post, x_1s)
plt.plot(x_1s[0][idx], y_p2_post[idx], 'C2', label=f'model order {order} sd = 100')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()

with pm.Model() as model_p3:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_1s)
    idata_p3 = pm.sample(2000, return_inferencedata=True)

alpha_p3_post = idata_p3.posterior['alpha'].mean(("chain", "draw")).values
beta_p3_post = idata_p3.posterior['beta'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p3_post = alpha_p3_post + np.dot(beta_p3_post, x_1s)
plt.plot(x_1s[0][idx], y_p3_post[idx], 'C2', label=f'model order {order} sd = np.array([10, 0.1, 0.1, 0.1, 0.1])')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()


