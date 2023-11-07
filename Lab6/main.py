import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

traces = []

for y in y_values:
    for theta in theta_values:
        with pm.Model() as model:
            n = pm.Poisson('n', mu=10)
            n_posterior = pm.Binomial('n_posterior', n=n, p=theta, observed=y)
            trace = pm.sample(100, return_inferencedata=True, cores=1)
            traces.append(trace)

fig, axes = plt.subplots(len(y_values), len(theta_values), figsize=(12, 8))
plt.subplots_adjust(hspace=0.5, wspace=1.0)

for i in range(len(y_values)):
    for j in range(len(theta_values)):
        az.plot_posterior(traces[i * len(theta_values)+j], var_names=['n'], ax=axes[i, j])
        axes[i, j].set_title(f'Y = {y_values[i]}, t = {theta_values[j]}')
plt.tight_layout()
plt.show()
