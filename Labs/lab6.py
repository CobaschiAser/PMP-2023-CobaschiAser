import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

traces = []
for y in y_values:
    for theta in theta_values:
        with pm.Model() as model:
            # distributia a priori pt n este Poisson(10)
            n = pm.Poisson('n', 10)
            # distributia a posteriori pt n este Binomial(n,theta, observed=y)
            n_posterior = pm.Binomial('n_posterior', n=n, p=theta, observed=y)
            # cream trace-ul
            trace = pm.sample(100, return_inferencedata=True, cores=1)
            traces.append(trace)

# plot comun pt toate trace-urile

fig, axes = plt.subplots(len(y_values), len(theta_values), figsize=(12, 8))
plt.subplots_adjust(hspace=0.5, wspace=1.0)

for i in range(len(y_values)):
    for j in range(len(theta_values)):
        az.plot_posterior(traces[i*len(theta_values)+j], var_names=['n'], ax=axes[i, j])
        axes[i, j].set_title(f'Y = {y_values[i]}, theta = {theta_values[j]}')

plt.tight_layout()
plt.show()
