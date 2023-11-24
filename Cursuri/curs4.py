import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm

# Aruncarea monedelor in pymc
data = stats.bernoulli.rvs(p=0.35, size=4)
print(data)

#modelul
with pm.Model() as our_first_model:
    # a priori
    θ = pm.Beta('θ', alpha=1., beta=1.)
    # likelihood
    y = pm.Bernoulli('y', p=θ, observed=data)
    idata = pm.sample(1000, random_seed=123, return_inferencedata=True)

