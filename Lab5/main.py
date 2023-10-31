import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy import stats
import csv
import pandas as pd
import arviz as az

f = open("trafic.csv", "r")
real_data = csv.reader(f)
rows = []
header = next(real_data)
for row in real_data:
    rows.append(int(row[1]))
print(rows)

traffic_data = np.array(rows)

with pm.Model() as traffic_model:
    alpha = 1.0 / traffic_data.mean()

    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    lambda_3 = pm.Exponential("lambda_3", alpha)
    lambda_4 = pm.Exponential("lambda_4", alpha)
    lambda_5 = pm.Exponential("lambda_5", alpha)


tau_1 = 60*(7-4)
tau_2 = 60*(8-4)
tau_3 = 60*(16-4)
tau_4 = 60*(17-4)


with traffic_model:
    idx = np.arange(1200);
    lambda_ = pm.math.switch(idx > tau_1, lambda_1,
                             pm.math.switch(idx > tau_2, lambda_2,
                                            pm.math.switch(idx > tau_3, lambda_3,
                                                           pm.math.switch(idx > tau_4, lambda_4, lambda_5))))

with traffic_model:
    observation = pm.Poisson("observation", lambda_, observed=traffic_data)

with traffic_model:
    step = pm.Metropolis()
    trace = pm.sample(100, tune=20, step=step, return_inferencedata=False,cores=1)


lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
lambda_3_samples = trace['lambda_1']
lambda_4_samples = trace['lambda_2']
lambda_5_samples = trace['lambda_1']

az.plot_posterior({"lambda_1" : lambda_1_samples,
                   "lambda_2" : lambda_2_samples,
                   "lambda_3" : lambda_3_samples,
                   "lambda_4" : lambda_4_samples,
                   "lambda_5" : lambda_5_samples}
                  )
plt.show()

print(lambda_1_samples.mean())
print(lambda_2_samples.mean())
print(lambda_3_samples.mean())
print(lambda_4_samples.mean())
print(lambda_5_samples.mean())


