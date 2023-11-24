import csv
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import scipy.stats as stats

# IMPORT DATA FROM CSV FILE

f = open("trafic.csv", "r")
data_row = csv.reader(f)
rows = []
# extragem header-ul
header = next(data_row)
# iteream prin perechile (minut, nr.masini) si extragem nr.de masini
for row in data_row:
    rows.append(int(row[1]))
# print(rows)

traffic_data = np.array(rows)

with pm.Model() as model:
    alpha = 1.0 / np.mean(traffic_data)

# avem 5 intervale de timp: 4-7, 7-8, 8-16 16-17, 17-24
# calculam un lambda pentru fiecare interval de timp
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    lambda_3 = pm.Exponential("lambda_3", alpha)
    lambda_4 = pm.Exponential("lambda_4", alpha)
    lambda_5 = pm.Exponential("lambda_5", alpha)

# stabilim minutele "Granita" ale intervalelor:

tau_1 = 60*(7-4)
tau_2 = 60*(8-4)
tau_3 = 60*(16-4)
tau_4 = 60*(19-4)

with model:
    idx = np.arange(1200) # 20 * 60 minute
    lambda_ = pm.math.switch(idx > tau_1, lambda_1,
                                pm.math.switch(idx > tau_2, lambda_2,
                                    pm.math.switch(idx > tau_3, lambda_3,
                                        pm.math.switch(idx > tau_4, lambda_4, lambda_5)
                                    )
                                )
                             )


# precizam care sunt datele observate

with model:
    observation = pm.Poisson("observation", lambda_, observed=traffic_data)

# inferenta
with model:
    step = pm.Metropolis()
    trace = pm.sample(100, tune=20, step=step, return_inferencedata=False,cores=1)

#esantionare

lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
lambda_3_samples = trace['lambda_3']
lambda_4_samples = trace['lambda_4']
lambda_5_samples = trace['lambda_5']

az.plot_posterior({"lambda_1": lambda_1_samples,
                  "lambda_2": lambda_2_samples,
                   "lambda_3": lambda_3_samples,
                   "lambda_4": lambda_4_samples,
                   "lambda_5": lambda_5_samples,
                   })

plt.show()
print(lambda_1_samples.mean())
print(lambda_2_samples.mean())
print(lambda_3_samples.mean())
print(lambda_4_samples.mean())
print(lambda_5_samples.mean())