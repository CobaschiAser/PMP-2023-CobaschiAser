import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
# Ex 1

centered_eight_data = az.load_arviz_data("centered_eight")

print("Modelul centrat:")
print("Numărul de lanțuri:", centered_eight_data.posterior.chain.size)
print("Mărimea totală a eșantionului generat:", centered_eight_data.posterior.chain.size * centered_eight_data.posterior.draw.size)

#az.plot_posterior(centered_eight_data, round_to=2)
#plt.show()

non_centered_eight_data = az.load_arviz_data("non_centered_eight")

print("\nModelul necentrat:")
print("Numărul de lanțuri:", non_centered_eight_data.posterior.chain.size)
print("Mărimea totală a eșantionului generat:", non_centered_eight_data.posterior.chain.size * non_centered_eight_data.posterior.draw.size)

#az.plot_posterior(non_centered_eight_data, round_to=2)
#plt.show()


#Results
'''
Modelul centrat:
Numărul de lanțuri: 4
Mărimea totală a eșantionului generat: 2000

Modelul necentrat:
Numărul de lanțuri: 4
Mărimea totală a eșantionului generat: 2000
'''

# Ex 2

parameters = ["mu", "tau"]

# calculam Rhat pentru fiecare model și parametru
rhat_centered = az.rhat(centered_eight_data, var_names=parameters)
rhat_non_centered = az.rhat(non_centered_eight_data, var_names=parameters)

print("Rhat pentru modelul centrat:")
print(rhat_centered)

print("\nRhat pentru modelul necentrat:")
print(rhat_non_centered)


az.plot_autocorr(centered_eight_data, var_names=parameters)
plt.show()

az.plot_autocorr(non_centered_eight_data, var_names=parameters)
plt.show()

# Results
'''
Rhat pentru modelul centrat:
<xarray.Dataset>
Dimensions:  ()
Data variables:
    mu       float64 1.02
    tau      float64 1.062

Rhat pentru modelul necentrat:
<xarray.Dataset>
Dimensions:  ()
Data variables:
    mu       float64 1.003
    tau      float64 1.003

'''

# Ex 3

divergences_centered = centered_eight_data.sample_stats.diverging.sum().values
divergences_non_centered = non_centered_eight_data.sample_stats.diverging.sum().values

print("Numărul de divergențe pentru modelul centrat:", divergences_centered)
print("Numărul de divergențe pentru modelul necentrat:", divergences_non_centered)

az.plot_pair(centered_eight_data, var_names=["mu", "tau"], divergences=True)
plt.suptitle("Modelul Centrat - Pairs plot cu divergențe")
plt.show()

az.plot_pair(non_centered_eight_data, var_names=["mu", "tau"], divergences=True)
plt.suptitle("Modelul Necentrat - Pairs plot cu divergențe")
plt.show()

""" Results:
Numărul de divergențe pentru modelul centrat: 48
Numărul de divergențe pentru modelul necentrat: 0
"""

####### VARIANTA 2

c8 = az.load_arviz_data("centered_eight")
nc8 = az.load_arviz_data("non_centered_eight")


print('nr. de lanturi pentru modelul centrat este:', len(c8['posterior']['chain']))
print('marimea totala a esantionului pentru modelul centrat este:', len(c8['posterior']['draw'])*len(c8['posterior']['chain']))
az.plot_trace(c8,divergences='top')

print('nr. de lanturi pentru modelul necentrat este:', len(c8['posterior']['chain']))
print('marimea totala a esantionului pentru modelul necentrat este:', len(c8['posterior']['draw'])*len(c8['posterior']['chain']))
az.plot_trace(nc8,divergences='top')

az.rhat(c8,var_names=['mu','tau'])
az.rhat(nc8,var_names=['mu','tau'])
az.plot_autocorr(c8,var_names=['mu','tau'])
az.plot_autocorr(nc8,var_names=['mu','tau'])

print('nr. de divergente pt. modelul centrat este:', c8.sample_stats.diverging.sum().item())
print('nr. de divergente pt. modelul centrat este:', nc8.sample_stats.diverging.sum().item())

az.plot_pair(c8, var_names=['mu','tau'], divergences=True)
az.plot_parallel(c8, var_names=['mu','tau'])

az.plot_pair(nc8, var_names=['mu','tau'], divergences=True)
az.plot_parallel(nc8, var_names=['mu','tau'])
plt.show()