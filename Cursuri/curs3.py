from scipy import stats
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
# generare de distributii

# Distributie normala N(mu, sigma)
mu = 0.
sigma = 1.
X = stats.norm.rvs(mu, sigma, size=20)
print(X)

# Exemple de densitati pt distributii normale( cu diversi parametrii)
# EXMEPLU DE PLOT PT MAI MULTE TABELE ITERATE IN ACELASI GRAFIC

"""mu_params = [-1, 0, 1]
sd_params = [0.5, 1, 1.5]
x = np.linspace(-7, 7, 200)
_, ax = plt.subplots(len(mu_params), len(sd_params), sharex=True,
sharey=True, figsize=(9, 7), constrained_layout=True)
for i in range(3):
    for j in range(3):
        mu = mu_params[i]
        sd = sd_params[j]
        X = stats.norm(mu, sd)
        y = X.pdf(x)
        z = X.rvs(10)
        ax[i,j].plot(x, y)
        ax[i,j].plot([], label="μ = {:3.2f}\nσ = {:3.2f}".format(mu,sd), alpha=0)
        ax[i,j].scatter(z, np.zeros(10), alpha=0.6)
        ax[i,j].legend(loc=1)
ax[2,1].set_xlabel('x')
ax[1,0].set_ylabel('p(x)', rotation=0, labelpad=20)
ax[1,0].set_yticks([])
plt.show()
"""
# Exemple de distributii binomiale

"""n_params = [1, 2, 4] # Number of trials
p_params = [0.25, 0.5, 0.75] # Probability of success
x = np.arange(0, max(n_params)+1)
f,ax = plt.subplots(len(n_params), len(p_params), sharex=True, sharey=True,

figsize=(8, 7), constrained_layout=True)

for i in range(len(n_params)):
    for j in range(len(p_params)):
        n = n_params[i]
        p = p_params[j]
        y = stats.binom(n=n, p=p).pmf(x)
        ax[i,j].vlines(x, 0, y, colors='C0', lw=5)
        ax[i,j].set_ylim(0, 1)
        ax[i,j].plot(0, 0, label="N = {:3.2f}\nθ = {:3.2f}".format(n,p), alpha=0)
        ax[i,j].legend()
ax[2,1].set_xlabel('y')
ax[1,0].set_ylabel('p(y | θ)')
ax[0,0].set_xticks(x)
plt.show()
"""

# Calcului si reprezentarea grafica a distributiei A POSTERIORI
"""
plt.figure(figsize=(10, 8))
n_trials = [0, 1, 2, 3, 4, 8, 16, 32, 50, 150]
data = [0, 1, 1, 1, 1, 4, 6, 9, 13, 48]
theta_real = 0.35
beta_params = [(1, 1), (20, 20), (1, 4)]
dist = stats.beta
x = np.linspace(0, 1, 200)

for idx, N in enumerate(n_trials):
    if idx == 0:
        plt.subplot(4, 3, 2)
        plt.xlabel('θ')
    else:
        plt.subplot(4, 3, idx+3)
        plt.xticks([])
    y = data[idx]
    for (a_prior, b_prior) in beta_params:
        p_theta_given_y = dist.pdf(x, a_prior + y, b_prior + N - y)
        plt.fill_between(x, 0, p_theta_given_y, alpha=0.7)
        plt.axvline(theta_real, ymax=0.3, color='k')
        plt.plot(0, 0, label=f'{N:4d} aruncari\n{y:4d} steme', alpha=0)
        plt.xlim(0, 1)
        plt.ylim(0, 12)
        plt.legend()
        plt.yticks([])
plt.tight_layout()
plt.show()"""

np.random.seed(1)
az.plot_posterior({'θ':stats.beta.rvs(5, 11, size=1000)})
plt.show()

