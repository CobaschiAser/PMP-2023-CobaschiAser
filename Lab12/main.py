import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import scipy.stats as stats

# Ex 1

def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1 / grid_points, grid_points)  # uniform prior
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


def posterior_grid_1(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = (grid <= 0.5).astype(int)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


def posterior_grid_2(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = abs(grid - 0.5)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


def operation(function, n, zeros):
    data = np.repeat([0, 1], (zeros, n - zeros))
    points = n
    h = data.sum()
    t = len(data) - h
    grid, posterior = function(points, h, t)
    plt.plot(grid, posterior, 'o-')
    plt.title(f'heads = {h}, tails = {t}')
    plt.yticks([])
    plt.xlabel('θ')
    plt.show()


operation(posterior_grid_1, 100, 65)
operation(posterior_grid_1, 20, 16)

operation(posterior_grid_2, 100, 65)
operation(posterior_grid_2, 20, 16)


# Ex 2


errors = {100: [], 1000: [], 10000: []}

N_test = [100, 1000, 10000]
num_simulations = 100

for N in N_test:
    for _ in range(num_simulations):
        x, y = np.random.uniform(-1, 1, size=(2, N))
        inside = (x**2 + y**2) <= 1
        pi = inside.sum()*4/N
        error = abs((pi - np.pi))
        errors[N].append(error)

mean_errors = {N: np.mean(errors[N]) for N in N_test}
std_errors = {N: np.std(errors[N]) for N in N_test}

plt.figure(figsize=(8, 6))
for N in N_test:
    plt.errorbar([N] * num_simulations, errors[N], yerr=std_errors[N], fmt='o', alpha=0.5, label=f'N={N}')

plt.errorbar(N_test, [mean_errors[N] for N in N_test], yerr=[std_errors[N] for N in N_test], fmt='o', color='black', capsize=5, label='Mean Error')
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Error vs N (100 simulations per N)')
plt.xscale('log')
plt.legend()
plt.show()



# Cu cat N creste, cu atat eroarea scade(se observa din grafic)

# Ex 3
def metropolis(func, draws=10000):
    trace = np.zeros(draws)
    old_x = func.mean()
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace


beta_params = [(1, 5), (15, 15), (2, 3)]
for a, b in beta_params:
    func = stats.beta(a, b)
    trace = metropolis(func=func)
    x = np.linspace(0.01, .99, 100)
    y = func.pdf(x)
    plt.xlim(0, 1)
    plt.plot(x, y, 'C1-', lw=3, label='True distribution')
    plt.hist(trace[trace > 0], bins=25, density=True, label='Estimated distribution')
    plt.xlabel('x')
    plt.ylabel('pdf(x)')
    plt.yticks([])
    plt.legend()
    plt.show()
