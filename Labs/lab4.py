import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

# numarul de clienti care intra: dist poisson cu lambda=20
clients = stats.poisson.rvs(20, size=1000)
# timpul de plasare comanda si plata: dist normala cu media=2 si dev stand. = 0.5
order_pay_time = stats.norm.rvs(loc=2, scale=0.5, size=1000)
# o statie de gatit pregateste o comanda : distr exponentiala cu media alpha
# aici in mod normal am fi pus scale=alpha, dar cum nu stim alpha inca, punem orice, cat mai aproape de realitate
cook_time = stats.expon.rvs(scale=4.03, size=1000)

az.plot_posterior({"clients": clients, "order_pay_time": order_pay_time, "cook_time": cook_time})
# plt.show()

# trebuie sa calculam alpha maxim astfel incat sa servim toti clientii care intra intr-o ora
# intr-un timp mai scrut de 15 minute, cu o probab de 95 %

# cu un alpha dat, calculam procentajul
def calc_percentage(alpha):
    clients = stats.poisson.rvs(20, 1000)
    order_pay_time = stats.norm.rvs(loc=2, scale=0.5, size=1000)
    cook_time = stats.expon.rvs(scale=alpha, size=1000)

    total_time = order_pay_time + cook_time
    count = 0
    for i in total_time:
        if i < 15:
            count += 1
    return count/len(total_time)

# acum calculam alpha maxim


def calculate_max_alpha():
    alpha = 1/120
    while calc_percentage(alpha) >= 0.95:
        alpha += 1/120
    return alpha

# print(calculate_max_alpha())

# cu acest alpha calculat, timpul mediu de asteptare de servire al unui client

clients = stats.poisson.rvs(20, size=1000)
order_pay_time = stats.norm.rvs(loc=2, scale=0.5, size=1000)
cook_time = stats.expon.rvs(scale=calculate_max_alpha(), size=1000)
total_time = order_pay_time + cook_time

print(np.mean(total_time))

az.plot_posterior({"total_time": total_time})
plt.show()