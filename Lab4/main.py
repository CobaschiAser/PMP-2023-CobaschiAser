import numpy
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

clientCount = stats.poisson.rvs(20, size=1000)
orderTime = stats.norm.rvs(loc=2, scale=0.5, size=1000)
cookTime = stats.expon.rvs(scale=4.03, size=1000)

#totalTime = orderTime + cookTime

az.plot_posterior({'clientCount': clientCount, 'orderTime': orderTime, 'cookTime': cookTime})

plt.xlabel('variabla aleatoare')
plt.show()

def calc_percentage(alpha):
    count = 0
    orderTime = stats.norm.rvs(loc=2, scale=0.5, size=1000)
    cookTime = stats.expon.rvs(scale=alpha, size=1000)
    totalTime = orderTime + cookTime

    for it in totalTime:
        if it < 15:
            count += 1

    return count / len(totalTime)
def found_alpha():
    alpha = 1/60
    while calc_percentage(alpha) >= 0.95:
        alpha = alpha + 1 / 120
    return alpha


print(found_alpha())


def calculate_average_time():
    orderTime = stats.norm.rvs(loc=2, scale=0.5, size=1000)
    cookTime = stats.expon.rvs(scale=found_alpha(), size=1000)
    totalTime = orderTime + cookTime
    sum = 0
    for it in totalTime:
        sum += it
    return sum/len(totalTime)

print(calculate_average_time())