# importam librarii necesare
import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

# EXERCITIUL 1

"""# creaam distributiile pentru cei doi mecanici

mecanic_1 = stats.expon.rvs(0, 1/4, size=10000)  # distributie exponentiala cu lambda = 1/4, pt 10000 valori
mecanic_2 = stats.expon.rvs(0, 1/6, size=10000)  # distributie exponentiala cu lambda = 1/6, pt 10000 valori

servit_1 = stats.binom.rvs(1, 0.4, size=10000)  # ni se spune in enunt ca mecanicul 2 serveste de 1.5 ori mai multi clienti decat mecanicul 1

# X este concatenarea cazurilor cand clientul este preluat de macanicul 1

X = np.concatenate((mecanic_1[servit_1 == 1], mecanic_2[servit_1 == 0]))

# facem plot la distributie

az.plot_posterior({"mecanic_1": mecanic_1, "mecanic_2": mecanic_2, "Servit_de_mecanic_1": servit_1, "X": X})
plt.show()"""

# EXERCITIUL 2
"""
# creeam distributiile pentru cele 4 servere
server_1 = stats.gamma.rvs(4, scale=1/3, size=10000)
server_2 = stats.gamma.rvs(4, scale=1/2, size=10000)
server_3 = stats.gamma.rvs(5, scale=1/2, size=10000)
server_4 = stats.gamma.rvs(5, scale=1/3, size=10000)

# creeam distributia pentru latenta

latenta = stats.expon.rvs(0, 1/4, size=10000)

# modelam probabilitatea de a fi ales un server

alegere_server = stats.multinomial.rvs(1, [0.25, 0.25, 0.3, 0.2], size=10000)

# Formam o matrice 4x10000, si indexam cu matricea 4x10000 care este in format one-hot, adunam la sfarsit latenta care este prezenta in toate cazurile
X = np.stack((server_1, server_2, server_3, server_4), axis=1)[alegere_server == 1] + latenta

# probabilitatea ca sa dureze mai mult de 3 milisecunde
probab_3ms = len(X[X > 3])/len(X)
print(probab_3ms)

az.plot_posterior({
    "server_1": server_1,
    "server_2": server_2,
    "server_3": server_3,
    "server_4": server_4,
    "alegere_server": np.argmax(alegere_server, axis=1),
    "latenta": latenta
})
plt.show()
"""

# EXERCITIUL 3

# METODA 1 - COMPACT

"""# ss, sb, bs, bb

rez_aruncari = stats.multinomial.rvs(10,[0.5 * 0.3, 0.5 * 0.7, 0.5 * 0.3, 0.5 * 0.7], size=100)
# 10 vine de la numarul de aruncari ale celor 2 monede dintr-un experiment
az.plot_posterior({
    "ss": rez_aruncari[:, 0], # prima coloana corespunde ss
    "sb": rez_aruncari[:, 1], # a doua coloana corespunde sb
    "bs": rez_aruncari[:, 2],
    "bb": rez_aruncari[:, 3]
})
plt.show()
"""

# METODA 2 - detaliat

ss = []
sb = []
bs = []
bb = []

for i in range(100):
    s_s = 0
    s_b = 0
    b_s = 0
    b_b = 0
    stema_m1 = stats.binom.rvs(1, 0.5, size=10)
    stema_m2 = stats.binom.rvs(1, 0.3, size=10)

    for i in range(10):
        if stema_m1[i] == 1 and stema_m2[i] == 1:
            s_s += 1
        if stema_m1[i] == 1 and stema_m2[i] == 0:
            s_b += 1
        if stema_m1[i] == 0 and stema_m2[i] == 1:
            b_s += 1
        if stema_m1[i] == 0 and stema_m2[i] == 0:
            b_b += 1

    ss.append(s_s)
    sb.append(s_b)
    bs.append(b_s)
    bb.append(b_b)

az.plot_posterior({
    "ss": ss,
    "sb": sb,
    "bs": bs,
    "bb": bb
})
plt.show()
