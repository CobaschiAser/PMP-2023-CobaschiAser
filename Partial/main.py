import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import csv
import pymc as pm
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx

# Ex 1

# 1


# nr de winuri pt fiecare player
wins_j0 = 0
wins_j1 = 0
# simulam 10000 de runde de joc
for i in range(10000):
    # simulam o runda de joc
    # stabilim cine incepe
    start_joc = stats.binom.rvs(1, 0.5, size=1)
    # print(start_joc)
    # daca start_joc = 0, incepe J0, altfel incepe J1
    if start_joc == 0:
        steme_j0 = 0
        steme_j1 = 0
        arunca_moneda_j0 = stats.binom.rvs(1,0.5,size=1)
        n = arunca_moneda_j0 # nr de steme obtinue
        if (arunca_moneda_j0 == 1):
            steme_j0 += 1
        # urmeaza runa a doua
        arunca_moneda_j1 = stats.binom.rvs(1,2/3,size=n+1)
        if arunca_moneda_j1[0] == 1:
            steme_j1 += 1
        print(arunca_moneda_j1)
        if len(arunca_moneda_j1 == 2):
            if arunca_moneda_j1[1] == 1:
                steme_j1 += 1
        if steme_j0 >= steme_j1:
            wins_j0 += 1
        else:
            wins_j1 += 1
    else:
        steme_j0 = 0
        steme_j1 = 0
        arunca_moneda_j1 = stats.binom.rvs(1, 2/3, size=1)
        n = arunca_moneda_j1  # nr de steme obtinue
        if (arunca_moneda_j1 == 1):
            steme_j1 += 1
        # urmeaza runda a doua
        arunca_moneda_j0 = stats.binom.rvs(1, 0.5, size=n + 1)
        if arunca_moneda_j0[0] == 1:
            steme_j0 += 1
        if len(arunca_moneda_j0) == 2:
            if arunca_moneda_j0[1] == 1:
                steme_j0 += 1
        if steme_j1 >= steme_j0:
            wins_j1 += 1
        else:
            wins_j0 += 1

# stabilim cine are sanse mai mari sa castige
if wins_j0 / 10000 <= wins_j1 / 10000:
    print("J1 are sanse mai mari sa castige")
else:
    print("J0 are sanse mai mari sa castige")

# 2

# J_R1 = jucatorul din runda 1: 0 daca incepe J0, 1 daca incepe J1
# J_R2 = jucatorul din runda 2: 0 daca este J0, 1 daca este J1
# Nr_S_1 = numarul de steme obtinute in prima runda: 0 sau 1 ( Nr_S_1 determina si numarul de aruncari din R2)
# Nr_S_2 = numarul de steme obtinute in prima runda: 0 sau 1 sau 2


bayes_model = BayesianNetwork([('J_R1', 'J_R2'), ('J_R1', 'Nr_S_1'), ('Nr_S_1', 'Nr_S_2'), ('J_R2','Nr_S_2')])

# variabila radacina
CPD_J_R1 = TabularCPD(variable='J_R1', variable_card=2,
                    values=[[0.5], [0.5]])
#print(CPD_J_R1)

# variabilele cu un singur parinte
CPD_J_R2 = TabularCPD(variable='J_R2', variable_card=2,
                    values=[[0.0, 1.0], [1.0, 0.0]], evidence=['J_R1'], evidence_card=[2])
#print(CPD_J_R2)

CPD_Nr_S_1 = TabularCPD(variable='Nr_S_1', variable_card=2,
                        values=[[0.5, 0.33],[0.5, 0.67]], evidence=['J_R1'], evidence_card=[2])
#print(CPD_Nr_S_1)
# variabilele cu 2 parinti
CPD_Nr_S_2 = TabularCPD(variable='Nr_S_2', variable_card=3,
                        values=[
                            [0.5,0.33,0.5*0.5,1/3 * 1/3],
                            [0.5,0.67,0.5*0.5,1/3 * 2/3],
                            [0.0, 0.0, 0.5*0.5,2/3 * 2/3]
                        ], evidence=['Nr_S_1','J_R2'],evidence_card=[2,2])

#print(CPD_Nr_S_2)

print(bayes_model.add_cpds(CPD_J_R1, CPD_J_R2, CPD_Nr_S_1, CPD_Nr_S_2))
#print(bayes_model.check_model())

#3

infer = VariableElimination(bayes_model)
posterior_inceput = infer.query(["J_R1"], evidence={"Nr_S_2": 1})
# probab pt J_R1 stiind ca Nr_S_2 = 1


# Ex 2
# 1
miu = 2
sigma = 0.5
#waiting_time = stats.norm.rvs(loc=miu, scale=sigma, size=100)

# in aceasta lista vom alcatui cei 100 de timpi medii de asteptare
meanlist = []
for i in range(100):
    wait_nr = stats.norm.rvs(loc=miu, scale=sigma,size=100).mean()
    meanlist.append(wait_nr)
# print(waiting_time)


# 2

with pm.Model() as model:
    nr_clienti = pm.Poisson("nr_clienti", mu=20)
    timp_asteptare = pm.Normal("timp_asteptare", mu=2, sigma=0.5)
    observation = pm.Poisson("obs", mu=timp_asteptare, observed=meanlist)

    with model:
        trace = pm.sample(1000, cores=1)
        az.plot_posterior(trace)
        plt.show()

# 3


