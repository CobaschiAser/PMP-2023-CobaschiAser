# importam bibliotecile necesare pt a lucra cu retele bayesiene
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx

# definim modelul/ reteaua bayesiana: fiecare pereche ('A', 'B') semnifica: B depinde de A
student_model = BayesianNetwork([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S'), ('L', 'A'), ('S', 'A')])
pos = nx.circular_layout(student_model)
#nx.draw(student_model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)

# definim variabilele radacine
CPD_D = TabularCPD(variable='D', variable_card=2, values=[[0.3], [0.7]]) # lui D ii dam probabilitatea 0.3 sa fie 0 si 0.7 sa fie 1
print(CPD_D) # variable_card = 2 -> D ia 2 valori posibile
CPD_I = TabularCPD(variable='I', variable_card=2, values=[[0.2], [0.8]]) # lui I ii dam probab 0.2 sa fie 0 si 0.8 sa fie 1
print(CPD_I)

# definim variabilele cu un singur parinte
CPD_L = TabularCPD(variable='L', variable_card=2, values=[[0.9, 0.6, 0.01],[0.1, 0.4, 0.99]],evidence=['G'],evidence_card=[3])
# L ia 2 valori, dar parintele lui este G care ia 3 valori, asadar, fiecare valoare pe care o ia L este explicitata in functie de valorile pe care le ia G
CPD_S = TabularCPD(variable='S', variable_card=2,values=[[0.8, 0.1],[0.2, 0.9]],evidence=['I'],evidence_card=[2])
# S ia 2 valori si parintele lui este I care ia 2 valori, asadar, fiecare valoare pe care o ia S este explicitata in functie de valorile pe care le ia I
print(CPD_L)
print(CPD_S)

# definim variabilele cu 2 parinti
CPD_G = TabularCPD(variable='G', variable_card=3,
#  I D   00   01    10   11
values=[[0.3, 0.7, 0.02, 0.2],
[0.4, 0.25, 0.08, 0.3],
[0.3, 0.05, 0.9, 0.5]],
evidence=['I', 'D'],
evidence_card=[2, 2])
# G are ia 3 valori, si are 2 parinti, care iau fiecare cate 2 valori, deci fiecare valoare a lui G va fi explicitata in functie
# fiecare valoare pe care o ia fiecare dintre cei 2 parinti
print(CPD_G)


CPD_A = TabularCPD(variable='A', variable_card=2,
values=[[0.9, 0.8, 0.7, 0.2],
[0.1, 0.2, 0.3, 0.8]],
evidence=['L', 'S'],
evidence_card=[2, 2])

print(CPD_A)

# adaugarea distributiilor conditionale la model
student_model.add_cpds(CPD_D, CPD_I, CPD_S, CPD_G, CPD_L, CPD_A)
print(student_model.get_cpds())

# verificarea modelului
print(student_model.check_model())

# verificarea independentelor
print(student_model.local_independencies(['D','G','S','I','L','A']))

# INFERENTA
from pgmpy.inference import VariableElimination
infer = VariableElimination(student_model)
posterior_p = infer.query(["D","I"], evidence={"A": 1})
# probab pt D si I stiind ca A = 1( a avut loc)
print(posterior_p)


