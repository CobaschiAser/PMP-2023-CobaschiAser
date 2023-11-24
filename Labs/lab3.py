from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx

# in modelul descris de problema(in reteaua bayesiana), avem urmatoarele dependente:
# I depinde de C, A depinde de C si A depinde de I
depozit_model = BayesianNetwork([('C', 'I'), ('C', 'A'), ('I', 'A')])

# definim variabilele radacina( avem una singura - C)
# varibila C ia 2 valori posibile(0 sau 1), cu probabilitatile date in enunt
CPD_C = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]])
print(CPD_C)

# definim variabilele cu un singur parinte( avem una singura - I)

# variabila I ia 2 valori posibile(0 sau 1). Parintele ei ia 2 valori posibile(0 sau 1)
# I    0 1
CPD_I = TabularCPD(variable='I', variable_card=2, values=[[0.99, 0.97], [0.01, 0.03]], evidence=['C'], evidence_card=[2])
print(CPD_I)

# definim variabilele cu 2 parinti( avem una singura - A)

# variabila A ia 2 valori posibile(0 sau 1). Parintii ei iau respectiv 2 valori posibile(0 sau 1)

#
CPD_A = TabularCPD(variable='A', variable_card=2, values=[[0.9999, 0.05, 0.98, 0.02], [0.0001, 0.95, 0.02, 0.98]], evidence=['C', 'I'], evidence_card=[2,2])
print(CPD_A)

# adaugam variabilele la model
depozit_model.add_cpds(CPD_C, CPD_I, CPD_A)

# facem check la model
print(depozit_model.check_model())

## Stiind ca alarma de incendiu a fost declansata, calculati probabilitatea sa fi avut loc un cutremur

infer = VariableElimination(depozit_model)
posterior_cutremur = infer.query(["C"], evidence={"A": 1})
# probab pt C stiind ca A = 1
print(posterior_cutremur)

## Stiind ca alarma nu s-a activat, calculati probabilitatea sa fi avut loc un incendiu

posterior_incendiu = infer.query(["I"], evidence={"A": 0})
# probab pt I stiind ca A = 0
print(posterior_incendiu)