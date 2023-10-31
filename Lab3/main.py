from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination

#1
depozit_model = BayesianNetwork([('Cutremur', 'Incendiu'),
                                 ('Cutremur', 'Alarma'),
                                 ('Incendiu', 'Alarma')])

pos = nx.circular_layout(depozit_model)
nx.draw(depozit_model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)

cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])
cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2, values=[[0.99, 0.97], [0.01, 0.03]],
                          evidence=['Cutremur'], evidence_card=[2])
cpd_alarma = TabularCPD(variable='Alarma', variable_card=2,
                        values=[[0.9999, 0.05, 0.98, 0.02], [0.0001, 0.95, 0.02, 0.98]],
                        evidence=['Cutremur', 'Incendiu'], evidence_card=[2, 2])

print(cpd_cutremur)
print(cpd_incendiu)
print(cpd_alarma)

depozit_model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarma)
# print(depozit_model.get_cpds())
# print(depozit_model.check_model())

#2
infer = VariableElimination(depozit_model)
probab_cutremur_stiind_alarma = infer.query(['Cutremur'], evidence={"Alarma": 1})
print(probab_cutremur_stiind_alarma)

#3
probab_incendiu_stiind_not_alarma = infer.query(['Incendiu'],evidence={"Alarma":0})
print(probab_incendiu_stiind_not_alarma)

