from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# 通过边来定义贝叶斯模型

model = BayesianModel([('A','L'),('N','L'),('N','C'),
                       ('L','G'),('L','P'),('G','P'),('I','P'),('S','P'),
                       ('C','S')])


# 定义条件概率分布
cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.3,0.7]])
cpd_n = TabularCPD(variable='N', variable_card=2, values=[[0.4,0.6]])
cpd_i = TabularCPD(variable='I', variable_card=3, values=[[0.33,0.34,0.33]])

cpd_l = TabularCPD(variable='L', variable_card=3,
                   values=[[0.3,0.8,0.2,0.5],
                           [0.4,0.15,0.4,0.35],
                           [0.3,0.05,0.4,0.15]],
                   evidence=['A','N'],
                   evidence_card=[2,2])
cpd_c = TabularCPD(variable='C', variable_card=2,
                   values=[[0.6,0.3],
                           [0.4,0.7]],
                   evidence=['N'],
                   evidence_card=[2])

cpd_s = TabularCPD(variable='S', variable_card=2,
                   values=[[0.7,0.8],
                           [0.3,0.2]],
                   evidence=['C'],
                   evidence_card=[2])
cpd_g = TabularCPD(variable='G', variable_card=2,
                   values=[[0.3,0.6,0.9],
                           [0.7,0.4,0.1]],
                   evidence=['L'],
                   evidence_card=[3])
cpd_p = TabularCPD(variable='P', variable_card=3,
                   values=[[0.5,0.4,0.35,0.4,0.35,0.3,0.45,0.4,0.35,0.25,0.2,0.1,0.7,0.65,0.65,0.55,0.5,0.45,0.6,0.55,0.5,0.4,0.3,0.3,0.8,0.75,0.75,0.65,0.6,0.55,0.7,0.64,0.61,0.48,0.41,0.37],
                           [0.4,0.45,0.45,0.3,0.3,0.25,0.4,0.45,0.45,0.3,0.25,0.2,0.299,0.33,0.32,0.35,0.35,0.4,0.35,0.35,0.4,0.4,0.4,0.3,0.1999,0.24,0.23,0.3,0.33,0.37,0.27,0.3,0.32,0.42,0.39,0.33],
                           [0.1,0.15,0.2,0.3,0.35,0.45,0.15,0.15,0.2,0.45,0.55,0.7,0.001,0.02,0.03,0.1,0.15,0.15,0.05,0.1,0.1,0.2,0.3,0.4,0.0001,0.01,0.02,0.05,0.07,0.08,0.03,0.06,0.07,0.1,0.2,0.3]],
                   evidence=['L','G','S','I'],
                   evidence_card=[3,2,2,3])

model.add_cpds(cpd_a,cpd_c,cpd_g,cpd_i,cpd_l,cpd_n,cpd_p,cpd_s)
infer = VariableElimination(model)

# q = infer.query(variables = ['P'],evidence = {'S':1,'L':2})
# print(q['P'])
# q = infer.query(variables = ['A'],evidence = {'L':1,'N':1})
# print(q['A'])

q = infer.query(variables = ['A'],evidence = {'S':1,'L':2})
print(q['A'])
