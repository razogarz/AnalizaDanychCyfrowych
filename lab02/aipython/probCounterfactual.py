# probCounterfactual.py - Counterfactual Query Example
# AIFCA Python code Version 0.9.12 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from variable import Variable
from probFactors import Prob, ProbDT, IFeq, Dist
from probGraphicalModels import BeliefNetwork
from probRC import ProbRC
from probDo import queryDo

boolean = [False, True]

# without a deterministic system
Ap = Variable("Ap", boolean, position=(0.2,0.8))
Bp = Variable("Bp", boolean, position=(0.2,0.4))
Cp = Variable("Cp", boolean, position=(0.2,0.0))
p_Ap = Prob(Ap, [], [0.5,0.5])
p_Bp = Prob(Bp, [Ap], [[0.6,0.4], [0.6,0.4]]) # does not depend on A!
p_Cp = Prob(Cp, [Bp], [[0.2,0.8], [0.9,0.1]])
abcSimple = BeliefNetwork("ABC Simple",
                        [Ap,Bp,Cp],
                        [p_Ap, p_Bp, p_Cp])
ABCsimpq = ProbRC(abcSimple)
# ABCsimpq.show_post(obs = {Ap:True, Cp:True})

# as a deterministic system with independent noise
A = Variable("A", boolean, position=(0.2,0.8))
B = Variable("B", boolean, position=(0.2,0.4))
C = Variable("C", boolean, position=(0.2,0.0))
Aprime = Variable("A'", boolean, position=(0.8,0.8))
Bprime = Variable("B'", boolean, position=(0.8,0.4))
Cprime = Variable("C'", boolean, position=(0.8,0.0))
BifA = Variable("B if a", boolean, position=(0.4,0.8))
BifnA = Variable("B if not a", boolean, position=(0.6,0.8))
CifB = Variable("C if b", boolean, position=(0.4,0.4))
CifnB = Variable("C if not b", boolean, position=(0.6,0.4))

# if1then2else3 is a probability table
# if1then2else3[x][y][z] is the deterministic probability that
#  is the value of y if x is 1 otherwise it is the value of z
if1then2else3 = [[[[1,0],[0,1]],[[1,0],[0,1]]],
                  [[[1,0],[1,0]],[[0,1],[0,1]]]]

    
p_A = Prob(A, [], [0.5,0.5])
p_B = Prob(B, [A, BifA, BifnA], if1then2else3) 
p_C = Prob(C, [B, CifB, CifnB], if1then2else3) 
p_Aprime = Prob(Aprime,[], [0.5,0.5])
p_Bprime = Prob(Bprime, [Aprime, BifA, BifnA], if1then2else3)
p_Cprime = Prob(Cprime, [Bprime, CifB, CifnB], if1then2else3)
p_bifa = Prob(BifA, [], [0.6,0.4]) 
p_bifna = Prob(BifnA, [], [0.6,0.4])
p_cifb = Prob(CifB, [], [0.9,0.1])
p_cifnb = Prob(CifnB, [], [0.2,0.8])

abcCounter = BeliefNetwork("ABC Counterfactual Example",
                     [A,B,C,Aprime,Bprime,Cprime,BifA, BifnA, CifB, CifnB],
                     [p_A,p_B,p_C,p_Aprime,p_Bprime, p_Cprime, p_bifa, p_bifna, p_cifb, p_cifnb])

abcq = ProbRC(abcCounter)
# abcq.queryDo(Cprime, obs = {Aprime:False, A:True})
# abcq.queryDo(Cprime, obs = {C:True, Aprime:False})
# abcq.queryDo(Cprime, obs = {A:True, C:True, Aprime:False})
# abcq.queryDo(Cprime, obs = {A:True, C:True, Aprime:False})
# abcq.queryDo(Cprime, obs = {A:False, C:True, Aprime:False})
# abcq.queryDo(CifB, obs = {C:True,Aprime:False})
# abcq.queryDo(CifnB, obs = {C:True,Aprime:False})

# abcq.show_post(obs = {})
# abcq.show_post(obs = {Aprime:False, A:True})
# abcq.show_post(obs = {A:True, C:True, Aprime:False})
# abcq.show_post(obs = {A:True, C:True, Aprime:True})

Order = Variable("Order", boolean, position=(0.4,0.8))
S1 = Variable("S1", boolean, position=(0.3,0.4))
S1o = Variable("S1o", boolean, position=(0.1,0.8))
S1n = Variable("S1n", boolean, position=(0.0,0.6))
S2 = Variable("S2", boolean, position=(0.5,0.4))
S2o = Variable("S2o", boolean, position=(0.7,0.8))
S2n = Variable("S2n", boolean, position=(0.8,0.6))
Dead = Variable("Dead", boolean, position=(0.4,0.0))

def eqto(var):
    return IFeq(var,True,Dist([0,1]), Dist([1,0]))
    
p_S1 = ProbDT(S1, [Order, S1o, S1n],
                   IFeq(Order,True, eqto(S1o), eqto(S1n)))
p_S2 = ProbDT(S2, [Order, S2o, S2n],
                   IFeq(Order,True, eqto(S2o), eqto(S2n)))
p_dead = Prob(Dead, [S1,S2], [[[1,0],[0,1]],[[0,1],[0,1]]])
p_order = Prob(Order, [], [0.9, 0.1])
p_s1o = Prob(S1o, [], [0.01, 0.99])
p_s1n = Prob(S1n, [], [0.99, 0.01])
p_s2o = Prob(S2o, [], [0.01, 0.99])
p_s2n = Prob(S2n, [], [0.99, 0.01])

firing_squad = BeliefNetwork("Firing  squad",
                           [Order, S1, S1o, S1n, S2, S2o, S2n, Dead],
                           [p_order, p_dead, p_S1, p_s1o, p_s1n, p_S2, p_s2o, p_s2n])
fsq = ProbRC(firing_squad)
# fsq.queryDo(Dead)
# fsq.queryDo(Order, obs={Dead:True})
# fsq.queryDo(Dead, obs={Order:True})
# fsq.show_post({})
# fsq.show_post({Dead:True})
# fsq.show_post({Order:True})
