"""
This code file will be for my unitary calculator. I will use the perceval framework from quandela.

It occurs to me that a good way of going about it for my purposes is to just drum up some Haar-random unitary (with dimensions given by circuit mode number) and then do circuit decomposition from quandela. So then i should be able to obtain 'true values' for the program to infer, although then it isn't symbolically.
"""
import datetime
import time
import numpy as np
from scipy.optimize import basinhopping
import random
import perceval as pcvl
import os
import math
import perceval as pcvl
import numpy as np
#from perceval.rendering.circuit import SymbSkin, PhysSkin
import perceval.components.unitary_components as comp
#from perceval.components import BS
#R = 0.45
#beam_splitter = BS(BS.r_to_theta(R))

np.random.seed(0)
#circuit=pcvl.Circuit(3)
#circuit.add()
#pcvl.pdisplay(circuit, skin=SymbSkin())
#Reck scheme
#c1 = pcvl.Circuit.generic_interferometer(3, lambda i: comp.BS() // comp.PS(pcvl.P("φ%d" % i)), shape="triangle")
#Clements scheme
#c2 = pcvl.Circuit.generic_interferometer(3, lambda i: comp.BS() // comp.PS(pcvl.P("φ%d" % i)), shape="rectangle")

#pcvl.pdisplay(c1)
#print()
#pcvl.pdisplay(c1.U)
#print() #outputs the same, as expected.
#pcvl.pdisplay(c2)
#print()
#pcvl.pdisplay(c2.U)
#N=4
#bs = pcvl.Circuit.generic_interferometer(N, lambda idx : pcvl.BS(theta=pcvl.P("theta_%d"%idx))//(0, pcvl.PS(phi=np.pi*2*random.random())), shape="rectangle", depth = 2*N, phase_shifter_fun_gen=lambda idx: pcvl.PS(phi=np.pi*2*random.random()))
#pcvl.pdisplay(bs, recursive = True)
#pcvl.pdisplay(bs.U)
#U = bs.compute_unitary(use_symbolic=True)
#pcvl.pdisplay(U)

N=4
bs = pcvl.Circuit.generic_interferometer(N, lambda idx : pcvl.BS(theta=pcvl.P("theta_%d"%idx))//(0, pcvl.PS(pcvl.P("phi_%d"%idx))), shape="rectangle", depth = 2*N)
#pcvl.pdisplay(bs, recursive = True)
pcvl.pdisplay(bs.U)