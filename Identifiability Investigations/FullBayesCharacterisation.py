"""
This code file is to generalise the work in bayesianPS to characterise a full Clements scheme. Will leverage the routine presented in Niko di Giano thesis (https://www.politesi.polimi.it/retrieve/8eefcc53-fdb5-470a-8f92-8f1a9b4844b6/Di_Giano_2021_Universal_Photonic_Processors.pdf).

Notes/thoughts:
- How to set phase to transmission point, in case of beamsplitter deviations, will have to account for error propagation if can't achieve 100% transmission or reflection?
- Mean point estimates aren't guaranteed to be allow the maths to work, e.g. for P_0, (-B_max/A_max) can come out greater than 1 and thus taking arccos won't work. Perhaps go full Niko di giano with characterisation method?
- Niko di Giano description of Clements characterisation process is split into three parts. Main diagonal is input top and output bottom, so work through that, characterise one element, set to transmit to next along diagonal until reached end, then switch to characterising upper triangle of the circuit so essentially switch to next diagonal by changing top MZI in diagonal to reflection then work through diagonal and repeat until upper triangle characterised. For characterising bottom triangle, switch diagonals by dropping input 2 modes down and then work through each diagonal as you do. 
- Writing an automatic code to do it appears to be convoluted to implement, even if conceptually simple, having to hardcode it is likely.
"""


#Import modules
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.stats
import multiprocessing
import pymc as pm
import arviz as az

#Priming
cpucount=multiprocessing.cpu_count()
RANDOM_SEED = 0
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")


#Define initialisation parameters
m=2 #Number of circuit modes ('rows' in the circuit)
Vmax=10 #Max voltage setting on phase shifters
N=100 #Top of page 108 ->N=number of experiments

#for _ in range(m*(m-1)/2):
    


N = 1000

a = 2E-4
b = 2E-4
c = 2E-4

A = 1
B = 0
C=200
#C = -200  # 200 and -200 give same result = locally identifiable
theta_0 = 0

n_phaseshifters = 1

V = np.random.uniform(low=0, high=Vmax, size=(n_phaseshifters, N))  # without noise, value we believe we are setting
V_noisy = V + rng.normal(scale=0.05, size=(n_phaseshifters, N))  # extra noise on voltage, value actually set

P_elect = a * V_noisy + b * (V_noisy ** 2) + c * (V_noisy ** 3)  # electrical power

phi = C * P_elect - theta_0  # phase
P_opt = A * np.cos(phi) + B  # optical power

with pm.Model() as model:  # model specifications in PyMC are wrapped in a with-statement
    # Define priors
    #a = pm.Normal("a", 2E-4,sigma=1)
    #b = pm.Normal("b", 2E-4,sigma=1)
    #c = pm.Normal("c", 2E-4,sigma=1)
    A = pm.Normal("A", 1,sigma=.1)
    B = pm.Normal("B", 0,sigma=.1)
    C = pm.Normal("C", 200,sigma=.1)
    theta_0 = pm.Normal("theta_0",0,sigma=.1)

    # Define likelihood
    likelihood = pm.Normal("P_opt", mu=A*pm.math.cos(C * (a * V[0] + b * (V[0] ** 2) + c * (V[0] ** 3)) - theta_0) + B, sigma=.1, observed=P_opt)

    # Inference!
    # draw 3000 posterior samples using NUTS sampling
    idata = pm.sample(3000,cores=1)

#print(az.summary(idata)) #Can look at sd to see standard deviation to compare to curveasmin
table=az.summary(idata)
#print(table['mean']['A'])
#Take max likelihood value for each of the parameters
#A_max=print(idata.posterior['A'])
A_max=table['mean']['A']
B_max=table['mean']['B']
C_max=table['mean']['C']
theta_0_max=table['mean']['theta_0']

print(A_max)
print(B_max)
print(C_max)
print(theta_0_max)

#Transmission/reflection means just find point where minimum is or maximum is
P_ref=(np.pi+theta_0_max)/C_max
V_ref=np.roots([c,b,a,-P_ref]) #Don't forget the minus sign!
filtered = [i.real for i in V_ref if i.imag == 0]
#print(filtered)
V_ref=filtered[0] #hopefully there will only be one non complex number

P_trans=(theta_0_max)/C_max
V_trans=np.roots([c,b,a,-P_trans]) #Don't forget the minus sign!
filtered = [i.real for i in V_trans if i.imag == 0]
#print(filtered)
V_trans=filtered[0] #hopefully there will only be one non complex number

#az.plot_trace(data=idata) #Show traceplots


"""
Trying to infer a,b,c as well botches performance, again likely rooted in identifiability with too many free params/freedom?
Also sigma=1 leads to OK performance but stricter priors (0.1) leads to far better performance.
"""