"""
File to reformulate the phase shifter curve fitting as a minimisation problem (this is better as more general since curve fit function doesn't permit regularisation).
"""

#Import modules
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.stats
import multiprocessing
import collections
from collections import defaultdict
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
from scipy.optimize import curve_fit

#Priming
cpucount=multiprocessing.cpu_count()
RANDOM_SEED = 0
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")


#Define initialisation parameters
m=2 #Number of circuit modes ('rows' in the circuit)
Vmax=10 #Max voltage setting on phase shifters
N=100 #Top of page 108 ->N=number of experiments

N = 1000

a = 2E-4
b = 2E-4
c = 2E-4

A = 1
B = 0
# C=200
C = 0  # 200 and -200 give same result = locally identifiable
theta_0 = 0

n_phaseshifters = 1

V = np.random.uniform(low=0, high=Vmax, size=(n_phaseshifters, N))  # without noise, value we believe we are setting
V_noisy = V + rng.normal(scale=0.05, size=(n_phaseshifters, N))  # extra noise on voltage, value actually set

P_elect = a * V_noisy + b * (V_noisy ** 2) + c * (V_noisy ** 3)  # electrical power

phi = C * P_elect - theta_0  # phase
P_opt = A * np.cos(phi)**2 + B  # optical power


def func(x, A, B, C, theta_0):
    return A * np.cos(C * (a * x + b * (x ** 2) + c * (x ** 3)) - theta_0) + B

def cost(x):
    A,B,C,theta_0=x
    return np.mean((func(V[0],A,B,C, theta_0)-P_opt[0])**2)

p0 = [1.2, 0, 1,0]  # initial guess
#res = scipy.optimize.minimize(cost, p0)
minimizer_kwargs = { "method": "L-BFGS-B"}
res=scipy.optimize.basinhopping(func=cost,x0=p0,minimizer_kwargs=minimizer_kwargs)
print(res)  # see if minimization succeeded.

"""
Basin hopping is a global method but lowest optimisation result from print(res) has 
Hessian Inverse 'hess_inv', need to use todense() to make a matrix array, then you 
can obtain the standard error of the estimated parameters by taking the sqrt of the 
diagonal of the Hessian inverse 
"""
print(np.sqrt(np.diag(res.lowest_optimization_result.hess_inv.todense())))