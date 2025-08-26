"""
Bayesian version of PS curve fitting characterisation
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

N = 1000

a = 1
b = 1E-3
c = 2E-4

A = 1
B = 0.1
C=1.5
#C = -200  # 200 and -200 give same result = locally identifiable
theta_0 = 0

n_phaseshifters = 1

V = np.random.uniform(low=0, high=Vmax, size=(n_phaseshifters, N))  # without noise, value we believe we are setting
#no noise
V_noisy=V
#with noise
#V_noisy = V + rng.normal(scale=0.05, size=(n_phaseshifters, N))  # extra noise on voltage, value actually set

P_elect = a * V + b * (V ** 2) + c * (V ** 3)  # electrical power

phi = C * P_elect - theta_0  # phase
P_opt = A * np.cos(phi)**2 + B  # optical power

with pm.Model() as model:  # model specifications in PyMC are wrapped in a with-statement
    # Define priors
    a = pm.Normal("a", 1,sigma=.001)
    b = pm.Normal("b", 0,sigma=.001)
    c = pm.Normal("c", 0,sigma=.001)
    A = pm.Normal("A", 1,sigma=1)
    B = pm.Normal("B", 0,sigma=.1)
    C = pm.Normal("C", 1,sigma=1)
    theta_0 = pm.Normal("theta_0",0,sigma=.1)

    # Define likelihood
    likelihood = pm.Normal("P_opt", mu=A*pm.math.cos(C * (a * V_noisy[0] + b * (V_noisy[0] ** 2) + c * (V_noisy[0] ** 3)) - theta_0)**2 + B, sigma=.1, observed=P_opt)

    # Inference!
    # draw 3000 posterior samples using NUTS sampling
    idata = pm.sample(3000,cores=1)
#print(idata.posterior['A'])
az.plot_trace(data=idata) #Show traceplots

print(az.summary(idata)) #Can look at sd to see standard deviation to compare to curveasmin
"""
Trying to infer a,b,c as well botches performance, again likely rooted in identifiability with too many free params/freedom?
Also sigma=1 leads to OK performance but stricter priors (0.1) leads to far better performance.
"""