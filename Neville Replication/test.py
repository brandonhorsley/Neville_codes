"""
Code to test using a simple uniform prior (+normal) and multinomial likelihood
to see what the posterior looks like.

This plot indicates then that uniform prior doesn't really change all that much so
so it is possible then that my procedure is fine but
isn't the one i should be implementing.
"""

#This code works!

import pymc as pm
from pymc import Normal,Uniform
import numpy as np
import arviz as az
import pandas as pd

#True parameter values
a1_true=0.5
b1_true=0.788

a2_true=0.48
b2_true=0.756

#Predictor variable
V=np.random.uniform(low=0,high=5,size=10)
#print(V)

#Simulate outcome variable
phi1=a1_true+b1_true*V**2
phi2=a2_true+b2_true*V**2
#print(phi1) #Checks out
  
with pm.Model():
    a= Uniform("a", lower=-np.pi, upper=np.pi,initval=0)

    likelihood=pm.Normal("likelihood",mu=0,sigma=1,observed=np.random.uniform(-np.pi,np.pi,size=100))
    #likelihood = pm.Multinomial("likelihood",n=10, p=a/(np.pi), observed=np.random.uniform(0,2*np.pi,size=10))
    idata = pm.sample(1000,cores=1)
    
az.plot_trace(idata)
    
#print(manual_map_estimate)