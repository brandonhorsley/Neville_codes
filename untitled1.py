"""
Simple code to practice implementing a custom likelihood function

https://stackoverflow.com/questions/27545420/custom-likelihood-in-pymc3
https://www.pymc.io/projects/examples/en/latest/case_studies/blackbox_external_likelihood_numpy.html
https://www.pymc.io/projects/docs/en/v4.2.2/api/distributions/generated/pymc.DensityDist.html
https://discourse.pymc.io/t/pm-densitydist-problem-under-pmyc-4/9624/7
https://discourse.pymc.io/t/creating-a-logstudentt-distribution/11225

Works decently
"""

"""
#This code block works

import pymc as pm
import numpy as np
import arviz as az

def logp(value, mu):
    return -(value - mu)**2

with pm.Model():
    mu = pm.Normal('mu',0,1)
    pm.DensityDist('density_dist',mu,logp=logp, observed=np.random.randn(100),)
    
    idata = pm.sample(100,cores=1)
    
az.plot_trace(idata)
"""
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
    a1= Uniform("a1", lower=-np.pi, upper=np.pi,initval=0)
    a2= Uniform("a2", lower=-np.pi, upper=np.pi,initval=0)
    b1= Normal("b1", mu=0.7, sigma=0.07,initval=0.5)
    b2= Normal("b2", mu=0.7, sigma=0.07,initval=0.5)
    
    mu1=a1+b1*V**2
    mu2=a2+b2*V**2
    
    phi1=Normal("phi1",mu=mu1,sigma=1,observed=phi1)
    phi2=Normal("phi2",mu=mu2,sigma=1,observed=phi2)
        
    idata = pm.sample(1000,cores=1)
    # compute maximum a-posteriori estimate
    # for logistic regression weights
    manual_map_estimate = pm.find_MAP()
    
az.plot_trace(idata)
    
#print(manual_map_estimate)
"""

