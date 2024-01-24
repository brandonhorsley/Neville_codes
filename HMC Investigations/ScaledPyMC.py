"""
This code file will be to finally try and incorporate a scaled up version of HMC, there are still some issues
since my workflow is very inefficient and so perhaps later refinements can be done but at this stage i just want 
some data as a kind of proof-of-concept and a limited gauge of average error as you try to estimate more parameters, 
as well as gauging runtime (not counting the time taken to incorporate the model into the code).

Active notes:

"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy as sp
import scipy.stats
#import seaborn as sns
import pytensor.tensor as pt

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")

#List of terms used




#####################Data generation

Vmax=10
N=50 #Top of page 108 ->N=number of experiments
M=2 #Number of modes


V_dist=[np.random.uniform(low=0, high=Vmax,size=N) for i in range(N)].reshape((12,N))
#V_dist+=rng.normal(scale=0.02, size=(12,N)) #Adding gaussian noise, top of page 108 says 2% voltage noise

eta_true=[np.random.normal(loc=0.5,scale=0.05) for i in range(12)]
a_true=[np.random.normal(loc=0,scale=np.pi/200) for i in range(12)]
b_true=[np.random.normal(loc=0.7,scale=0.07) for i in range(12)]


eta_est=eta_true+np.random.normal(loc=0,scale=0.05,size=12)
a_est=a_true+np.random.normal(loc=0,scale=np.pi/200,size=12)
b_est=b_true+np.random.normal(loc=0,scale=0.07,size=12)

def DataGen(InputNumber, Voltages, poissonian=True): #InputNumber=# of input photons= should average to about 1000
    data=np.empty((N,M))
    C=np.empty(N)
    P=np.empty((N,M))
    
    #Need to output data/counts, will do through the sympy expressions and substitute in values

    ############
    # CODE HERE#
    ############

    return data,C,P

data,C,P=DataGen(InputNumber=1000,Voltages=V,poissonian=False)

L_dev=4E-9 #delta L, chosen so it coincides with Alex's sd to be ~np.pi/200
wv=1550E-9
a_dev=(2*np.pi*L_dev)/wv

with pm.Model() as model_multinomial:
    # Define priors
    eta_0=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta0_true)
    a_0=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a0_true)  
    b_0=pm.Normal("b", mu=b_est, sigma=0.07,initval=b0_true)
    
    Volt_0=pm.Deterministic("Volt",pt.as_tensor(V_dist0))
    phi_0=pm.Deterministic("phi",(a_0+b_0*pm.math.sqr(Volt_0)))
    
    p1=pm.Deterministic("p1",)
    p2=pm.Deterministic("p2",)
    p3=pm.Deterministic("p3",)

    P=pm.Deterministic("P",pm.math.stack([p1,p2],axis=-1))
    P=pm.Deterministic("P",pm.math.stack([p1,p2,p3],axis=-1))
    
    likelihood=pm.Multinomial("likelihood",n=C,p=P,shape=(N,M),observed=data)


with model_multinomial:
    trace_multinomial_HMC = pm.sample(draws=int(5e3), chains=4, cores=1,return_inferencedata=True)
    

az.calculate_point_estimate()


"""
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy as sp
import scipy.stats
#import seaborn as sns
import pytensor.tensor as pt

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")

#List of terms used




#####################Data generation

Vmax=10
N=50 #Top of page 108 ->N=number of experiments
M=2 #Number of modes

#All these voltages will be the same while the random seed is set I think?
V_dist0=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and V_max
V_dist1=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and V_max
V_dist2=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and V_max
V_dist3=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and V_max
V_dist4=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and V_max
V_dist5=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and V_max
V_dist6=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and V_max
V_dist7=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and V_max
V_dist8=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and V_max
V_dist9=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and V_max
V_dist10=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and V_max
V_dist11=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and V_max
#V_dist0=V_dist0+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#V_dist1=V_dist1+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#V_dist2=V_dist2+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#V_dist3=V_dist3+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#V_dist4=V_dist4+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#V_dist5=V_dist5+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#V_dist6=V_dist6+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#V_dist7=V_dist7+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#V_dist8=V_dist8+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#V_dist9=V_dist9+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#V_dist10=V_dist10+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#V_dist11=V_dist11+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise

eta0_true=np.random.normal(loc=0.5,scale=0.05)
eta1_true=np.random.normal(loc=0.5,scale=0.05)
eta2_true=np.random.normal(loc=0.5,scale=0.05)
eta3_true=np.random.normal(loc=0.5,scale=0.05)
eta4_true=np.random.normal(loc=0.5,scale=0.05)
eta5_true=np.random.normal(loc=0.5,scale=0.05)
eta6_true=np.random.normal(loc=0.5,scale=0.05)
eta7_true=np.random.normal(loc=0.5,scale=0.05)
eta8_true=np.random.normal(loc=0.5,scale=0.05)
eta9_true=np.random.normal(loc=0.5,scale=0.05)
eta10_true=np.random.normal(loc=0.5,scale=0.05)
eta11_true=np.random.normal(loc=0.5,scale=0.05)

a0_true=np.random.normal(loc=0,scale=np.pi/200)
a1_true=np.random.normal(loc=0,scale=np.pi/200)
a2_true=np.random.normal(loc=0,scale=np.pi/200)
a3_true=np.random.normal(loc=0,scale=np.pi/200)
a4_true=np.random.normal(loc=0,scale=np.pi/200)
a5_true=np.random.normal(loc=0,scale=np.pi/200)
a6_true=np.random.normal(loc=0,scale=np.pi/200)
a7_true=np.random.normal(loc=0,scale=np.pi/200)
a8_true=np.random.normal(loc=0,scale=np.pi/200)
a9_true=np.random.normal(loc=0,scale=np.pi/200)
a10_true=np.random.normal(loc=0,scale=np.pi/200)
a11_true=np.random.normal(loc=0,scale=np.pi/200)

b0_true=np.random.normal(loc=0.7,scale=0.07)
b1_true=np.random.normal(loc=0.7,scale=0.07)
b2_true=np.random.normal(loc=0.7,scale=0.07)
b3_true=np.random.normal(loc=0.7,scale=0.07)
b4_true=np.random.normal(loc=0.7,scale=0.07)
b5_true=np.random.normal(loc=0.7,scale=0.07)
b6_true=np.random.normal(loc=0.7,scale=0.07)
b7_true=np.random.normal(loc=0.7,scale=0.07)
b8_true=np.random.normal(loc=0.7,scale=0.07)
b9_true=np.random.normal(loc=0.7,scale=0.07)
b10_true=np.random.normal(loc=0.7,scale=0.07)
b11_true=np.random.normal(loc=0.7,scale=0.07)


eta_est=eta_true+np.random.normal(loc=0,scale=0.05)
a_est=a_true+np.random.normal(loc=0,scale=np.pi/200)
if a_est>np.pi:
    a_est=np.pi
if a_est<-np.pi:
    a_est=-np.pi
b_est=b_true+np.random.normal(loc=0,scale=0.07)

def DataGen(InputNumber, Voltages, poissonian=True): #InputNumber=# of input photons= should average to about 1000
    data=np.empty((N,M))
    C=np.empty(N)
    P=np.empty((N,M))
    
    #Need to output data/counts, will do through the sympy expressions and substitute in values

    ############
    # CODE HERE#
    ############

    return data,C,P

data,C,P=DataGen(InputNumber=1000,Voltages=V,poissonian=False)

L_dev=4E-9 #delta L, chosen so it coincides with Alex's sd to be ~np.pi/200
wv=1550E-9
a_dev=(2*np.pi*L_dev)/wv

with pm.Model() as model_multinomial:
    # Define priors
    eta_0=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta0_true)
    eta_1=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta1_true)
    eta_2=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta2_true)
    eta_3=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta3_true)
    eta_4=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta4_true)
    eta_5=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta5_true)
    eta_6=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta6_true)
    eta_7=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta7_true)
    eta_8=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta8_true)
    eta_9=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta9_true)
    eta_10=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta10_true)
    eta_11=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta11_true)

    a_0=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a0_true)
    a_1=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a1_true)
    a_2=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a2_true)
    a_3=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a3_true)
    a_4=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a4_true)
    a_5=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a5_true)
    a_6=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a6_true)
    a_7=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a7_true)
    a_8=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a8_true)
    a_9=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a9_true)
    a_10=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a10_true)
    a_11=pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a11_true)

    
    b_0=pm.Normal("b", mu=b_est, sigma=0.07,initval=b0_true)
    b_1=pm.Normal("b", mu=b_est, sigma=0.07,initval=b1_true)
    b_2=pm.Normal("b", mu=b_est, sigma=0.07,initval=b2_true)
    b_3=pm.Normal("b", mu=b_est, sigma=0.07,initval=b3_true)
    b_4=pm.Normal("b", mu=b_est, sigma=0.07,initval=b4_true)
    b_5=pm.Normal("b", mu=b_est, sigma=0.07,initval=b5_true)
    b_6=pm.Normal("b", mu=b_est, sigma=0.07,initval=b6_true)
    b_7=pm.Normal("b", mu=b_est, sigma=0.07,initval=b7_true)
    b_8=pm.Normal("b", mu=b_est, sigma=0.07,initval=b8_true)
    b_9=pm.Normal("b", mu=b_est, sigma=0.07,initval=b9_true)
    b_10=pm.Normal("b", mu=b_est, sigma=0.07,initval=b10_true)
    b_11=pm.Normal("b", mu=b_est, sigma=0.07,initval=b11_true)
    
    Volt_0=pm.Deterministic("Volt",pt.as_tensor(V_dist0))
    Volt_1=pm.Deterministic("Volt",pt.as_tensor(V_dist1))
    Volt_2=pm.Deterministic("Volt",pt.as_tensor(V_dist2))
    Volt_3=pm.Deterministic("Volt",pt.as_tensor(V_dist3))
    Volt_4=pm.Deterministic("Volt",pt.as_tensor(V_dist4))
    Volt_5=pm.Deterministic("Volt",pt.as_tensor(V_dist5))
    Volt_6=pm.Deterministic("Volt",pt.as_tensor(V_dist6))
    Volt_7=pm.Deterministic("Volt",pt.as_tensor(V_dist7))
    Volt_8=pm.Deterministic("Volt",pt.as_tensor(V_dist8))
    Volt_9=pm.Deterministic("Volt",pt.as_tensor(V_dist9))
    Volt_10=pm.Deterministic("Volt",pt.as_tensor(V_dist10))
    Volt_11=pm.Deterministic("Volt",pt.as_tensor(V_dist11))

    phi_0=pm.Deterministic("phi",(a_0+b_0*pm.math.sqr(Volt_0)))
    phi_1=pm.Deterministic("phi",(a_1+b_1*pm.math.sqr(Volt_1)))
    phi_2=pm.Deterministic("phi",(a_2+b_2*pm.math.sqr(Volt_2)))
    phi_3=pm.Deterministic("phi",(a_3+b_3*pm.math.sqr(Volt_3)))
    phi_4=pm.Deterministic("phi",(a_4+b_4*pm.math.sqr(Volt_4)))
    phi_5=pm.Deterministic("phi",(a_5+b_5*pm.math.sqr(Volt_5)))
    phi_6=pm.Deterministic("phi",(a_6+b_6*pm.math.sqr(Volt_6)))
    phi_7=pm.Deterministic("phi",(a_7+b_7*pm.math.sqr(Volt_7)))
    phi_8=pm.Deterministic("phi",(a_8+b_8*pm.math.sqr(Volt_8)))
    phi_9=pm.Deterministic("phi",(a_9+b_9*pm.math.sqr(Volt_9)))
    phi_10=pm.Deterministic("phi",(a_10+b_10*pm.math.sqr(Volt_10)))
    phi_11=pm.Deterministic("phi",(a_11+b_11*pm.math.sqr(Volt_11)))
    
    p1=pm.Deterministic("p1",)
    p2=pm.Deterministic("p2",)
    p3=pm.Deterministic("p3",)

    P=pm.Deterministic("P",pm.math.stack([p1,p2],axis=-1))
    P=pm.Deterministic("P",pm.math.stack([p1,p2,p3],axis=-1))
    
    likelihood=pm.Multinomial("likelihood",n=C,p=P,shape=(N,M),observed=data)


with model_multinomial:
    trace_multinomial_HMC = pm.sample(draws=int(5e3), chains=4, cores=1,return_inferencedata=True)
    

az.calculate_point_estimate()
"""