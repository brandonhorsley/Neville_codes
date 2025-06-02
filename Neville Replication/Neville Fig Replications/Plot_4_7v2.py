"""
Code block to practice implementing categorical likelihood with 
continuous parameters which I think will help

https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/mixture_models/dirichlet_mixture_of_multinomials.html
https://gist.github.com/jeetsukumaran/2840020

"""


#This code works but a1,a2 is still yielding a different result
#Could be because my procedure isn't the same as Alex's, like eq 4.11 says
#that the single parameter proposals are normal distributions with a given sd


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy as sp
import scipy.stats
#import seaborn as sns

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")

#Define formula for Phase shifter unitary, will define just dim=2 for now
def construct_PS(phi):
    mat=np.array(([np.exp(1j*phi/2),0],[0,np.exp(-1j*phi/2)]))
    return mat


def construct_BS(eta):
    mat=np.array(([np.sqrt(eta), 1j*np.sqrt(1-eta)],[1j*np.sqrt(1-eta),np.sqrt(eta)]))
    return mat

top_ket=np.array([1,0])
top_ket.shape=(2,1)

top_bra=np.array([1,0])
top_bra.shape=(1,2)

bottom_ket=np.array([0,1])
bottom_ket.shape=(2,1)

bottom_bra=np.array([0,1])
bottom_bra.shape=(1,2)

def ConstructU(eta1,eta2,eta3,phi1,phi2):
    U=construct_BS(eta3)@construct_PS(phi2)@construct_BS(eta2)@construct_PS(phi1)@construct_BS(eta1)
    return U

#####################Data generation
I=[2,500,50,50,500,100,100,100000]
Vmax=10
N=50 #Top of page 108 ->N=number of experiments
M=2 #Number of modes
V_dist=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5
#V=V_dist+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
V=V_dist

a1_true=0
a2_true=0

b1_true=0.788
b2_true=0.711

eta1_true=0.447
eta2_true=0.548
eta3_true=0.479

def DataGen(InputNumber, Voltages, poissonian=True): #InputNumber=# of input photons= should average to about 1000
    data=np.empty((N,M))
    C=np.empty(N)
    P=np.empty((N,M))

    for i in range(N):
        #Input into mth mode of beamsplitter
        phi1_true=a1_true+b1_true*Voltages[i]**2 #phi=a+bV**2
        phi2_true=a2_true+b2_true*Voltages[i]**2 #phi=a+bV**2
        U_true=ConstructU(eta1_true,eta2_true,eta3_true,phi1_true,phi2_true) #Generate double MZI Unitary
        P_click1_true=abs(top_bra@U_true@top_ket)**2 #Probability of click in top
        P_click1_true=P_click1_true[0][0]
        P_click2_true=abs(bottom_bra@U_true@top_ket)**2 #Probability of click in bottom
        P_click2_true=P_click2_true[0][0]
        P_true=[P_click1_true,P_click2_true]
        P[i]=P_true
        #n=C,p=P,x=array of clicks
        data[i]=scipy.stats.multinomial.rvs(n=InputNumber,p=P_true)
        #Need to add poissonian noise
        #if poissonian==True:
        #    data[i]+=rng.poisson(size=len(data[i]))
        C[i]=np.sum(data[i])

    return data,C,P

data,C,P=DataGen(InputNumber=1000,Voltages=V,poissonian=False)
#print(np.shape(data))
print(data) #Correct
print(C)
print(P)

with pm.Model() as model_multinomial:
    #frac = pm.Dirichlet("frac", a=np.ones(M))
    # Define priors
    eta1 = pm.Normal("eta1", mu=0.5, sigma=0.05,initval=0.5)
    eta2 = pm.Normal("eta2", mu=0.5, sigma=0.05,initval=0.5)
    eta3= pm.Normal("eta3", mu=0.5, sigma=0.05,initval=0.5)
    a1= pm.Uniform("a1", lower=-np.pi, upper=np.pi,initval=0)
    a2= pm.Uniform("a2", lower=-np.pi, upper=np.pi,initval=0)
    #a1= pm.Normal("a1", mu=0, sigma=np.pi/200,initval=0)
    #a2= pm.Normal("a2", mu=0, sigma=np.pi/200,initval=0)
    b1= pm.Normal("b1", mu=0.7, sigma=0.07,initval=0.5)
    b2= pm.Normal("b2", mu=0.7, sigma=0.07,initval=0.5)
    
    
    counts = pm.Multinomial("counts", n=C, p=P, shape=(N, M), observed=data)

with model_multinomial:
    #trace_multinomial = pm.sample(draws=int(5e3), chains=4, step=pm.Metropolis(), cores=1,return_inferencedata=True)
    trace_multinomial = pm.sample(draws=int(5e3), chains=4, cores=1,return_inferencedata=True)
    prior = pm.sample_prior_predictive()
    #posterior=pm.sample_posterior_predictive(int(5e3))
    #trace_multinomial.extend(az.from_pymc3(prior=prior))
    
az.plot_trace(data=trace_multinomial)
#az.plot_dist_comparison(data=trace_multinomial, var_names=["eta1"])
#az.plot_ppc(prior,group="prior")
#az.plot_ppc(posterior)