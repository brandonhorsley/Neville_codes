"""
Custom document to practice implementing a custom likelihood function.

P doesn't sum to 1?
-> Why does U have complex-valued elements
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pymc as pm

from pymc import Model, Normal, Uniform,Multinomial,Metropolis,sample

print(f"Running on PyMC v{pm.__version__}")

#####################Generating Data#######################

RANDOM_SEED = 8927 #Seed for numpy random
rng = np.random.default_rng(RANDOM_SEED)

#%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

#########################Supporting functions

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
N=1 #Top of page 108 ->N=number of experiments
M=2 #Number of modes
#V_dist=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5
#V=V_dist+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#V=V_dist
V=2.5

a1_true=0
a2_true=0

b1_true=0.788
b2_true=0.711

eta1_true=0.447
eta2_true=0.548
eta3_true=0.479

def DataGen(InputNumber): #InputNumber=# of input photons= should average to about 1000
    
    #Input into mth mode of beamsplitter
    phi1_true=a1_true+b1_true*V**2 #phi=a+bV**2
    phi2_true=a2_true+b2_true*V**2 #phi=a+bV**2
    U_true=ConstructU(eta1_true,eta2_true,eta3_true,phi1_true,phi2_true) #Generate double MZI Unitary
    #print(U_true.conj().T@U_true) #verifies unitarity
    #print(U_true)
    P_click1_true=np.abs(top_bra@U_true@top_ket)**2 #Probability of click in top
    P_click1_true=P_click1_true[0][0]
    P_click2_true=np.abs(bottom_bra@U_true@top_ket)**2 #Probability of click in bottom
    P_click2_true=P_click2_true[0][0]
    P_true=[P_click1_true,P_click2_true]
    #print(P_true)
    #n=C,p=P,x=array of clicks
    data=scipy.stats.multinomial.rvs(n=InputNumber,p=P_true)
    C=np.sum(data)

    return data,C

data,C=DataGen(InputNumber=1000)
#print(np.shape(data))
#print(data) #Correct
#print(C)

############################Model

with Model() as model:
    # Define priors
    eta1 = Normal("eta1", mu=0.5, sigma=0.05,initval=0.5)
    eta2 = Normal("eta2", mu=0.5, sigma=0.05,initval=0.5)
    eta3= Normal("eta3", mu=0.5, sigma=0.05,initval=0.5)
    a1= Uniform("a1", lower=-np.pi, upper=np.pi,initval=0)
    a2= Uniform("a2", lower=-np.pi, upper=np.pi,initval=0)
    b1= Normal("b1", mu=0.7, sigma=0.07,initval=0.5)
    b2= Normal("b2", mu=0.7, sigma=0.07,initval=0.5)

    def Likelihood(eta1,eta2,eta3,a1,a2,b1,b2):
        #To be called after data generation

        phi1=a1+b1*V**2 #phi=a+bV**2
        phi2=a2+b2*V**2 #phi=a+bV**2
        U=ConstructU(eta1,eta2,eta3,phi1,phi2) #Generate double MZI Unitary
        #print(U.conj().T@U)
        P_click1=abs(top_bra@U@top_ket)**2 #Probability of click in top
        P_click1=P_click1[0][0]
        P_click2=abs(bottom_bra@U@top_ket)**2 #Probability of click in bottom
        P_click2=P_click2[0][0]
        P=np.array([P_click1.eval(),P_click2.eval()])
        #P[i]=[P_click1.eval(),P_click2.eval()]
        #n=C,p=P,x=array of clicks
        #prob[i]=scipy.stats.multinomial.pmf(x=data[i],n=C[i],p=P)
        #print(np.sum(prob))
        #return prob
        #print(P)
        P=np.log(P)
        #prob=pm.Multinomial.dist(n=C,p=P)
        #return prob
        return P
    
    #likelihood = Multinomial("likelihood", n=C, p=Likelihood(eta1,eta2,eta3,a1,a2,b1,b2,V), shape=(N,M), observed=data)
    pm.DensityDist("likelihood", eta1, eta2, eta3, a1,a2, b1, b2, Likelihood, observed=data)

    idata = sample(draws=int(1e5), chains=4,step=pm.Metropolis(), return_inferencedata=True,cores=1)

###################Bayesian analysis
az.plot_trace(idata)
#az.summary(idata, round_to=2)
#az.plot_ess(idata)