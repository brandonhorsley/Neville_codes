"""
Modifying the example found at: https://www.pymc.io/projects/examples/en/latest/case_studies/blackbox_external_likelihood_numpy.html
"""

import arviz as az
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from Aux_Nev import *

print(f"Running on PyMC v{pm.__version__}")

#%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
##############################################################################################

RANDOM_SEED = 8927 #Seed for numpy random
rng = np.random.default_rng(RANDOM_SEED)

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

##########################################################

#####################Data generation
Vmax=5
N=100 #Top of page 108 ->N=number of experiments
M=2 #Number of modes
V1=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter 1
#V1=V1+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
V2=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter 2
#V2=V2+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
V=[V1,V2]

a1_true=0
a2_true=0

b1_true=0.788
b2_true=0.711

eta1_true=0.447
eta2_true=0.548
eta3_true=0.479

def DataGen(InputNumber, V1, V2,poissonian=False): #InputNumber=# of input photons= should average to about 1000
    data=np.empty((N,M))
    C=np.empty(N)

    for i in range(N):
        #Input into mth mode of beamsplitter
        phi1_true=a1_true+b1_true*V1[i]**2 #phi=a+bV**2
        phi2_true=a2_true+b2_true*V2[i]**2 #phi=a+bV**2
        U_true=ConstructU(eta1_true,eta2_true,eta3_true,phi1_true,phi2_true) #Generate double MZI Unitary
        P_click1_true=abs(top_bra@U_true@top_ket)**2 #Probability of click in top
        P_click1_true=P_click1_true[0][0]
        P_click2_true=abs(bottom_bra@U_true@top_ket)**2 #Probability of click in bottom
        P_click2_true=P_click2_true[0][0]
        P_true=[P_click1_true,P_click2_true]
        #n=C,p=P,x=array of clicks
        data[i]=scipy.stats.multinomial.rvs(n=InputNumber,p=P_true)
        #Need to add poissonian noise
        if poissonian==True:
            data[i]+=rng.poisson(size=len(data[i]))
        C[i]=np.sum(data[i])

    return data,C

#data,C=DataGen(InputNumber=1000,Voltages=V,poissonian=False)
data,C=DataGen(InputNumber=1000,V1=V1,V2=V2,poissonian=False)
#print(np.shape(data))
#print(data) #Correct
#print(C)

###################################################################################################

#simple model
def my_model(theta: tuple, V: list) -> list:
    eta1,eta2,eta3,a1,a2,b1,b2 = theta
    V1,V2=V

    return m * x + c

def my_loglike(theta, x, data, sigma):
    model = my_model(theta, x)
    return -(0.5 / sigma**2) * np.sum((data - model) ** 2)
