 """
Code for replicating Figure 4.7

My initial idea is that PyMC will be needed for this one, i project that my main difficulty will have to be in using
the custom MCMC sampling method.

Quote from thesis:
In this process the size of the dataset produced was varied between 10 and 640 experiments (where an “experiment” 
here means setting one set of voltages V and collecting data).

Another quote:
This was achieved by taking the overall set of voltages V used to produce some data D,
adding some amount of Gaussian noise to each individual V_j in each V_i and using this perturbed set of voltages 
in computing the likelihood (Eqn. (4.7)) for a given set of parameters. The aim of this was to simulate the noisy 
read-out of voltages which might occur experimentally. In a similar way, Poissonian noise was added to
the photon count statistics in D, assuming an average total number of counts per
experiment of 1000.

It seems like PyMC advises against using any alternative sampling method against their own HMC because it leads
to poorer performance (https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/pymc3_howto/sampling_conjugate_step.html), 

https://discourse.pymc.io/t/multivariate-multinomial-logistic-regression/5242/2
https://gist.github.com/jeetsukumaran/2840020
https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/mixture_models/dirichlet_mixture_of_multinomials.html

Current layout not working because I am not using the likelihood correctly in the model context block.
https://www.pymc.io/projects/examples/en/latest/case_studies/blackbox_external_likelihood_numpy.html


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
V_dist=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5
V=V_dist+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#V=V_dist

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
        #n=C,p=P,x=array of clicks
        data[i]=scipy.stats.multinomial.rvs(n=InputNumber,p=P_true)
        #Need to add poissonian noise
        if poissonian==True:
            data[i]+=rng.poisson(size=len(data[i]))
        C[i]=np.sum(data[i])

    return data,C

data,C=DataGen(InputNumber=1000,Voltages=V,poissonian=False)
#print(np.shape(data))
#print(data) #Correct
#print(C)

def Likelihood(eta1,eta2,eta3,a1,a2,b1,b2,Voltages):
    #To be called after data generation
    P=np.empty((N,M))
    prob=np.empty(N)

    for i in range(len(Voltages)):
        phi1=a1+b1*Voltages[i]**2 #phi=a+bV**2
        phi2=a2+b2*Voltages[i]**2 #phi=a+bV**2
        U=ConstructU(eta1,eta2,eta3,phi1,phi2) #Generate double MZI Unitary
        P_click1=np.abs(top_bra@U@top_ket)**2 #Probability of click in top
        P_click1=P_click1[0][0]
        P_click2=np.abs(bottom_bra@U@top_ket)**2 #Probability of click in bottom
        P_click2=P_click2[0][0]
        P[i]=np.array([P_click1.eval(),P_click2.eval()])
        #P[i]=[P_click1.eval(),P_click2.eval()]
        #n=C,p=P,x=array of clicks
        #prob[i]=scipy.stats.multinomial.pmf(x=data[i],n=C[i],p=P)
        #print(np.sum(prob))
        #return prob
        print(P)
        return P

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


    #likelihood = Multinomial("likelihood", n=C, p=Likelihood(eta1,eta2,eta3,a1,a2,b1,b2,V), shape=(N,M), observed=data)
    #pm.Potential("likelihood", Likelihood(eta1,eta2,eta3,a1,a2,b1,b2,V), shape=(N,M), observed=data)

    idata = sample(draws=int(1e5), chains=4,step=pm.Metropolis(), return_inferencedata=True,cores=1)

###################Bayesian analysis
az.plot_trace(idata)
az.summary(idata, round_to=2)
az.plot_ess(idata)