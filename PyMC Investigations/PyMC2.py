"""
Following my struggle with PyMC1 and hitting roadblocks i am going to 
instead start from first principles and put together a PyMC code blind.
"""

#import pymc as pm # import PyMC3
#import numpy as np
#print('PyMC version: {}'.format(pm.__version__))


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pymc as pm
import aesera as A
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

def DataGen(InputNumber, Voltages): #InputNumber=# of input photons= should average to about 1000
    #Input into mth mode of beamsplitter
    #phi1_true=a1_true+b1_true*Voltages[i]**2 #phi=a+bV**2
    phi1_true=a1_true+b1_true*V**2 #phi=a+bV**2    
    #phi2_true=a2_true+b2_true*Voltages[i]**2 #phi=a+bV**2
    phi2_true=a2_true+b2_true*V**2 #phi=a+bV**2
    U_true=ConstructU(eta1_true,eta2_true,eta3_true,phi1_true,phi2_true) #Generate double MZI Unitary
    P_click1_true=abs(top_bra@U_true@top_ket)**2 #Probability of click in top
    P_click1_true=P_click1_true[0][0]
    P_click2_true=abs(bottom_bra@U_true@top_ket)**2 #Probability of click in bottom
    P_click2_true=P_click2_true[0][0]
    P_true=[P_click1_true,P_click2_true]
    #n=C,p=P,x=array of clicks
    data=scipy.stats.multinomial.rvs(n=InputNumber,p=P_true)
    C=np.sum(data)

    return data,C

data,C=DataGen(InputNumber=1000,Voltages=V)
#print(np.shape(data))
#print(data) #Correct
#print(C)

def Likelihood(eta1,eta2,eta3,a1,a2,b1,b2,Voltages):
    #To be called after data generation
    phi1=a1+b1*V**2 #phi=a+bV**2
    phi2=a2+b2*V**2 #phi=a+bV**2
    U=ConstructU(eta1,eta2,eta3,phi1,phi2) #Generate double MZI Unitary
    P_click1=np.abs(top_bra@U@top_ket)**2 #Probability of click in top
    P_click1=P_click1[0][0]
    P_click2=np.abs(bottom_bra@U@top_ket)**2 #Probability of click in bottom
    P_click2=P_click2[0][0]
    P=np.array([P_click1.eval(),P_click2.eval()])
    #P[i]=[P_click1.eval(),P_click2.eval()]
    #n=C,p=P,x=array of clicks
    #prob[i]=scipy.stats.multinomial.pmf(x=data[i],n=C[i],p=P)
    #print(np.sum(prob))
    #return prob
    #print(P)
    return P

# set the PyMC3 model
model = pm.Model()

with model:
    # set prior parameters
    amin = -np.pi # lower range of uniform distribution on a
    amax = np.pi  # upper range of uniform distribution on a
    
    etamu = 0.5     # mean of Gaussian distribution on eta
    etasigma = np.pi/200 # standard deviation of Gaussian distribution on eta
    
    bmu = 0.7     # mean of Gaussian distribution on b
    bsigma = 0.7 # standard deviation of Gaussian distribution on b
    
    # set priors for unknown parameters
    a1model = pm.Uniform('a1', lower=amin, upper=amax) # uniform prior on y-intercept
    eta1model = pm.Normal('eta1', mu=etamu, sigma=etasigma)       # Gaussian prior on gradient
    b1model = pm.Normal('b1', mu=bmu, sigma=bsigma)
    a2model = pm.Uniform('a2', lower=amin, upper=amax) # uniform prior on y-intercept
    eta2model = pm.Normal('eta2', mu=etamu, sigma=etasigma)       # Gaussian prior on gradient
    b2model = pm.Normal('b2', mu=bmu, sigma=bsigma)
    eta3model = pm.Normal('eta3', mu=etamu, sigma=etasigma)       # Gaussian prior on gradient
    #sigmamodel = sigma # set a single standard deviation
    
    # Expected value of outcome, aka "the model"
    #mu = mmodel*x + cmodel
    switchpoint=Likelihood(eta1model,eta2model,eta3model,a1model,a2model,b1model,b2model,2.5)
    switchpoint=A.as_tensor_variable(switchpoint)
    
    P=pm.Deterministic('P', Likelihood(eta1model,eta2model,eta3model,a1model,a2model,b1model,b2model,2.5))
    # Gaussian likelihood (sampling distribution) of observations, "data"
    #Y_obs will be my observed clicks distribution amongst modes
    #noise sigma will be poissonian?
    Y_obs = pm.Bernoulli('Y_obs', n=C, p=P, observed=data)
    
    pm.sample()