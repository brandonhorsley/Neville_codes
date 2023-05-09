"""
Will attempt to modify this code to change the Voltage array to reflect what it says in the algorithm
"""

import arviz as az
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc import Model, Normal, Uniform,Multinomial,Dirichlet,sample

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

    for i in range(N):
        #Input into mth mode of beamsplitter
        phi1_true=a1_true+b1_true*Voltages[i]**2 #phi=a+bV**2
        phi2_true=a2_true+b2_true*Voltages[i]**2 #phi=a+bV**2
        U_true=ConstructU(eta1_true,eta2_true,eta3_true,phi1_true,phi2_true) #Generate double MZI Unitary
        P_click1_true=abs(top_bra@U_true@bottom_ket)**2 #Probability of click in top
        P_click1_true=P_click1_true[0][0]
        P_click2_true=abs(bottom_bra@U_true@bottom_ket)**2 #Probability of click in bottom
        P_click2_true=P_click2_true[0][0]
        P_true=[P_click1_true,P_click2_true]
        #n=C,p=P,x=array of clicks
        data[i]=scipy.stats.multinomial.rvs(n=InputNumber,p=P_true)
        C[i]=np.sum(data[i])

    return data,C

data,C=DataGen(InputNumber=1000,Voltages=V,poissonian=False)
#print(np.shape(data))
print(data) #Correct
print("#################################################################")
print(C)
print("#################################################################")

############################################################################################################
#Below code copied from: https://www.pymc.io/projects/examples/en/latest/case_studies/blackbox_external_likelihood_numpy.html
##############################################################################################################


def my_model(theta, x):
    m, c = theta
    return m * x + c

def my_model2(theta,Voltages):
    #To be called after data generation
    eta1,eta2,eta3,a1,a2,b1,b2=theta
    prob=np.empty(len(Voltages))
    for i in range(len(Voltages)):
        phi1=a1+b1*Voltages[i]**2 #phi=a+bV**2
        phi2=a2+b2*Voltages[i]**2 #phi=a+bV**2
        U=ConstructU(eta1,eta2,eta3,phi1,phi2) #Generate double MZI Unitary
        P_click1=abs(top_bra@U@top_ket)**2 #Probability of click in top
        P_click1=P_click1[0][0]
        P_click2=abs(bottom_bra@U@bottom_ket)**2 #Probability of click in bottom
        P_click2=P_click2[0][0]
        P=np.array([P_click1.eval(),P_click2.eval()])
        #n=C,p=P,x=array of clicks
        print("likelihood start")
        print(data[i])
        print(C[i])
        print(P)
        prob[i]=scipy.stats.multinomial.pmf(x=data[i],n=C[i],p=P)
        return prob
    

#Calling this won't work hence why we call the pytensor op below
def my_loglike(theta, x, data, sigma):
    model = my_model(theta, x)
    return -(0.5 / sigma**2) * np.sum((data - model) ** 2)

def my_loglike2(theta,Voltages):
    model=my_model2(theta,Voltages)
    return np.log(model)

# define a pytensor Op for our likelihood function
class LogLike(pt.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, Voltages):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """
        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.Voltages=Voltages

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables
        # call the log-likelihood function
        logl = self.likelihood(theta, Voltages)
        outputs[0][0] = np.array(logl)  # output the log-likelihood

# set up our data
#N = 10  # number of data points
#sigma = 1.0  # standard deviation of noise
#x = np.linspace(0.0, 9.0, N)

#mtrue = 0.4  # true gradient
#ctrue = 3.0  # true y-intercept

#truemodel = my_model([mtrue, ctrue], x)

# make data
#rng = np.random.default_rng(716743)
#data = sigma * rng.normal(size=N) + truemodel

# create our Op
#logl = LogLike(my_loglike, data, x, sigma)
logl = LogLike(my_loglike2,data,V)
# use PyMC to sampler from log-likelihood
with pm.Model():
    # uniform priors on m and c
    #m = pm.Uniform("m", lower=-10.0, upper=10.0)
    #c = pm.Uniform("c", lower=-10.0, upper=10.0)
    # Define priors
    eta1 = Normal("eta1", mu=0.5, sigma=0.05,initval=0.5)
    eta2 = Normal("eta2", mu=0.5, sigma=0.05,initval=0.5)
    eta3= Normal("eta3", mu=0.5, sigma=0.05,initval=0.5)
    a1= Uniform("a1", lower=-np.pi, upper=np.pi,initval=0)
    a2= Uniform("a2", lower=-np.pi, upper=np.pi,initval=0)
    b1= Normal("b1", mu=0.7, sigma=0.07,initval=0.5)
    b2= Normal("b2", mu=0.7, sigma=0.07,initval=0.5)
    #Voltages=Uniform("Voltages",0,Vmax)

    # convert m and c to a tensor vector
    #theta = pt.as_tensor_variable([m, c])
    theta=pt.as_tensor_variable([eta1,eta2,eta3,a1,a2,b1,b2])
    # use a Potential to "call" the Op and include it in the logp computation
    pm.Potential("likelihood", logl(theta,V))

    # Use custom number of draws to replace the HMC based defaults
    idata_mh = pm.sample(3000, tune=1000)

# plot the traces
#az.plot_trace(idata_mh, lines=[("m", {}, mtrue), ("c", {}, ctrue)]);
az.plot_trace(idata_mh)

"""
def Likelihood(eta1,eta2,eta3,a1,a2,b1,b2,Voltages):
    #To be called after data generation
    
    prob=np.empty(len(Voltages))
    for i in range(len(Voltages)):
        phi1=a1+b1*Voltages[i]**2 #phi=a+bV**2
        phi2=a2+b2*Voltages[i]**2 #phi=a+bV**2
        U=ConstructU(eta1,eta2,eta3,phi1,phi2) #Generate double MZI Unitary
        P_click1=abs(top_bra@U@top_ket)**2 #Probability of click in top
        P_click1=P_click1[0][0]
        P_click2=abs(bottom_bra@U@bottom_ket)**2 #Probability of click in bottom
        P_click2=P_click2[0][0]
        P=np.array([P_click1.eval(),P_click2.eval()])
        #n=C,p=P,x=array of clicks
        print("likelihood start")
        print(data[i])
        print(C[i])
        print(P)
        prob[i]=scipy.stats.multinomial.pmf(x=data[i],n=C[i],p=P)
        return prob

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
    #Voltages=Uniform("Voltages",0,Vmax)
    counts = Multinomial("counts", n=C, p=Likelihood(eta1,eta2,eta3,a1,a2,b1,b2,V), observed=data)
    trace_multinomial = sample(draws=int(1e5), step=pm.metropolis(), return_inferencedata=True,cores=1)

###################Bayesian analysis
az.plot_trace(data=trace_multinomial)

"""