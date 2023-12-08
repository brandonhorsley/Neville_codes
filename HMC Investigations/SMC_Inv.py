"""
Experimenting with using SMC for sampling since apparently it is more resistant to multimodality.
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

def ConstructU(eta,phi):
    U=construct_BS(eta)@construct_PS(phi)
    return U

#####################Data generation
I=[2,500,50,50,500,100,100,100000]
Vmax=10
N=50 #Top of page 108 ->N=number of experiments
M=2 #Number of modes
V_dist=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5
#V=V_dist+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
V=V_dist

#a_true=0
#b_true=0.788
#eta_true=0.447
eta_true=np.random.normal(loc=0.5,scale=0.05)
#a_true=np.random.uniform(low=-np.pi,high=np.pi)
#altering a_true to be more reasonably generated
a_true=np.random.normal(loc=0,scale=np.pi/200)
b_true=np.random.normal(loc=0.7,scale=0.07)


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

    for i in range(N):
        #Input into mth mode of beamsplitter
        phi_true=a_true+b_true*Voltages[i]**2 #phi=a+bV**2
        U_true=ConstructU(eta_true,phi_true) #Generate double MZI Unitary
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
#print(data) #Correct
#print(C)
#print(P)

L_dev=4E-9 #delta L, chosen so it coincides with Alex's sd to be ~np.pi/200
wv=1550E-9
a_dev=(2*np.pi*L_dev)/wv
with pm.Model() as model_multinomial:
    # Define priors
    #eta = pm.Normal("eta", mu=0.5, sigma=0.05,initval=0.5)
    eta=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta_true,transform=None)
    #a= pm.Uniform("a", lower=-np.pi, upper=np.pi,initval=a_true)
    #a= pm.TruncatedNormal("a", mu=a_est, sigma=np.pi/200,lower=-np.pi,upper=np.pi,initval=a_true)
    #Centering on zero because that would be the expected design is to have no delta L in actual chip
    a= pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a_true,transform=None)
    b= pm.Normal("b", mu=b_est, sigma=0.07,initval=b_true,transform=None)
    
    Volt=pm.Deterministic("Volt",pt.as_tensor(V_dist))
    phi=pm.Deterministic("phi",(a+b*pm.math.sqr(Volt)))
    
    p1=pm.Deterministic("p1",pm.math.sqrt(eta)*pm.math.cos(phi/2))
    p2=pm.Deterministic("p2",pm.math.sqrt(eta)*pm.math.sin(phi/2))
    p=pm.Deterministic("p",pm.math.sqr(pm.math.abs(p1))+pm.math.sqr(pm.math.abs(p2)))
    pinv=pm.Deterministic("pinv",1-p)
    P=pm.Deterministic("P",pm.math.stack([p,pinv],axis=-1))
    
    likelihood=pm.Multinomial("likelihood",n=C,p=P,shape=(N,M),observed=data)


with model_multinomial:
    trace_g=pm.sample_smc(2000)
    #trace_multinomial = pm.sample(draws=int(5e3), step=step, chains=4, cores=1,return_inferencedata=True)
    #trace_multinomial = pm.sample(draws=int(5e3), step=[step1,step2,step3],chains=4, cores=1,return_inferencedata=True)
    prior = pm.sample_prior_predictive()
    


lines={"eta":eta_true,"a":a_true,"b":b_true}
#note that you have to specify var_names otherwise it will plot all the deterministic nodes as well and screw up plot ordering
#ax=az.plot_trace(data=trace_multinomial,var_names=["eta","a","b"])
ax=az.plot_trace(data=trace_g,var_names=["eta","a","b"])
#ax[0,0].axvline(x=eta1_true)
for i,(k,v) in enumerate(lines.items()):
    ax[i,0].axvline(x=v,c="red")
    ax[i,1].axhline(y=v,c="red")
#az.plot_posterior(idata)
#az.summary(idata, round_to=2)
#az.plot_ess(idata)

#note that you have to specify var_names otherwise it will plot all the deterministic nodes as well and screw up plot ordering
#az.plot_trace(data=trace_multinomial,var_names=["eta","a","b"])
