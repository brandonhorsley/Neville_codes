"""
This code will be for implementing a custom sampling method akin to what 
Alex Neville does in his thesis in order to converge to a good 'a' and then 
a good 'a' and 'b'. Unfortunately raw HMC does not seem to meet this end and 
so we get solid eta estimates while 'a' and 'b' estimates are rubbish. Alex 
implements a hierarchical approach of using sampling to determine alpha model 
(only changing 'a') then determine beta model (starting from 'a' values from 
end of alpha model, then change 'a' and 'b' together) before finally changing 
all eta's,a's and b's together.

Active notes:
Copied over code from Neville_MZI.py.

Custom sampling still not converging so it could either be multimodal posterior 
(most samplers including HMC doesn't cooperate well with that, could give SMC a 
go) or just the fact that a uniform prior apparently imposes a nasty geometry to 
sample from (https://discourse.pymc.io/t/nuts-sampler-converges-to-wrong-value/5733/5). 
Changing 'a' to a TruncatedNormal works well, but i'm keen to still act in good 
faith with prior choice. So have 3 paths: test SMC sampler, if a and b can be roughly 
estimated in design then better prior choice is good, or reparameterise (however i'm 
not sure on how i would do that).
 - Revisiting Neville thesis points to a arises due to standard deviation in path length, and so increasing sd of normal distribution with a modulo of 2pi means that past a threshold the normal distribution becomes a uniform distribution. He notes that if the standard deviation of the change in length distribution is ~λ/2, where λ is the wavelength of the light, then a can be treated as a uniformly random variable on the interval [−π, π]. So then how do we probe the standard deviation of the length, and what wavelength of light do we use, this can be incorporated into the code. https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-8-10456&id=408126 gives the reasoning for phi=a+bV^2, since phi prop to L and L prop to R which is prop to V2 so phi =bV^2 and a is just error term from delta L (fabrication error)
 - Nearing the end of my investigation into thermo-optic phase shifters and what we can get away with. 'a' is essentially a term for imperfections in waveguide lengths and so if it was sufficiently small and sufficiently abundant phase shifters (to avoid waveguide imperfection length aggregation), so to be fully general uniform is the best course of action, so if we can find a better solution, then that would be ideal but we still may be able to get away with truncated normal with sd governed by average delta L from fabrication and if we have sufficiently many phase shifters like in a classic Reck/Clements scheme. But even then we should still address the multimodality.
 - Changed how 'a' was being generated to truncated normal and changed prior accordingly. Results show that quality results.
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
    eta=pm.TruncatedNormal("eta",mu=eta_est,sigma=0.05,lower=0.0,upper=1.0,initval=eta_true)
    #a= pm.Uniform("a", lower=-np.pi, upper=np.pi,initval=a_true)
    #a= pm.TruncatedNormal("a", mu=a_est, sigma=np.pi/200,lower=-np.pi,upper=np.pi,initval=a_true)
    #Centering on zero because that would be the expected design is to have no delta L in actual chip
    a= pm.TruncatedNormal("a", mu=0, sigma=a_dev,lower=-np.pi,upper=np.pi,initval=a_true)
    b= pm.Normal("b", mu=b_est, sigma=0.07,initval=b_true)
    
    Volt=pm.Deterministic("Volt",pt.as_tensor(V_dist))
    phi=pm.Deterministic("phi",(a+b*pm.math.sqr(Volt)))
    
    p1=pm.Deterministic("p1",pm.math.sqrt(eta)*pm.math.cos(phi/2))
    p2=pm.Deterministic("p2",pm.math.sqrt(eta)*pm.math.sin(phi/2))
    p=pm.Deterministic("p",pm.math.sqr(pm.math.abs(p1))+pm.math.sqr(pm.math.abs(p2)))
    pinv=pm.Deterministic("pinv",1-p)
    P=pm.Deterministic("P",pm.math.stack([p,pinv],axis=-1))
    
    likelihood=pm.Multinomial("likelihood",n=C,p=P,shape=(N,M),observed=data)


with model_multinomial:
    #step1=pm.NUTS([a])
    #step1=pm.Metropolis([a])
    #step2=pm.NUTS([a,b])
    #step2=pm.Metropolis([a,b])
    #step3=pm.NUTS([eta,a,b])
    #step3=pm.Metropolis([eta,a,b])
    trace_multinomial = pm.sample(draws=int(5e3), chains=4, cores=1,return_inferencedata=True)
    #trace_multinomial = pm.sample(draws=int(5e3), step=[step1,step2,step3],chains=4, cores=1,return_inferencedata=True)
    prior = pm.sample_prior_predictive()
    


lines={"eta":eta_true,"a":a_true,"b":b_true}
#note that you have to specify var_names otherwise it will plot all the deterministic nodes as well and screw up plot ordering
ax=az.plot_trace(data=trace_multinomial,var_names=["eta","a","b"])
#ax[0,0].axvline(x=eta1_true)
for i,(k,v) in enumerate(lines.items()):
    ax[i,0].axvline(x=v,c="red")
    ax[i,1].axhline(y=v,c="red")
#az.plot_posterior(idata)
#az.summary(idata, round_to=2)
#az.plot_ess(idata)

#note that you have to specify var_names otherwise it will plot all the deterministic nodes as well and screw up plot ordering
#az.plot_trace(data=trace_multinomial,var_names=["eta","a","b"])
