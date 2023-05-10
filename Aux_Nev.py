"""
Auxiliary code for basic stuff for Neville experiments
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

#####################Generating Data#######################

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

def Likelihood(p,Voltages):
    eta1=p[0]
    eta2=p[1]
    eta3=p[2]
    a1=p[3]
    a2=p[4]
    b1=p[5]
    b2=p[6]
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