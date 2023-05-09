"""
Code for replicating fig 4.6 in Alex Neville thesis.

Plot doesn't fully match the thesis figure but it is my assumption that it 
is due to the choice of voltage sampling.
"""

#Import modules
import numpy as np
import scipy
import matplotlib.pyplot as plt

a1_true=0 #true value mentioned in figure caption
a2_true=0 #true value mentioned in figure caption

b1=0.788
b2=0.711

eta1=0.447
eta2=0.548
eta3=0.479

#For reproducibility
np.random.seed(0)

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

Voltages=np.random.uniform(low=0, high=10,size=100) #random voltage between 1 and 5

m=2 #number of modes in interferometer

a1=np.linspace(-np.pi,np.pi,100)
a2=np.linspace(-np.pi,np.pi,100)

data=np.empty((len(Voltages),2))
C=np.empty((len(Voltages)))

for i in range(len(Voltages)):
    #Input into mth mode of beamsplitter
    phi1_true=b1*Voltages[i]**2 #phi=a+bV**2
    phi2_true=b2*Voltages[i]**2 #phi=a+bV**2
    U=ConstructU(eta1,eta2,eta3,phi1_true,phi2_true) #Generate double MZI Unitary
    P_click1_true=abs(top_bra@U@top_ket)**2 #Probability of click in top
    P_click1_true=P_click1_true[0][0]
    P_click2_true=abs(bottom_bra@U@top_ket)**2 #Probability of click in bottom
    P_click2_true=P_click2_true[0][0]
    P_true=[P_click1_true,P_click2_true]
    #n=C,p=P,x=array of clicks
    data[i]=scipy.stats.multinomial.rvs(n=50,p=P_true)
    C[i]=np.sum(data[i])

results=np.empty((len(a1),len(a2)))

for i in range(len(a1)):
    for j in range(len(a2)):
        res=[]
        for k in range(len(Voltages)):
            phi1=a1[i]+b1*Voltages[k]**2 #phi=a+bV**2
            phi2=a2[j]+b2*Voltages[k]**2 #phi=a+bV**2
            U=ConstructU(eta1,eta2,eta3,phi1,phi2) #Generate double MZI Unitary
            P_click1=abs(top_bra@U@top_ket)**2 #Probability of click in top
            P_click1=P_click1[0][0]
            P_click2=abs(bottom_bra@U@bottom_ket)**2 #Probability of click in bottom
            P_click2=P_click2[0][0]
            P=[P_click1,P_click2]
            #n=C,p=P,x=array of clicks
            prob=scipy.stats.multinomial.pmf(x=data[k],n=C[k],p=P)
            res.append(np.log(prob))
        results[i][j]=np.sum(res)
        
fig,ax=plt.subplots()
im = ax.imshow(results, cmap='turbo', interpolation='nearest', extent=[-np.pi,np.pi,-np.pi,np.pi])
fig.colorbar(im, ax=ax)
plt.xlabel('a1')
plt.ylabel('a2')
plt.show()