"""
Code block for replicating fig 4.4 from Alex Neville's thesis. 

The description of how to replicate seems pretty straightforward, it will basically just be a version 
of Neville_thesis_4.py but with the optimisation step included.

Have started by just copying over fig 4.3b code, i expect i will just define an optimise function that
performs the required optimisation.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm

#Data generation
phi1=np.linspace(0,2*np.pi,100)
phi2=np.linspace(0,2*np.pi,100)

eta1=0.5
eta2=0.5
eta3=0.5

eta1_flaw=0.485
eta2_flaw=0.400
eta3_flaw=0.629

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

def DensityMatrix(Unitary):
    rho=Unitary@top_ket@top_bra@Unitary.conj().T
    return rho

def Optimise(phi1_item,phi2_item):
    phi1_prime=np.linspace(0,2*np.pi,100)
    phi2_prime=np.linspace(0,2*np.pi,100)
    results_prime=np.empty((len(phi1_prime),len(phi2_prime)))
    #Taken below line out of nested for loop since it doesn't need
    #to be recalculated each time
    unitary_toy1=construct_BS(eta3)@construct_PS(phi2_item)@construct_BS(eta2)@construct_PS(phi1_item)@construct_BS(eta1)
    for i in range(len(phi1_prime)):
        for j in range(len(phi2_prime)):
            #unitary_toy1=construct_BS(eta3)@construct_PS(phi2_item)@construct_BS(eta2)@construct_PS(phi1_item)@construct_BS(eta1)
            unitary_toy2=construct_BS(eta3_flaw)@construct_PS(phi2_prime[j])@construct_BS(eta2_flaw)@construct_PS(phi1_prime[i])@construct_BS(eta1_flaw)
            results_prime[i][j]=abs(top_bra@unitary_toy1.conj().T@unitary_toy2@top_ket)**2
    max_value=np.max(results_prime)
    max_value_indices=np.where(results_prime==max_value)
    return phi1_prime[max_value_indices[0][0]],phi2_prime[max_value_indices[1][0]]


results=np.empty((len(phi1),len(phi2)))

for i in range(len(phi1)):
    for j in range(len(phi2)):
        x,y=Optimise(phi1[i],phi2[j])
        U1=construct_BS(eta3)@construct_PS(phi2[j])@construct_BS(eta2)@construct_PS(phi1[i])@construct_BS(eta1)
        U2=construct_BS(eta3_flaw)@construct_PS(y)@construct_BS(eta2_flaw)@construct_PS(x)@construct_BS(eta1_flaw)
        rho1=DensityMatrix(U1)
        rho2=DensityMatrix(U2)
        results[i][j]=(np.trace(sqrtm(sqrtm(rho1)@rho2@sqrtm(rho1))))**2    

fig,ax=plt.subplots()
im = ax.imshow(results, cmap='turbo', interpolation='nearest', extent=[0,2*np.pi,0,2*np.pi])
fig.colorbar(im, ax=ax)
plt.xlabel(r'$\phi_1$')
plt.ylabel(r'$\phi_2$')
plt.show()