"""
This code is for reproducing Figure 4.3b in Alex Neville thesis.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm

#Data generation
phi1=np.linspace(0,2*np.pi,100)
phi2=phi1

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

results=np.empty((len(phi1),len(phi2)))

for i in range(len(phi1)):
    for j in range(len(phi2)):
        unitary_toy1=construct_BS(eta3)@construct_PS(phi2[j])@construct_BS(eta2)@construct_PS(phi1[i])@construct_BS(eta1)
        unitary_toy2=construct_BS(eta3_flaw)@construct_PS(phi2[j])@construct_BS(eta2_flaw)@construct_PS(phi1[i])@construct_BS(eta1_flaw)
        rho1=DensityMatrix(unitary_toy1)
        rho2=DensityMatrix(unitary_toy2)
        trace_dist[i][j]=0.5*np.trace(sqrtm((rho1-rho2).conj().T@(rho1-rho2)))
        results[i][j]=(np.trace(sqrtm(sqrtm(rho1)@rho2@sqrtm(rho1))))**2
        

fig,ax=plt.subplots()
im = ax.imshow(results, cmap='turbo', interpolation='nearest', extent=[0,2*np.pi,0,2*np.pi])
fig.colorbar(im, ax=ax)
plt.xlabel(r'$\phi_1$')
plt.ylabel(r'$\phi_2$')
plt.show()

"""
Figure successfully replicated.
"""