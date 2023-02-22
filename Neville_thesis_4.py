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
#print(phi1)

eta1=0.5
eta2=0.5
eta3=0.5
#Define formula for Phase shifter unitary, will define just dim=2 for now
def construct_PS(phi):
    mat=np.array(([np.exp(1j*phi/2),0],[0,np.exp(-1j*phi/2)]))
    return mat



def construct_BS(eta):
    mat=np.array(([np.sqrt(eta), 1j*np.sqrt(1-eta)],[1j*np.sqrt(1-eta),np.sqrt(eta)]))
    return mat

"""
#a=np.array([0,1])
a=np.array([1,0])
a.shape=(2,1)

#b=np.array([0,1])
b=np.array([1,0])
b.shape=(1,2)
"""

top_ket=np.array([1,0])
top_ket.shape=(2,1)

top_bra=np.array([1,0])
top_bra.shape=(1,2)

bottom_ket=np.array([0,1])
bottom_ket.shape=(2,1)

bottom_bra=np.array([0,1])
bottom_bra.shape=(1,2)

def Fidelity(Unitary):
    p_top=abs(top_bra@Unitary@top_ket)**2
    #p_top=top_bra@Unitary@top_ket
    p_bottom=abs(bottom_bra@Unitary@top_ket)**2
    #p_bottom=bottom_bra@Unitary@top_ket
    rho1=top_ket@top_bra
    #rho2=p_top*(Unitary@top_ket@top_bra@Unitary.conj().T)+p_bottom*(Unitary@bottom_ket@top_bra@Unitary.conj().T)
    rho2=p_top*(Unitary@top_ket@top_bra@Unitary.conj().T)
    fid=(np.trace(sqrtm(sqrtm(rho1)@rho2@sqrtm(rho1))))**2
    return fid


#Fidelity can be defined as overlap between two states
#https://strawberryfields.ai/photonics/demos/run_state_learner.html

#This proved less successful


results=np.empty((len(phi1),len(phi2)))


#test_U=construct_BS(eta3)@construct_PS(np.pi/2)@construct_BS(eta2)@construct_PS(np.pi/2)@construct_BS(eta1)
#print(test_U)

for i in range(len(phi1)):
    for j in range(len(phi2)):
        unitary_toy=construct_BS(eta3)@construct_PS(phi2[j])@construct_BS(eta2)@construct_PS(phi1[i])@construct_BS(eta1)
        results[i][j]=Fidelity(unitary_toy)
        #results[i][j]=abs(top_bra@unitary_toy.conj().T@top_ket)**2
        
#print(results)
fig,ax=plt.subplots()
im = ax.imshow(results, cmap='turbo', interpolation='nearest', extent=[0,2*np.pi,0,2*np.pi])
fig.colorbar(im, ax=ax)
#Show all ticks and label them with the respective list entries
#ax.set_xticks(np.arange(len(phi2)))
#ax.set_yticks(np.arange(len(phi1)))
#ax.xaxis.set_major_locator(plt.MaxNLocator(3))

#plt.imshow(results, cmap='hot', interpolation='nearest', extent=[0,2*np.pi,0,2*np.pi])
plt.xlabel(r'$\phi_1$')
plt.ylabel(r'$\phi_2$')
plt.show()

"""
Not seeming to be able to replicate the fig 4.3b plot.
"""