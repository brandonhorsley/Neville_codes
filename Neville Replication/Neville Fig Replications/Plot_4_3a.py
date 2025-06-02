"""
Code for reproducing Fig 4.3a of Alex Neville thesis. This code is basically the same as
fig 4.2 but with different eta values.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#Data generation
phi1=np.linspace(0,2*np.pi,100)
phi2=phi1

eta1=0.485
eta2=0.400
eta3=0.629

#Define formula for Phase shifter unitary, will define just dim=2 for now
def construct_PS(phi):
    mat=np.array(([np.exp(1j*phi/2),0],[0,np.exp(-1j*phi/2)]))
    return mat

def construct_BS(eta):
    mat=np.array(([np.sqrt(eta), 1j*np.sqrt(1-eta)],[1j*np.sqrt(1-eta),np.sqrt(eta)]))
    return mat


results=np.empty((len(phi1),len(phi2)))
a=np.array([1,0])
a.shape=(2,1)

b=np.array([1,0])
b.shape=(1,2)


for i in range(len(phi1)):
    for j in range(len(phi2)):
        unitary_toy=construct_BS(eta3)@construct_PS(phi2[j])@construct_BS(eta2)@construct_PS(phi1[i])@construct_BS(eta1)
        results[i][j]=abs(b@unitary_toy@a)**2
        

fig,ax=plt.subplots()
im = ax.imshow(results, cmap='turbo', interpolation='nearest', extent=[0,2*np.pi,0,2*np.pi])
fig.colorbar(im, ax=ax)

plt.xlabel(r'$\phi_1$')
plt.ylabel(r'$\phi_2$')
plt.show()

"""
Plot sufficiently replicated.
"""
