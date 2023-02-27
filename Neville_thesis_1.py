"""
Code for reproducing Fig 4.2 of Alex Neville thesis
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

results=np.empty((len(phi1),len(phi2)))
#a=np.array([0,1])
a=np.array([1,0])
a.shape=(2,1)

#b=np.array([0,1])
b=np.array([1,0])
b.shape=(1,2)

#test_U=construct_BS(eta3)@construct_PS(np.pi/2)@construct_BS(eta2)@construct_PS(np.pi/2)@construct_BS(eta1)
#print(test_U)

for i in range(len(phi1)):
    for j in range(len(phi2)):
        unitary_toy=construct_BS(eta3)@construct_PS(phi2[j])@construct_BS(eta2)@construct_PS(phi1[i])@construct_BS(eta1)
        #print(np.shape(a))
        #print(np.shape(b))
        #print(np.shape(unitary_toy))
        results[i][j]=abs(b@unitary_toy@a)**2
        
#print(results)
#results=results.T
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
Results have the same form but the opposite way round, so at (pi/2,pi/2) it is zero 
instead of 1. Flipping the fock state arrays from 0,1 to 1,0 has no effect and not does
flipping around the minus sign in the exponential terms of the phase shifter unitary.

So the only remaining outcome is to just calculate it by hand.

Pen and paper calculations as well as a test calculation for the (pi/2,pi/2) coordinate 
confirms it, so i will move on under that premise.
"""