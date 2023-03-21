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
    #print(phi1_item)
    #print(phi2_item)
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
    #print(max_value_indices) #Format: (array([53], dtype=int64), array([15], dtype=int64))
    #print(max_value_indices[0][0])
    #print(max_value_indices[1][0])
    #print(max_value) #All give very high values
    #print(phi1_prime[max_value_indices[0][0]])
    #print(phi2_prime[max_value_indices[0][0]])
    #testU=construct_BS(eta3_flaw)@construct_PS(phi2_prime[])@construct_BS(eta2_flaw)@construct_PS(phi1_prime[i])@construct_BS(eta1_flaw)
    return phi1_prime[max_value_indices[0][0]],phi2_prime[max_value_indices[1][0]]


results=np.empty((len(phi1),len(phi2)))
#trace_dist=np.empty((len(phi1),len(phi2)))


for i in range(len(phi1)):
    for j in range(len(phi2)):
        x,y=Optimise(phi1[i],phi2[j])
        U1=construct_BS(eta3)@construct_PS(phi2[j])@construct_BS(eta2)@construct_PS(phi1[i])@construct_BS(eta1)
        U2=construct_BS(eta3_flaw)@construct_PS(y)@construct_BS(eta2_flaw)@construct_PS(x)@construct_BS(eta1_flaw)
        rho1=DensityMatrix(U1)
        rho2=DensityMatrix(U2)
        #trace_dist[i][j]=0.5*np.trace(sqrtm((rho1-rho2).conj().T@(rho1-rho2)))
        results[i][j]=(np.trace(sqrtm(sqrtm(rho1)@rho2@sqrtm(rho1))))**2
        print(results[i][j])     

#print(results)
fig,ax=plt.subplots()
im = ax.imshow(results, cmap='turbo', interpolation='nearest', extent=[0,2*np.pi,0,2*np.pi])
#im = ax.imshow(trace_dist, cmap='turbo', interpolation='nearest', extent=[0,2*np.pi,0,2*np.pi])
fig.colorbar(im, ax=ax)
plt.xlabel(r'$\phi_1$')
plt.ylabel(r'$\phi_2$')
plt.show()


"""
Plot really doesn't look right, took a while too so i will have to perhaps downsize 
my arrays and fix the plot by seeing where my current code has gone wrong. 
Max_value gives high values like expected but fidelity seems to report the faulty
graph. 

I will do a test block for (2,2) to see if i am supposed to get zero for
fidelity like the plot is showing.Test code block gives 0.845. Which simultaneously 
isn't what the plot shows from the color bar but also isn't what is seen in 
Alex's thesis. Maybe two things are going on, one is that my phi value retrieval is
faulty and that i need multiple layers of iteration? Multiple iteration loops
doesn't sound right.

I made a correction to the optimisation function but that still doesn't seem to have
worked. I am now trying to use the trace distance measure to see if the error is 
how i am obtaining the fidelity again. 

Trace distance proved unsuccessful. Gives a similar plot to fidelity. NExt step
is to just check functionality and really understand what my code is actually
doing and perhaps the error lies in that.
"""