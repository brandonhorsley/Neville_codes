"""
This code file is for the intent of testing and verifying the functionality of the MIS subalgorithm.

For this since parameters don't need to be for the quantum case to have confidence that it works, 
although we can pivot to maybe a simple check afterwards like just a single beamsplitter estimate.

So for this i shall just use MIS to estimate mu and sigma for a normal distribution.
"""

import numpy as np

def normal(x,mu,sigma):
    numerator = np.exp((-(x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator

def uniform(x):
    return 1/(2*np.pi)

def random_coin(p):
    unif = np.random.uniform(0,1)
    if unif>=p:
        return False
    else:
        return True
#Draws from a standard normal: N(0,1)
sim_draws=np.random.standard_normal(size=100)

#N_iters=100000 #Number of iterations of algorithm
N_iters=1000 #Number of iterations of algorithm

p=[0,0.5]
mu=[p[0]]
sigma=[p[1]]

def Likelihood(p):
    res=1
    for _ in range(len(sim_draws)):
        prefactor=1/(np.sqrt(2*np.pi*p[1]**2))
        #exponential=-((sim_draws[_]-p[0])**2)/(2*sigma**2)
        exp_num=-(sim_draws[_]-p[0])**2
        exp_denom=2*p[1]**2
        exponential=exp_num/exp_denom
        element=prefactor*np.exp(exponential)
        res=1*element
    return res

for _ in range(N_iters):
    mu_new=np.random.normal(loc=mu[_])
    sigma_new=np.random.normal(loc=sigma[_])
    p_new=[mu_new,sigma_new]
    L1=Likelihood(p)
    L2=Likelihood(p_new)

    P1=normal(p[0],0,1)*normal(p[1],1,1)
    P2=normal(p_new[0],0,1)*normal(p_new[1],1,1)

    elem=(L1*P1)/(L2*P2)
    T=min(1,elem)
    move_prob=random_coin(T)
    if move_prob:
         p=p_new
    mu.append(p[0])
    sigma.append(p[1])

import matplotlib.pyplot as plt

plt.plot(mu)
plt.plot(sigma)
plt.show()