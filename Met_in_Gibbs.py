"""
Algorithm 4 in Alex Neville thesis. Metropolis within Gibbs sampling algorithm. Middle of page 94.

1. Label the current state vec(p).
2. For each component p_i ∈ vec(p):
   2.1. Propose a new state vec(p)' with components p'_k = p_k for all k not equal to i and p'_α,i 
        is picked randomly from the proposal distribution g_i(vec(p')|vec(p)). 
   2.2. Accept the proposed state and transition from vec(p) to vec(p') with probability
        T(vec(p)|vec(p')) = min(1,(P(D|vec(p),V)P(vec(p))g_i(vec(p')|vec(p)))/(P(D|vec(p'),V)P(vec(p'))g_i(vec(p)|vec(p')))
3. Go to 1
"""

"""
Want to make a module of the general definitions from the main figure plots (define PS and so on)
Also want to define a function to run the transition probability bit
"""

import numpy as np
from Aux_Nev import *

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

#Define initial state p=(eta's,a's,b's)
p=[0.5,0.5,0.5,0,0,0.5,0.5]
#V=2.5

#p={"eta1": 0.5, "eta2": 0.5, "eta3": 0.5, "a1": 0, "a2": 0, "b1": 0.5, "b1": 0.5} #initial vec(p)

#eq 4.11: g_i(p',p)=Normal(p_i,sigma_i)
#sigma_i=pi/200 for a, b_est for b, 0.005 for eta
eta_sigma=0.005
a_sigma=np.pi/200
b_sigma=0.75 #Based around true values from Neville_thesis_8.py

for i in range(len(p)):
    if i in [0,1,2]: #If it is eta's
        new_element=np.random.normal(loc=p[i],scale=eta_sigma) #draw random sample from proposal distribution
        p_prime=p
        p_prime[i]=new_element
        #Now i need to do the acceptance probability bit
        L1=Likelihood(p,V)
        L2=Likelihood(p_prime,V)
        P1= normal(p[0],eta_sigma),normal(p[1],eta_sigma),normal(p[2],eta_sigma),uniform(p[3],a_sigma),uniform(p[4],a_sigma),normal(p[5],b_sigma),normal(p[6],b_sigma) #Prior for p
        P2= normal(p_prime[0],eta_sigma),normal(p_prime[1],eta_sigma),normal(p_prime[2],eta_sigma),uniform(p_prime[3],a_sigma),uniform(p_prime[4],a_sigma),normal(p_prime[5],b_sigma),normal(p_prime[6],b_sigma) #prior for p'
        g1= normal(p[i],eta_sigma) #proposal for p'|p
        #g2= g1 #proposal for p|p', g is usually assumed to be symmetric???
        g2=normal(p_prime[i],eta_sigma)
        numerator=L1*P1*g1
        denominator=L2*P2*g2
        elem=numerator/denominator
        T=min(1,elem)
        move_prob=random_coin(T)
        if move_prob:
            p=p_prime
    if i in [3,4]: #If it is a's
        new_element=np.random.normal(loc=p[i],scale=a_sigma) #draw random sample from proposal distribution
        p_prime=p
        p_prime[i]=new_element
        #Now i need to do the acceptance probability bit
        L1=Likelihood(p,V)
        L2=Likelihood(p_prime,V)
        P1= normal(p[0],eta_sigma),normal(p[1],eta_sigma),normal(p[2],eta_sigma),uniform(p[3],a_sigma),uniform(p[4],a_sigma),normal(p[5],b_sigma),normal(p[6],b_sigma) #Prior for p
        P2= normal(p_prime[0],eta_sigma),normal(p_prime[1],eta_sigma),normal(p_prime[2],eta_sigma),uniform(p_prime[3],a_sigma),uniform(p_prime[4],a_sigma),normal(p_prime[5],b_sigma),normal(p_prime[6],b_sigma) #prior for p'
        g1= normal(p[i],a_sigma) #proposal for p'|p
        #g2= g1 #proposal for p|p', g is usually assumed to be symmetric???
        g2=normal(p_prime[i],a_sigma)
        numerator=L1*P1*g1
        denominator=L2*P2*g2
        elem=numerator/denominator
        T=min(1,elem)
        move_prob=random_coin(T)
        if move_prob:
            p=p_prime
    if i in [5,6]: #If it is b's
        new_element=np.random.normal(loc=p[i],scale=b_sigma) #draw random sample from proposal distribution
        p_prime=p
        p_prime[i]=new_element
        #Now i need to do the acceptance probability bit
        L1=Likelihood(p,V)
        L2=Likelihood(p_prime,V)
        P1= normal(p[0],eta_sigma),normal(p[1],eta_sigma),normal(p[2],eta_sigma),uniform(p[3],a_sigma),uniform(p[4],a_sigma),normal(p[5],b_sigma),normal(p[6],b_sigma) #Prior for p
        P2= normal(p_prime[0],eta_sigma),normal(p_prime[1],eta_sigma),normal(p_prime[2],eta_sigma),uniform(p_prime[3],a_sigma),uniform(p_prime[4],a_sigma),normal(p_prime[5],b_sigma),normal(p_prime[6],b_sigma) #prior for p'
        g1= normal(p[i],b_sigma) #proposal for p'|p
        #g2= g1 #proposal for p|p', g is usually assumed to be symmetric???
        g2=normal(p_prime[i],b_sigma)
        numerator=L1*P1*g1
        denominator=L2*P2*g2
        elem=numerator/denominator
        T=min(1,elem)
        move_prob=random_coin(T)
        if move_prob:
            p=p_prime

print(p)


    