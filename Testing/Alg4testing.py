import numpy as np
import time
import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '/Users/ck21395/PhD codes/Neville_codes')
from Aux_Nev import *

b_est=0.07 
"""
b_est is subject to change and consistency w. Aux_Nev should be certified, additionally sigma_i for b is 
suggested it should be b_est but the actual estimated value is closer to 0.7 so there is confusion there.
"""

eta_sigma=0.005
a_sigma=np.pi/200
b_sigma=b_est #Based around true values from Neville_thesis_8.py
N_iters=100000

I=[2,500,50,50,500,100,100,100000] #Determines iteration number for each algorithm call
#I=[2,500,50,50,500,100,100,100]
p=[0.5,0.5,0.5,0,0,0.7,0.7]

def normal(x,mu,sigma):
    numerator = np.exp((-(x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator

def uniform(x):
    return 1/(2*np.pi)

def random_coin(p):
    unif = np.random.uniform(0,1)
    if unif<=p:
        return True
    else:
        return False

def Alg4(p,Niters,Markov=False,ReturnAll=False):
    """
    This Algorithm is the Metropolis-Hastings within Gibbs sampling algorithm that is 
    described on the middle of page 94 in Alex Neville's thesis.
    """
    if Markov: #i.e. if Markov==True
        MCMC=Alg4(p,Niters, Markov=False, ReturnAll=True)
        return MCMC
    else: #If Markov==False
        MCMC=[]
        for n in range(Niters):
            for i in range(len(p)):
                if i in [0,1,2]: #If it is eta's
                    new_element=np.random.normal(loc=p[i],scale=eta_sigma) #draw random sample from proposal distribution
                    p_prime=list(p)
                    p_prime[i]=new_element
                    #Likelihood
                    L1=Likelihood(p,V)
                    L2=Likelihood(p_prime,V)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    #P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
                    P1=normal(p[i],0.5,0.05)
                    #P2= normal(p_prime[0],0.5,0.05)* normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                    P2=normal(p_prime[i],0.5,0.05)
                    #Candidates
                    g1= np.random.normal(p[i],eta_sigma)
                    g2=np.random.normal(p_prime[i],eta_sigma)
                    elem=(np.exp(L1)*P1*g1)/(np.exp(L2)*P2*g2)
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p=p_prime
                    #print("eta done")
                if i in [3,4]: #If it is a's
                    new_element=np.random.normal(loc=p[i],scale=a_sigma) #draw random sample from proposal distribution
                    p_prime=list(p)
                    p_prime[i]=new_element
                    #Likelihood
                    L1=Likelihood(p,V)
                    L2=Likelihood(p_prime,V)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    #P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
                    P1=uniform(p[i])
                    #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                    P2=uniform(p_prime[i])
                    #Candidates
                    g1= np.random.normal(p[i],a_sigma)
                    g2=np.random.normal(p_prime[i],a_sigma)
                    numerator=L1*P1*g1
                    denominator=L2*P2*g2
                    elem=numerator/denominator
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p=p_prime
                    #print("a done")
                if i in [5,6]: #If it is b's
                    new_element=np.random.normal(loc=p[i],scale=b_sigma) #draw random sample from proposal distribution
                    p_prime=list(p)
                    p_prime[i]=new_element
                    #Likelihood
                    L1=Likelihood(p,V)
                    L2=Likelihood(p_prime,V)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    #P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
                    P1=normal(p[i],0.7,0.07)
                    #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                    P2=normal(p_prime[i],0.7,0.07)
                    #Candidates
                    g1= np.random.normal(p[i],b_sigma) 
                    g2=np.random.normal(p_prime[i],b_sigma)
                    numerator=L1*P1*g1
                    denominator=L2*P2*g2
                    elem=numerator/denominator
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p=p_prime
                    #print("b done")
            if ReturnAll:
                MCMC.append(p)

        if ReturnAll:
            return MCMC
        else:
            return p

start=time.time()
Alg4(p,10,Markov=True) 
#1 iteration takes 0.08s,10 takes 0.8 so 100,000 should take ~10,000s=~1667min=~27 hours=~1.15 days
# Numba tend to boast an order or two orders of magnitude speedup in general (with caveats ofc)
# 1 order of magnitude:  ~1000s=~16-17mins
# 2 orders of magnitude: ~100s=~1-2 mins
end=time.time()

print(end-start)