"""
Algorithm 6 in Alex Neville thesis. Metropolised Independence Sampling (MIS) within Gibbs algorithm. Middle of page 98.

1. Label the current state vec(p_α).
2. For each component p_i ∈ vec(p_α):
   2.1. Propose a new state vec(p)' with components p'_α,k = p_α,k for all k not equal to i and random
        p'_α,i ~ unif(-π,π).
   2.2. Accept the proposed state and transition from vec(p_α) to vec(p'_α) with probability
        T(vec(p_α)|vec(p'_α)) = min(1,(P(D|vec(p_α),V)P(vec(p_α)))/(P(D|vec(p'_α),V)P(vec(p'_α))
3. Go to 1
"""

#I am not sure this procedure makes sense since the eta values shouldn't reasonably be sampled between -pi and pi,
# since those negative values would lead to unphysical answers?
#Well actually maybe that gets addressed since such answers would have P=0 and so have zero transition probability


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
N_iters=1000

eta1_arr=[p[0]]
eta2_arr=[p[1]]
eta3_arr=[p[2]]
a1_arr=[p[3]]
a2_arr=[p[4]]
b1_arr=[p[5]]
b2_arr=[p[6]]

for n in range(N_iters):
   for i in range(len(p)):
      if i in [0,1,2]: #If it is eta's
         new_element=np.random.uniform(low=-np.pi,high=np.pi)
         if new_element>=0 and new_element<=1: #Check if value is legitimate
               p_prime=p
               p_prime[i]=new_element
               #Now i need to do the acceptance probability bit
               #Likelihood
               L1=Likelihood(p,V)
               L2=Likelihood(p_prime,V)
               #Priors
               #eta: mu=0.5,sigma=0.05
               #a: uniform so N/A
               #b: mu=0.7,sigma=0.07
               P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
               P2= normal(p_prime[0],0.5,0.05)* normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
               #numerator=L1*P1*g1
               #denominator=L2*P2*g2

               #print(np.exp(L1)) #This is an array, how do i broadcast?
               #print(P1)
               #print(g1)
               #numerator=np.exp(L1)*P1*g1
               #denominator=np.exp(L2)*P2*g2
               #elem=numerator/denominator
               elem=(np.exp(L1)*P1)/(np.exp(L2)*P2)
               T=min(1,elem)
               move_prob=random_coin(T)
               if move_prob:
                  p=p_prime
         else:
               pass
      if i in [3,4]: #If it is a's
         new_element=np.random.uniform(low=-np.pi,high=np.pi)
         p_prime=p
         p_prime[i]=new_element
         #Now i need to do the acceptance probability bit
         #Likelihood
         L1=Likelihood(p,V)
         L2=Likelihood(p_prime,V)
         #Priors
         #eta: mu=0.5,sigma=0.05
         #a: uniform so N/A
         #b: mu=0.7,sigma=0.07
         P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
         P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
         numerator=L1*P1
         denominator=L2*P2
         elem=numerator/denominator
         T=min(1,elem)
         move_prob=random_coin(T)
         if move_prob:
               p=p_prime
      if i in [5,6]: #If it is b's
         new_element=np.random.uniform(low=-np.pi,high=np.pi)
         p_prime=p
         p_prime[i]=new_element
         #Now i need to do the acceptance probability bit
         #Likelihood
         L1=Likelihood(p,V)
         L2=Likelihood(p_prime,V)
         #Priors
         #eta: mu=0.5,sigma=0.05
         #a: uniform so N/A
         #b: mu=0.7,sigma=0.07
         P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
         P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
         numerator=L1*P1
         denominator=L2*P2
         elem=numerator/denominator
         T=min(1,elem)
         move_prob=random_coin(T)
         if move_prob:
               p=p_prime

   #print(p)
   eta1_arr.append(p[0])
   eta2_arr.append(p[1])
   eta3_arr.append(p[2])
   a1_arr.append(p[3])
   a2_arr.append(p[4])
   b1_arr.append(p[5])
   b2_arr.append(p[6])

   #Method is working?
#plt.hist(b1_arr,normed=1,bins=20) #Showing sampling
plt.plot(eta1_arr)