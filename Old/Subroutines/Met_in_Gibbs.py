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

Need to rework transition probability fraction to account for the fact that the likelihood needs to be multiplied 
but to avoid it getting too small i should divide each likelihood product term by the other on the denominator

error primarily seems to lie in numbers being the same but also priors getting to be too small that they are 
rounded to zero so elem becomes nan.This is also true of likelihood. This is mentioned more in the thesis to say 
about how poorly it performs for larger parameter spaces so it only really gets applied to the beta model.
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

eta1_arr=[p[0]]
eta2_arr=[p[1]]
eta3_arr=[p[2]]
a1_arr=[p[3]]
a2_arr=[p[4]]
b1_arr=[p[5]]
b2_arr=[p[6]]

#N_iters=50
N_iters=1000

for n in range(N_iters):
    for i in range(len(p)):
        if i in [0,1,2]: #If it is eta's
            new_element=np.random.normal(loc=p[i],scale=eta_sigma) #draw random sample from proposal distribution
            p_prime=p
            #print(p)
            p_prime[i]=new_element
            print(p_prime)
            #Now i need to do the acceptance probability bit
            #Likelihood
            L1=Likelihood(p,V)
            L2=Likelihood(p_prime,V)
            #Priors
            #eta: mu=0.5,sigma=0.05
            #a: uniform so N/A
            #b: mu=0.7,sigma=0.07
            #P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
            #P2= normal(p_prime[0],0.5,0.05)* normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
            P1= np.log(normal(p[0],0.5,0.05))+np.log(normal(p[1],0.5,0.05))+np.log(normal(p[2],0.5,0.05))+np.log(uniform(p[3]))+np.log(uniform(p[4]))+np.log(normal(p[5],0.7,0.07))+np.log(normal(p[6],0.7,0.07)) #Prior for p
            P2= np.log(normal(p_prime[0],0.5,0.05))+np.log(normal(p_prime[1],0.5,0.05))+np.log(normal(p_prime[2],0.5,0.05))+np.log(uniform(p_prime[3]))+np.log(uniform(p_prime[4]))+np.log(normal(p_prime[5],0.7,0.07))+np.log(normal(p_prime[6],0.7,0.07)) #prior for p'
            #Candidates
            g1= np.random.normal(p[i],eta_sigma) #proposal for p'|p
            #g2= g1 #proposal for p|p', g is usually assumed to be symmetric???
            g2=np.random.normal(p_prime[i],eta_sigma)
            #numerator=L1*P1*g1
            #denominator=L2*P2*g2
            #numerator=np.exp(L1)*P1*g1
            #denominator=np.exp(L2)*P2*g2
            #elem=numerator/denominator
            #elem=(np.exp(L1)*P1*g1)/(np.exp(L2)*P2*g2)
            #elem=np.exp(L1-L2)*(P1*g1)/(P2*g2)
            elem=np.exp(L1+P1-L2-P2)*g1/g2
            #elem=(np.exp(L1-L2) + np.exp(P1-P2))*g1/g2
            #print(L1) #can be -inf
            #print(L2) #can be -inf
            #print(np.exp(L1-L2)) #L2 keeps equalling L1, perhaps diff is too small???
            #print(P1)
            #print(P2)
            #print(P1/P2) #Same situation as likelihood???
            #print(g1)
            #print(g2)
            #print(g1/g2) #Marked difference
            #print(elem)
            if np.isnan(elem):
                elem=0
            print("#########################")
            T=min(1,elem) #when elem is nan it defaults to 1
            #print(T)
            move_prob=random_coin(T)
            if move_prob:
                p=p_prime
        if i in [3,4]: #If it is a's
            new_element=np.random.normal(loc=p[i],scale=a_sigma) #draw random sample from proposal distribution
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
            #P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
            #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
            P1= np.log(normal(p[0],0.5,0.05))+np.log(normal(p[1],0.5,0.05))+np.log(normal(p[2],0.5,0.05))+np.log(uniform(p[3]))+np.log(uniform(p[4]))+np.log(normal(p[5],0.7,0.07))+np.log(normal(p[6],0.7,0.07)) #Prior for p
            P2= np.log(normal(p_prime[0],0.5,0.05))+np.log(normal(p_prime[1],0.5,0.05))+np.log(normal(p_prime[2],0.5,0.05))+np.log(uniform(p_prime[3]))+np.log(uniform(p_prime[4]))+np.log(normal(p_prime[5],0.7,0.07))+np.log(normal(p_prime[6],0.7,0.07)) #prior for p'
            #Candidates
            g1= np.random.normal(p[i],a_sigma) #proposal for p'|p
            #g2= g1 #proposal for p|p', g is usually assumed to be symmetric???
            g2=np.random.normal(p_prime[i],a_sigma)
            #numerator=L1*P1*g1
            #denominator=L2*P2*g2
            #elem=np.exp(L1-L2)*(P1*g1)/(P2*g2)
            elem=np.exp(L1+P1-L2-P2)*g1/g2
            #elem=numerator/denominator
            if np.isnan(elem):
                elem=0
            T=min(1,elem)
            move_prob=random_coin(T)
            if move_prob:
                p=p_prime
        if i in [5,6]: #If it is b's
            new_element=np.random.normal(loc=p[i],scale=b_sigma) #draw random sample from proposal distribution
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
            #P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
            #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
            P1= np.log(normal(p[0],0.5,0.05))+np.log(normal(p[1],0.5,0.05))+np.log(normal(p[2],0.5,0.05))+np.log(uniform(p[3]))+np.log(uniform(p[4]))+np.log(normal(p[5],0.7,0.07))+np.log(normal(p[6],0.7,0.07)) #Prior for p
            P2= np.log(normal(p_prime[0],0.5,0.05))+np.log(normal(p_prime[1],0.5,0.05))+np.log(normal(p_prime[2],0.5,0.05))+np.log(uniform(p_prime[3]))+np.log(uniform(p_prime[4]))+np.log(normal(p_prime[5],0.7,0.07))+np.log(normal(p_prime[6],0.7,0.07)) #prior for p'
            #Candidates
            g1= np.random.normal(p[i],b_sigma) #proposal for p'|p
            #g2= g1 #proposal for p|p', g is usually assumed to be symmetric???
            g2=np.random.normal(p_prime[i],b_sigma)
            #numerator=L1*P1*g1
            #denominator=L2*P2*g2
            #elem=numerator/denominator
            #elem=np.exp(L1-L2)*(P1*g1)/(P2*g2)
            elem=np.exp(L1+P1-L2-P2)
            if np.isnan(elem):
                elem=0
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

#Success

plt.hist(eta1_arr,normed=1,bins=20) #Showing sampling
plt.plot(eta1_arr)
    