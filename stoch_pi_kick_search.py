"""
Algorithm 7 from Alex Neville thesis. Stochastic π kick search algorithm. Bottom of page 98.

1. Label the current state vec(p_α)
2. Generate vec(q_α) = (q_1,...,q_k) where each q_i ~ unif{−π, 0, π}.
3. If P (D|vec(p_α) + vec(q_α),V)P(vec(p)+vec(q_α)) > P(D|vec(p_α),V)P(vec(p_α)) then 
   transition from vec(p_α) to vec(p_α) + vec(q_α).
4. Go to 1.
"""

#Fairly confident that the three terms in unif is not valid expression but i think the expression means to say that
#it will be a string of terms that are either -pi or 0, e.g. q={-π,0,0,-π,-π,-π,0}???

#Or does it mean an approximately uniform balance of -pi,0, and pi. e.g. for q of length 9, then
#q={-π,0,0,π,-π,-π,π,0,π} so that there is 3 of each. But then what for lengths that aren't divisible by 3,
#how do i go for approximately uniform then?
import numpy as np

#q=[np.random.uniform(low=-np.pi,high=np.pi) for i in range(7)]
#x=np.random.randint(low=0,high=3,size=10,dtype='I')
#x=np.random.choice(3,10)

#x=np.random.choice(3,7)
#q=[(x[i]-1)*np.pi for i in range(len(x))]
#print(q)
#print(x)

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
#print(p)
#print(q)
#print(np.add(p,q))
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
   x=np.random.choice(3,len(p))
   q=[(x[i]-1)*np.pi for i in range(len(x))]
   #print(p)
   #print(q)
   #print(p+q)
   #Now i need to do the acceptance probability bit
   #Likelihood
   L1=Likelihood(np.add(p,q),V)
   L2=Likelihood(p,V)
   #Priors
   #eta: mu=0.5,sigma=0.05
   #a: uniform so N/A
   #b: mu=0.7,sigma=0.07
   P1= normal(p[0]+q[0],0.5,0.05)*normal(p[1]+q[1],0.5,0.05)*normal(p[2]+q[2],0.5,0.05)*uniform(p[3]+q[3])*uniform(p[4]+q[4])*normal(p[5]+q[5],0.7,0.07)*normal(p[6]+q[6],0.7,0.07) #Prior for p+q
   P2= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
   if (np.exp(L1)*P1)>(np.exp(L1)*P1):
       p=np.add(p,q)
   #print(p)
   eta1_arr.append(p[0])
   eta2_arr.append(p[1])
   eta3_arr.append(p[2])
   a1_arr.append(p[3])
   a1_arr.append(p[4])
   b1_arr.append(p[5])
   b2_arr.append(p[6])

plt.hist(b1_arr,normed=1,bins=20) #Showing sampling
plt.plot(b1_arr)