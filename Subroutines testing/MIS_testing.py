"""
This code file is for the intent of testing and verifying the functionality of the MIS subalgorithm.

For this since parameters don't need to be for the quantum case to have confidence that it works, 
although we can pivot to maybe a simple check afterwards like just a single beamsplitter estimate.

So for this i shall just use MIS to estimate mu and sigma for a normal distribution.

Situational example of trying to estimate mu and sigma for a normal distribution is actually not quite so simple since the prior is hierarchical:
https://www.statlect.com/fundamentals-of-statistics/normal-distribution-Bayesian-estimation

Will need to rethink example case? Maybe i should just do a single variable thing, like estimating just mu not sigma?

Likelihood is wrong and dodgy which i think is part of the reason as to why it drifts so far away from true value

Likelihood issue resolved, but if sim_draws increased then likelihood becomes smaller and so eventually can get rounded to zero so need to alternate form into loglikelihood and edit acceptance probability form.

Final bug is that i am occassionally get 'nan' results for elem, so i have used np.isnan to catch these and make 
them zero rather than change the acceptance rule since that would be awkward. However this seems to yield that mu_current is no longer changing
when i push to high sim_draws and N_iters so perhaps i will have to as it will be more general.

keep getting nans for elem ehich is being raised due to a divide by zero error. Investigation reveals that nan arises when sim_draws gets too high.

Fully debugged apart from the fact that Likelihood(mu) doesn't handle high number of sim_draws due to the 
exponentiation to the power of -length/2 which means it becomes zero so i can't really compare the posterior 
to a histogram of sim_draws...
"""

"""
import numpy as np
import matplotlib.pyplot as plt

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
sim_draws=np.random.standard_normal(size=1000)
#print(sim_draws)

#plt.hist(sim_draws,density=True) #checks out
#plt.show()

N_iters=1000 #Number of iterations of algorithm
#N_iters=1000 #Number of iterations of algorithm

p=[0,0.5] #p=[mu,sigma]
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
        res=res*element
    return res

for _ in range(N_iters):
    #print(mu)
    mu_new=np.random.normal(loc=mu[_])
    #print(mu_new)
    sigma_new=np.random.normal(loc=sigma[_])
    p_new=[mu_new,sigma_new]
    L1=Likelihood(p)
    L2=Likelihood(p_new)
    #print(L1) #Need to handle probs being rounded to zero
    #print(L2)
    #P1=normal(p[0],0,1)*normal(p[1],1,1)
    #P2=normal(p_new[0],0,1)*normal(p_new[1],1,1)
    P1=normal(p[0],0,p[1])
    P2=normal(p_new[0],0,p_new[1])
    #print(P1)
    #print(P2)
    elem=(L1*P1)/(L2*P2)
    T=min(1,elem)
    #print("Transition probability is: {}".format(T)) #issue is transition probabilty is too frequently 1 even when things are way off.
    move_prob=random_coin(T)
    #print(move_prob)
    #print("####################################")
    if move_prob:
         p=p_new
    mu.append(p[0])
    sigma.append(p[1])

plt.plot(mu)
plt.plot(sigma)
plt.show()
"""
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def normal(x,mu,sigma):
    numerator = np.exp((-(x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator

def random_coin(p):
    unif = np.random.uniform(0,1)
    if unif>=p:
        return False
    else:
        return True
#Draws from a standard normal: N(0,1)
sim_draws=np.random.standard_normal(size=100)
#print(sim_draws) 

#plt.hist(sim_draws,density=True) #checks out
#plt.show()

N_iters=100 #Number of iterations of algorithm
#N_iters=1000 #Number of iterations of algorithm
mu_current=0
mu=[mu_current]

def Likelihood(mu):
    summation=0
    sigma=1
    for _ in range(len(sim_draws)):
        #((2*np.pi*1**2)**(-(_+1)/2))*np.exp((-1/(2*1**2))*())
        sum_term=(sim_draws[_]-mu)**2
        summation+=sum_term
    result=((2*np.pi*sigma**2)**(-len(sim_draws)/2))*np.exp(-(summation)/(2*sigma**2))
    return result

for _ in range(N_iters):
    print(mu_current)
    mu_new=np.random.normal(loc=mu_current,scale=0.1)
    print(mu_new)
    #sigma_new=np.random.normal(loc=sigma[_])
    #p_new=[mu_new,sigma_new]
    L1=Likelihood(mu_current)
    L2=Likelihood(mu_new)
    print(L1) #coming out to zero?
    print(L2)
    #P1=normal(p[0],0,1)*normal(p[1],1,1)
    #P2=normal(p_new[0],0,1)*normal(p_new[1],1,1)
    P1=normal(mu_current,0,1)
    P2=normal(mu_new,0,1)
    #print(P1)
    #print(P2)
    elem=(L1*P1)/(L2*P2)
    T=min(1,elem)
    #print("Transition probability is: {}".format(T)) #issue is transition probabilty is too frequently 1 even when things are way off.
    move_prob=random_coin(T)
    #print(move_prob) #No move ever got rejected because T was always 1...
    print("####################################")
    if move_prob:
         mu_current=mu_new
    mu.append(mu_current)
    #sigma.append(p[1])

plt.plot(mu)
#eval_points = np.linspace(np.min(mu), np.max(mu),len(mu))
#kde=gaussian_kde(mu)
#evaluated=kde.evaluate(eval_points)
#evaluated/=sum(evaluated) #For normalisation
#plt.plot(eval_points,evaluated)
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def normal(x,mu,sigma):
    numerator = np.exp((-(x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator

def random_coin(p):
    unif = np.random.uniform(0,1)
    if unif>=p:
        return False
    else:
        return True
#Draws from a standard normal: N(0,1)
sim_draws=np.random.standard_normal(size=10)
#print(sim_draws) 

#plt.hist(sim_draws,density=True) #checks out
#plt.show()

N_iters=100000 #Number of iterations of algorithm
#N_iters=1000 #Number of iterations of algorithm
mu_current=0
mu=[mu_current]

def Likelihood(mu):
    summation=0
    sigma=1
    for _ in range(len(sim_draws)):
        #((2*np.pi*1**2)**(-(_+1)/2))*np.exp((-1/(2*1**2))*())
        sum_term=(sim_draws[_]-mu)**2
        summation+=sum_term
    #result=((2*np.pi*sigma**2)**(-len(sim_draws)/2))*np.exp(-(summation)/(2*sigma**2))
    #print(-summation/(2*sigma**2))
    #print((2*np.pi*sigma**2)**(-len(sim_draws)/2)) #this is coming out to zero which means log becomes inf
    result=(-summation/(2*sigma**2))*np.log((2*np.pi*sigma**2)**(-len(sim_draws)/2))
    return result

for _ in range(N_iters):
    #print(mu_current)
    mu_new=np.random.normal(loc=mu_current,scale=1)
    #print(mu_new)
    #print("#########")
    #sigma_new=np.random.normal(loc=sigma[_])
    #p_new=[mu_new,sigma_new]
    L1=Likelihood(mu_current)
    L2=Likelihood(mu_new)
    #print(L1) #getting inf
    #print(L2)
    #P1=normal(p[0],0,1)*normal(p[1],1,1)
    #P2=normal(p_new[0],0,1)*normal(p_new[1],1,1)
    P1=np.log(normal(mu_current,0,1))
    P2=np.log(normal(mu_new,0,1))
    #print(P1)
    #print(P2)
    #elem=(L1*P1)/(L2*P2)
    elem=(L1+P1-L2-P2)
    #elem=np.exp(elem)
    #if np.isnan(elem):
    #    elem=0
    #print(elem)
    #T=min(1,elem)
    coin_draw=np.random.uniform(low=0,high=1)
    #print(np.log(coin_draw))
    if np.log(coin_draw)<=elem:
        move_prob=True
    else:
        move_prob=False
    #print("Transition probability is: {}".format(T)) #issue is transition probabilty is too frequently 1 even when things are way off.
    #move_prob=random_coin(T)
    #print(move_prob) #No move ever got rejected because T was always 1...
    #print("####################################")
    if move_prob:
         mu_current=mu_new
    mu.append(mu_current)
    #sigma.append(p[1])

#plt.plot(mu)
eval_points = np.linspace(np.min(mu), np.max(mu),len(mu))
kde=gaussian_kde(mu)
evaluated=kde.evaluate(eval_points)
evaluated/=sum(evaluated) #For normalisation
plt.plot(eval_points,evaluated)

#plt.hist(sim_draws,density=True)
plt.show()