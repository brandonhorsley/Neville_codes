"""
This code file will be for investigating the impact of choice of priors on Alex's procedure and will be 
following up on Anthony's suggestion about investigating choice of priors and it's impact on things like 
convergence.

In this procedure the initial parameters are estimated so the choice of 'priors' are more the candidate proposals 
in each of the sub algorithms which are mostly normal distributions. Well more specifically a candidate solution is 
drawn and then that proposal informs the value for the prior, and this is done for varying different models so 
there is a lot of rough experimentation to be done here since that means i have three codependent degrees of 
freedom is candidate proposal to draw from, the prior, and the algorithm it is applied to.

What is the impact of dissonance between the candidate proposal and prior i wonder?

But before proceeding i suppose i should look at the figures of merit i am going to look at. The main thing i'd
wager should be convergence, so i should polish my code to reflect that investigation. I believe Alex makes an 
average error across multiple runs to generate a plot so that could be something to look at. I can also think 
about maybe seeing if i can implement a more definitive measure of convergence in the probabilistic sense. 

If the domain explored in q [proposal] is too small, compared with the range of f (the target's distribution density)
, the Markov Chain will have difficulties in exploring this range and will converge very slowly:
-> support is the range of possible values that have nonzero prob
https://stats.stackexchange.com/questions/360194/how-can-the-support-of-proposal-distribution-impact-convergence-of-rh-mh-algorit

So it looks like one way of boosting convergence time is increasing the sd of the normal distribution. What about shape of support then i wonder?

Proposal distribution and prior don't need to be the same:
https://stats.stackexchange.com/questions/168298/is-that-ok-to-have-the-same-prior-and-proposal-distribution-in-mh

List of distributions:
https://www.pymc.io/projects/docs/en/stable/api/distributions.html

Uniform
Normal
StudentT
Polya Gamma?
Moyal? (distribution is quite positively skewed)
Laplace
Gamma
ExGaussian/exponentially modified gaussian log likelihood
"""

import numpy as np
import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '/Users/ck21395/PhD codes/Neville_codes')
from Aux_Nev import *

import matplotlib.pyplot as plt
from scipy.stats import rv_continuous

"""
From Aux_Nev the true values:

Vmax=5
N=100 #Top of page 108 ->N=number of experiments
M=2 #Number of modes
V1=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter 1
#V1=V1+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
V2=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter 2
#V2=V2+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise

a1_true=0
a2_true=0

b1_true=0.788
b2_true=0.711

eta1_true=0.447
eta2_true=0.548
eta3_true=0.479

Likelihood and DataGen functions come from Aux_Nev.py
"""
#Data generation (step 1) stuff is all contained in Aux_Nev so i will leave it that way for now for readability 
#but if i need differed functionality then i can bring the code into this code document.

###Preliminaries###

def normal(x,mu,sigma):
    numerator = np.exp((-(x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator

#I can only apply uniform to eta and to a since they can only be bounded variables, b is an unbounded parameter 
# and you can't have a flat distribution over unbounded space
def uniform(p,lower, upper):
    if p>=lower and p<=upper:
        return 1/(upper-lower)
    else:
        return 0

def random_coin(p):
    unif = np.random.uniform(0,1)
    if unif<=p:
        return True
    else:
        return False

from scipy.special import gamma

"""
def StudentT(x,df,mu,sigma): #for when you know x (so in essence for defining the prior)
    Y=(x-mu)/sigma
    Z=abs(sigma)*np.sqrt(df*np.pi)*gamma(0.5*df)/gamma(0.5*(df+1))
    return (1+Y**2/df)**(-0.5(df+1))/Z

def StudentTcandidate(mu,sigma):
    v=sigma**2
    if v==1:
        df=np.inf
    else:
        df=2*v/(v-1)
    s=np.random.standard_t(df=df)
    s+=mu
    return s
"""

def StudentT(x,df): #for when you know x (so in essence for defining the prior)
    return ((gamma((df+1)/2))/(np.sqrt(np.pi*df)*gamma(df/2)))*(1+(x**2)/df)**(-(df+1)/2)

#eq 4.11: g_i(p',p)=Normal(p_i,sigma_i)
#sigma_i=pi/200 for a, b_est for b, 0.005 for eta

b_est=0.07 
"""
b_est is subject to change and consistency w. Aux_Nev should be certified, additionally sigma_i for b is 
suggested it should be b_est but the actual estimated value is closer to 0.7 so there is confusion there.
"""

eta_sigma=0.005
a_sigma=np.pi/200
b_sigma=b_est #Based around true values from Neville_thesis_8.py
#N_iters=100000

#I=[2,500,50,50,500,100,100,100000] #Determines iteration number for each algorithm call
I=[2,500,50,50,500,100,100,100000]

#I[-1]=1 iteration takes 0.08s,10 takes 0.8 so 100,000 should take ~10,000s=~1667min=~27 hours=~1.15 days
#AKA divide I[-1] by 10 to get approx. runtime
# Numba tend to boast an order or two orders of magnitude speedup in general (with caveats ofc)
# 1 order of magnitude:  ~1000s=~16-17mins
# 2 orders of magnitude: ~100s=~1-2 mins

runtime=(I[-1]/10)+60
print("runtime in seconds is around {}s".format(runtime))
print("runtime in minutes is around {}min".format(runtime/60))
print("runtime in hours is around {}hr".format(runtime/(3600)))

###Burn in###

p_alpha=[0.5,0.5,0.5,0,0,0.5,0.5] #step 2.1
print("p_alpha initial is {}".format(p_alpha))
#p_alpha=[0,0] #step 2.1

"""
Defining Algorithms from thesis. Have done other documents that implement them but i 
shall use those to define these functions which should give a specific output as wanted.
"""

def Alg4_alpha(p_alpha, Niters):
    """
    Algorithm variant of Algorithm 4 for estimating the alpha model (which involves 
    only learning the a values).
    """
    for n in range(Niters):
        for i in range(len(p_alpha)):
            if i in [3,4]: #If it is a's
                new_element=np.random.normal(loc=p_alpha[i],scale=a_sigma) #draw random sample from proposal distribution
                #new_element=np.random.uniform(low=-np.pi,high=np.pi)
                #new_element=scipy.stats.t(mu=p_alpha[i],df=(N-1))  #don't use until i figure out degrees of freedom for our context 
                #new_element=p_alpha[i]+np.random.standard_t(df=(N-1))
                p_prime=list(p_alpha)
                p_prime[i]=new_element
                #Likelihood
                L1=Likelihood(p_alpha,V1,V2)
                L2=Likelihood(p_prime,V1,V2)
                #Priors
                #eta: mu=0.5,sigma=0.05
                #a: uniform so N/A
                #b: mu=0.7,sigma=0.07
                #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                #P1=uniform(p_alpha[i])
                P1=uniform(p_alpha[i],-np.pi,np.pi)
                #P1=StudentT(x=(new_element-p_alpha[i]),df=(N-1))
                #P2=uniform(p_prime[i])
                #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                P2=uniform(p_alpha[i],-np.pi,np.pi)
                #P1=StudentT(x=(new_element-p_prime[i]),df=(N-1))
                #Candidates
                g1= np.random.normal(p_alpha[i],a_sigma)
                g2=np.random.normal(p_prime[i],a_sigma)
                numerator=L1*P1*g1
                denominator=L2*P2*g2
                elem=numerator/denominator
                T=min(1,elem)
                move_prob=random_coin(T)
                if move_prob:
                    p_alpha=p_prime

    return p_alpha

def Alg4_beta(p_beta, Niters):
    """
    Algorithm variant of Algorithm 4 for estimating the beta model (which involves 
    learning the a and b values).
    """
    for n in range(Niters):
        for i in range(len(p_beta)):
            if i in [3,4]: #If it is a's
                new_element=np.random.normal(loc=p_beta[i],scale=a_sigma) #draw random sample from proposal distribution
                #new_element=np.random.uniform(low=-np.pi,high=np.pi)
                #new_element=scipy.stats.t(mu=p_beta[i],df=(M-1))   #don't use until i figure out degrees of freedom for our context 
                #new_element=p_beta[i]+np.random.standard_t(df=(N-1))
                p_prime=list(p_beta)
                p_prime[i]=new_element
                #Likelihood
                L1=Likelihood(p_beta,V1,V2)
                L2=Likelihood(p_prime,V1,V2)
                #Priors
                #eta: mu=0.5,sigma=0.05
                #a: uniform so N/A
                #b: mu=0.7,sigma=0.07
                #P1= normal(p_beta[0],0.5,0.05)*normal(p_beta[1],0.5,0.05)*normal(p_beta[2],0.5,0.05)*uniform(p_beta[3])*uniform(p_beta[4])*normal(p_beta[5],0.7,0.07)*normal(p_beta[6],0.7,0.07) #Prior for p
                #P1=uniform(p_beta[i])
                P1=uniform(p_beta[i],-np.pi,np.pi)
                #P1=StudentT(x=(new_element-p_beta[i]),df=(N-1))
                #P2=uniform(p_prime[i])
                #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                P2=uniform(p_prime[i],-np.pi,np.pi)
                #P2=StudentT(x=(new_element-p_prime[i]),df=(N-1))
                #Candidates
                g1= np.random.normal(p_beta[i],a_sigma)
                g2=np.random.normal(p_prime[i],a_sigma)
                numerator=L1*P1*g1
                denominator=L2*P2*g2
                elem=numerator/denominator
                T=min(1,elem)
                move_prob=random_coin(T)
                if move_prob:
                    p_beta=p_prime
            if i in [5,6]: #If it is b's
                new_element=np.random.normal(loc=p_beta[i],scale=b_sigma) #draw random sample from proposal distribution
                #new_element=np.random.uniform(low=-np.pi,high=np.pi) #Not sure on range since b can be unbounded
                #new_element=scipy.stats.t(mu=p_beta[i],df=(M-1))   #don't use until i figure out degrees of freedom for our context 
                #new_element=p_beta[i]+np.random.standard_t(df=(N-1))
                p_prime=list(p_beta)
                p_prime[i]=new_element
                #Likelihood
                L1=Likelihood(p_beta,V1,V2)
                L2=Likelihood(p_prime,V1,V2)
                #Priors
                #eta: mu=0.5,sigma=0.05
                #a: uniform so N/A
                #b: mu=0.7,sigma=0.07
                #P1= normal(p_beta[0],0.5,0.05)*normal(p_beta[1],0.5,0.05)*normal(p_beta[2],0.5,0.05)*uniform(p_beta[3])*uniform(p_beta[4])*normal(p_beta[5],0.7,0.07)*normal(p_beta[6],0.7,0.07) #Prior for p_beta
                P1=normal(p_beta[i],0.7,0.07)
                #P1=StudentT(x=(new_element-p_beta[i]),df=(N-1))
                #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p_beta'
                P2=normal(p_prime[i],0.7,0.07)
                #P2=StudentT(x=(new_element-p_prime[i]),df=(N-1))
                #Candidates
                g1= np.random.normal(p_beta[i],b_sigma) 
                g2=np.random.normal(p_prime[i],b_sigma)
                numerator=L1*P1*g1
                denominator=L2*P2*g2
                elem=numerator/denominator
                T=min(1,elem)
                move_prob=random_coin(T)
                if move_prob:
                    p_beta=p_prime
    return p_beta

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
                    #new_element=np.random.uniform(low=0,high=1)
                    #new_element=scipy.stats.t(mu=p[i],df=(M-1))   #don't use until i figure out degrees of freedom for our context 
                    new_element=p[i]+np.random.standard_t(df=(N-1))
                    p_prime=list(p)
                    p_prime[i]=new_element
                    #Likelihood
                    L1=Likelihood(p,V1,V2)
                    L2=Likelihood(p_prime,V1,V2)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    #P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
                    P1=normal(p[i],0.5,0.05)
                    #P1=StudentT(x=(new_element-p[i]),df=(N-1))
                    #P2= normal(p_prime[0],0.5,0.05)* normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                    P2=normal(p_prime[i],0.5,0.05)
                    #P2=StudentT(x=(new_element-p_prime[i]),df=(N-1))
                    #Candidates
                    g1= np.random.normal(p[i],eta_sigma)
                    g2=np.random.normal(p_prime[i],eta_sigma)
                    elem=(np.exp(L1)*P1*g1)/(np.exp(L2)*P2*g2)
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p=p_prime
                if i in [3,4]: #If it is a's
                    new_element=np.random.normal(loc=p[i],scale=a_sigma) #draw random sample from proposal distribution
                    #new_element=np.random.uniform(low=-np.pi,high=np.pi)
                    #new_element=scipy.stats.t(mu=p[i],df=(M-1))   #don't use until i figure out degrees of freedom for our context 
                    #new_element=p[i]+np.random.standard_t(df=(N-1))
                    p_prime=list(p)
                    p_prime[i]=new_element
                    #Likelihood
                    L1=Likelihood(p,V1,V2)
                    L2=Likelihood(p_prime,V1,V2)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    #P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
                    #P1=uniform(p[i])
                    P1=uniform(p[i],-np.pi,np.pi)
                    #P1=StudentT(x=(new_element-p[i]),df=(N-1))
                    #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                    #P2=uniform(p_prime[i])
                    P2=uniform(p[i],-np.pi,np.pi)
                    #P2=StudentT(x=(new_element-p_prime[i]),df=(N-1))
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
                if i in [5,6]: #If it is b's
                    new_element=np.random.normal(loc=p[i],scale=b_sigma) #draw random sample from proposal distribution
                    #new_element=np.random.uniform(low=-np.pi,high=np.pi)
                    #new_element=scipy.stats.t(mu=p[i],df=(M-1))   #don't use until i figure out degrees of freedom for our context 
                    #new_element=p[i]+np.random.standard_t(df=(N-1))
                    p_prime=list(p)
                    p_prime[i]=new_element
                    #Likelihood
                    L1=Likelihood(p,V1,V2)
                    L2=Likelihood(p_prime,V1,V2)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    #P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
                    P1=normal(p[i],0.7,0.07)
                    #P1=StudentT(x=(new_element-p[i]),df=(N-1))
                    #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                    P2=normal(p_prime[i],0.7,0.07)
                    #P2=StudentT(x=(new_element-p_prime[i]),df=(N-1))
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
            if ReturnAll:
                MCMC.append(p)

        if ReturnAll:
            return MCMC
        else:
            return p

def Alg5(p_alpha,Niters):
    """
    This Algorithm is the Metropolised Independence Sampling (MIS) algorithm that is
    described at the top of page 98 in Alex Neville's thesis.
    """
    for n in range(Niters):
        #print(p_alpha)
        p_prime=list(p_alpha) #Need to use list, lesson learned after 4 years of coding
        #print(p_prime)
        new=[np.random.uniform(low=-np.pi,high=np.pi) for i in range(2)]
        p_prime[3]=new[0]
        p_prime[4]=new[1]
        #Likelihood
        L1=Likelihood(p_alpha,V1,V2)
        L2=Likelihood(p_prime,V1,V2)
        #Priors
        #eta: mu=0.5,sigma=0.05
        #a: uniform so N/A
        #b: mu=0.7,sigma=0.07
        #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
        P1=uniform(p_alpha[3],-np.pi,np.pi)*uniform(p_alpha[4],-np.pi,np.pi)
        #P2= normal(p_prime[0],0.5,0.05)* normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
        P2=uniform(p_prime[3],-np.pi,np.pi)*uniform(p_prime[4],-np.pi,np.pi)
        elem=(np.exp(L1)*P1)/(np.exp(L2)*P2)
        T=min(1,elem)
        move_prob=random_coin(T)
        if move_prob:
            p_alpha=p_prime
    return p_alpha

def Alg6(p_alpha,Niters):
    """
    This Algorithm is the Metropolised Independence Sampling (MIS) within Gibbs algorithm 
    that is described in the middle of page 98 in Alex Neville's thesis.
    """
    for n in range(Niters):
        for i in range(len(p_alpha)):
            if i in [3,4]: #If it is a's
                new_element=np.random.uniform(low=-np.pi,high=np.pi)
                #new_element=np.random.normal()
                #new_element=scipy.stats.t(mu=p_alpha[i],df=(M-1)) #don't use until i figure out degrees of freedom for our context 
                #new_element=p_alpha[i]+np.random.standard_t(df=(N-1))
                p_prime=list(p_alpha)
                p_prime[i]=new_element
                #Likelihood
                L1=Likelihood(p_alpha,V1,V2)
                L2=Likelihood(p_prime,V1,V2)
                #Priors
                #eta: mu=0.5,sigma=0.05
                #a: uniform so N/A
                #b: mu=0.7,sigma=0.07
                #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                P1=uniform(p_alpha[i],-np.pi,np.pi)
                #P1=StudentT(x=(new_element-p_alpha[i]),df=(N-1))
                #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                P2=uniform(p_prime[i],-np.pi,np.pi)
                #P2=StudentT(x=(new_element-p_prime[i]),df=(N-1))
                numerator=L1*P1
                denominator=L2*P2
                elem=numerator/denominator
                T=min(1,elem)
                move_prob=random_coin(T)
                if move_prob:
                    p_alpha=p_prime

    return p_alpha

def Alg7(p_alpha, Niters):
    """
    This Algorithm is the stochastic π kick search algorithm that is described 
    at the bottom of page 98 in Alex Neville's thesis.
    """
    for n in range(Niters):
        x=np.random.choice(3,2)
        q=[(x[i]-1)*np.pi for i in range(len(x))]
        #Likelihood
        #print(q)
        test=list(p_alpha)
        test[3]+=q[0]
        test[4]+=q[1]
        #print(test)
        L1=Likelihood(test,V1,V2)
        L2=Likelihood(p_alpha,V1,V2)
        #Priors
        #eta: mu=0.5,sigma=0.05
        #a: uniform so N/A
        #b: mu=0.7,sigma=0.07
        #P1= normal(test[0],0.5,0.05)*normal(test[1],0.5,0.05)*normal(test[2],0.5,0.05)*uniform(test[3])*uniform(test[4])*normal(test[5],0.7,0.07)*normal(test[6],0.7,0.07) #Prior for p+q
        P1=uniform(test[3],-np.pi,np.pi)*uniform(test[4],-np.pi,np.pi)
        #print(P1)
        #P2= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
        P2=uniform(p_alpha[3],-np.pi,np.pi)*uniform(p_alpha[4],-np.pi,np.pi)
        #print(P2)
        #print(np.exp(L1))
        #print(P1)
        #print(np.exp(L2))
        #print(P2)
        if (np.exp(L1)*P1)>(np.exp(L2)*P2):
            p_alpha=test
    return p_alpha

#Main code bulk

for i in range(I[0]): #step 2.2
    #step 2.2i
    p_alpha=Alg5(p_alpha,I[1])
    print(p_alpha)
    print("###step 2.2i done###")
    #step 2.2ii
    p_alpha=Alg6(p_alpha,I[2])
    print(p_alpha)
    print("###step 2.2ii done###")
    #step 2.2iii
    #p_alpha=Alg4(p_alpha, I[3]) #p_alpha is first p_alpha
    p_alpha=Alg4_alpha(p_alpha, I[3]) #p_alpha is second p_alpha
    print(p_alpha)
    print("###step 2.2iii done###")
    #step 2.2iv (and 2.2v)
    p_alpha=Alg7(p_alpha,I[4])
    print(p_alpha)
    print("###step 2.2iv done###")

#p_beta=[0.5,0.5,0.5,p_alpha[3],p_alpha[4],b_est,b_est] #step 2.3
p_beta=[0.5,0.5,0.5,p_alpha[3],p_alpha[4],0.7,0.7]

print("p_beta initial is: {}".format(p_beta))
#step 2.4
p_beta=Alg4_beta(p_beta, I[5])
print(p_beta)
print("###step 2.4 done###")

p_zero=[0.5,0.5,0.5,p_beta[3],p_beta[4],p_beta[5],p_beta[6]] #step 2.5
print("p_zero is: {}".format(p_zero))
#step 2.6
p_zero=Alg4(p_zero,I[6], Markov=False)
print(p_zero)
print("###step 2.6 done###")

p_conv=p_zero #step 2.7
print("p_conv is: {}".format(p_conv))


###Main Markov Chain Generation###

#Step 3
chain=Alg4(p_conv,I[7], Markov=True)

###Parameter estimation###

"""
I imagine this chain object should contain all the values for each parameter at each 
markov chain state number (i.e. I[7] by 7 matrix).
To start with i shall just generate the markov chain plot and comment out the histogram 
plot and polish can later be applied to get the standard plot with the smoothed histogram in the left column and
markov state number plot in the right column with a Plot() function.
"""

from scipy.stats import gaussian_kde

names=["eta1","eta2","eta3","a1","a2","b1","b2"]
trues=[eta1_true,eta2_true,eta3_true,a1_true,a2_true,b1_true,b2_true]

def Plot(chain): #Chain should contain all necessary markov chain data
    """
    Custom plot function to generate the standard traceplot format where there is a smoothed
    histogram for each parameter in the left hand column and the markov chain plot in the right hand column.
    e.g. https://python.arviz.org/en/stable/examples/plot_trace.html
    """
    fig,axs=plt.subplots(len(p_conv),2,constrained_layout=True) #Can use sharex,sharey for further polish if wanted
    for i in range(len(p_conv)):
        #histogram
        #axs[i,0].hist(chain[:,i],bins=30)
        #smoothed kde with scott's rule bandwidth selection (bw selection is important consideration in kde)
        eval_points = np.linspace(np.min(chain[:,i]), np.max(chain[:,i]),len(chain[:,i]))
        kde=gaussian_kde(chain[:,i])
        evaluated=kde.evaluate(eval_points)
        evaluated/=sum(evaluated) #For normalisation
        axs[i,0].plot(eval_points,evaluated)
        axs[i,0].axvline(x=trues[i],c="red")
        #Add axs polish like axes labelling
        axs[i,0].set_ylabel(str(names[i])) #Add label to show which parameter is which
        axs[i,1].plot(chain[:,i])
        axs[i,1].axhline(y=trues[i],c="red")
        axs[i,1].set_xlabel("Markov chain State Number") #Aid understanding of Markov chain plot
    #fig.tight_layout()
    plt.show()

chain=np.array(chain)
#print(chain)
Plot(chain)

for i in range(len(p_conv)): #step 4
    par_array=chain[:,i]
    #Plot markov chain plot
    print(names[i])
    #plt.plot(par_array)
    print("Mean is {}".format(np.mean(par_array)))
    print("Standard deviation is {}".format(np.std(par_array)))
    
###########################################

#For testing to compare unitaries, say V=1 as a simpler test

U_true=ConstructU(eta1_true,eta2_true,eta3_true,a1_true+b1_true,a2_true+b2_true)

phi_proof1=np.mean(chain[:,3])+np.mean(chain[:,5])
phi_proof2=np.mean(chain[:,4])+np.mean(chain[:,6])

U_proof=ConstructU(np.mean(chain[:,0]),np.mean(chain[:,1]),np.mean(chain[:,2]),phi_proof1,phi_proof2)

print("U_true is")
print(U_true)

print("U_proof is")
print(U_proof)

#################################################
"""
from scipy.linalg import sqrtm

print("Trace Distance is")
print(0.5*np.trace(sqrtm((U_true-U_proof).conj().T@(U_true-U_proof))))
"""

#tools for comparing closeness of unitaries? I thought about trace distance but that is more for states...
#This link suggests the Hilbert Schmidt norm:
#https://quantumcomputing.stackexchange.com/questions/6821/what-would-be-an-ideal-fidelity-measure-to-determine-the-closeness-between-two-n

#print("Hilbert-Schmidt norm is")
#print(np.trace(U_true.conj().T@U_proof))

#The link also suggests For unitary operators, our analysis of the error used in approximating one unitary by another involves the distance induced by the operator norm:
#This corresponds to the largest eigenvalues of sqrm((U-V).conj().T@(U-V))
#ref: https://physics.stackexchange.com/questions/588940/operator-norm-and-action
#If U=V then operator norm will be zero, is upper bound maybe the same as dimension (e.g. 2)
print("The operator norm is:")
#phi_test=np.array([1,0])
#phi_test.shape=(2,1)

#O_norm=np.linalg.norm(((U_true-U_proof)@phi_test),ord=2)/np.linalg.norm(phi_test,ord=2)
#O_norm=np.linalg.norm(U_true-U_proof,ord=np.inf)

from scipy.linalg import sqrtm

w,v=np.linalg.eig(sqrtm((U_true-U_proof).conj().T@(U_true-U_proof)))
O_norm=max(w)
print(O_norm)