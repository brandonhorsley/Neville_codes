"""
Code to implement a parallel version of MCMC chains with the MCMC procedure from Alex Neville's thesis, 
parallel chains isn't mentioned in his main procedure but is useful to have as a file here for later 
implementations. Speed up of an MCMC chain itself will have to be through numba which i haven't properly 
sorted yet, but even then further investigation may require just an alternative method since Metropolis-Hastings 
requires a lot of steps in general.

https://nealhughes.net/parallelcomp/
The link above raises a valid point that parallel chains would just generate basically the same samples without changing
the random seed.

https://stackoverflow.com/questions/8533318/multiprocessing-pool-when-to-use-apply-apply-async-or-map

My investigation of numba has hit a wall but this code is now working so i can parallelise separate
MCMC chains. My next step is to find a better sense of burn-in using different diagnostics.
https://stats.stackexchange.com/questions/261539/when-we-run-many-chains-at-once-in-an-mcmc-model-how-are-they-combined-together
https://stats.stackexchange.com/questions/65614/in-mcmc-how-is-burn-in-time-chosen

https://search.r-project.org/CRAN/refmans/LaplacesDemon/html/Geweke.Diagnostic.html
https://search.r-project.org/CRAN/refmans/LaplacesDemon/html/Heidelberger.Diagnostic.html
https://search.r-project.org/CRAN/refmans/LaplacesDemon/html/Raftery.Diagnostic.html
https://search.r-project.org/CRAN/refmans/LaplacesDemon/html/Gelman.Diagnostic.html
"""

from Aux_Nev import *
import numpy as np
#from functools import reduce
#from multiprocessing import Pool, cpu_count
#from multiprocessing import Process
#from multiprocessing.queues import Queue
import multiprocessing
import matplotlib.pyplot as plt
"""
From Aux_Nev the true values:

Vmax=10
N=100 #Top of page 108 ->N=number of experiments
M=2 #Number of modes
V_dist=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5
#V=V_dist+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
V=V_dist

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

def uniform(x):
    return 1/(2*np.pi)

def random_coin(p):
    unif = np.random.uniform(0,1)
    if unif<=p:
        return True
    else:
        return False

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
I=[2,500,50,50,500,100,100,100]

#Example p
p_start=[0.43,0.52,0.5,-0.3,0.2,0.75,0.8]

#Alg4 run in serial
def Alg4_serial(p,chainlength,chainnum,Markov=False,ReturnAll=False):
    
    #This Algorithm is the Metropolis-Hastings within Gibbs sampling algorithm that is 
    #described on the middle of page 94 in Alex Neville's thesis.
    
    if Markov: #i.e. if Markov==True
        MCMC=Alg4_serial(p,chainlength, chainnum,Markov=False, ReturnAll=True)
        return MCMC
    else: #If Markov==False
        MCMC=np.empty((chainnum,chainlength))
        for _ in range(chainnum):
            for n in range(chainlength):
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
                if ReturnAll:
                    MCMC[_][n]=p

        if ReturnAll:
            return MCMC
        else:
            return p

def Markov(p,Niters):
    
    #This Algorithm is the Metropolis-Hastings within Gibbs sampling algorithm that is 
    #described on the middle of page 94 in Alex Neville's thesis.
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
        MCMC.append(p)
    #print("done")
    #plt.plot(MCMC)
    #Q.put(MCMC)
    return MCMC

from multiprocessing import Pool

if __name__=="__main__":
    cpus=multiprocessing.cpu_count()
    #print(cpus)
    chain_num=2

    with Pool(processes=cpus) as pool:
        for i in range(chain_num):
            #Change random seed otherwise candidate sampling will basically repeat and every chain will be essentially the same
            np.random.seed(i)
            result=pool.apply_async(Markov, (p_start,I[-1],))
            MCMC=np.array(result.get())
            #print(MCMC)
            #eta1_MCMC=MCMC[:,0]
            

            