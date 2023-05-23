"""
Code to implement the main parameter estimation procedure from Alex Neville's code which is described in 
the entirety of chapter 4.3.4.

I will keep it specific to the toy example case first and then generalise later.

Current draft yields poor results, which i think is due to malfunctioning of sub algorithms.

Through multiple iterations of troubleshooting i have managed to reel in answers into more reasonable  and 
physical answers (so eta bounded between 0 and 1 and so on). I also learnt after 4 years of coding that you 
can't just do a=b and then modify only a because it will change b too, so instead i need to do a=list(b).

For further work it sounds like in the thesis he refers to randomly calibrating circuit using randomly 
selecting eta values and a values and b values. But as my code is i am just using predefined true values for 
eta, a and b that doesn't change. I can edit the code to implement this when i figure out why alex chose to 
do that.

Alg4 where the Markov chain is being constructed seems to take the longest amount of time so if i can get as 
much speedup as possible then that will be extremely useful.
"""

import numpy as np
from Aux_Nev import *
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
                p_prime=list(p_alpha)
                p_prime[i]=new_element
                #Likelihood
                L1=Likelihood(p_alpha,V)
                L2=Likelihood(p_prime,V)
                #Priors
                #eta: mu=0.5,sigma=0.05
                #a: uniform so N/A
                #b: mu=0.7,sigma=0.07
                #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                P1=uniform(p_alpha[i])
                P2=uniform(p_prime[i])
                #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                
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
                p_prime=list(p_beta)
                p_prime[i]=new_element
                #Likelihood
                L1=Likelihood(p_beta,V)
                L2=Likelihood(p_prime,V)
                #Priors
                #eta: mu=0.5,sigma=0.05
                #a: uniform so N/A
                #b: mu=0.7,sigma=0.07
                #P1= normal(p_beta[0],0.5,0.05)*normal(p_beta[1],0.5,0.05)*normal(p_beta[2],0.5,0.05)*uniform(p_beta[3])*uniform(p_beta[4])*normal(p_beta[5],0.7,0.07)*normal(p_beta[6],0.7,0.07) #Prior for p
                P1=uniform(p_beta[i])
                P2=uniform(p_prime[i])
                #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
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
                p_prime=list(p_beta)
                p_prime[i]=new_element
                #Likelihood
                L1=Likelihood(p_beta,V)
                L2=Likelihood(p_prime,V)
                #Priors
                #eta: mu=0.5,sigma=0.05
                #a: uniform so N/A
                #b: mu=0.7,sigma=0.07
                #P1= normal(p_beta[0],0.5,0.05)*normal(p_beta[1],0.5,0.05)*normal(p_beta[2],0.5,0.05)*uniform(p_beta[3])*uniform(p_beta[4])*normal(p_beta[5],0.7,0.07)*normal(p_beta[6],0.7,0.07) #Prior for p_beta
                P1=normal(p_beta[i],0.7,0.07)
                #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p_beta'
                P2=normal(p_prime[i],0.7,0.07)
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
        L1=Likelihood(p_alpha,V)
        L2=Likelihood(p_prime,V)
        #Priors
        #eta: mu=0.5,sigma=0.05
        #a: uniform so N/A
        #b: mu=0.7,sigma=0.07
        #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
        P1=uniform(p_alpha[3])*uniform(p_alpha[4])
        #P2= normal(p_prime[0],0.5,0.05)* normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
        P2=uniform(p_prime[3])*uniform(p_prime[4])
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
                p_prime=list(p_alpha)
                p_prime[i]=new_element
                #Likelihood
                L1=Likelihood(p_alpha,V)
                L2=Likelihood(p_prime,V)
                #Priors
                #eta: mu=0.5,sigma=0.05
                #a: uniform so N/A
                #b: mu=0.7,sigma=0.07
                #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                P1=uniform(p_alpha[i])
                #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                P2=uniform(p_prime[i])
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
        test=list(p_alpha)
        test[3]+=q[0]
        test[4]+=q[1]
        L1=Likelihood(test,V)
        L2=Likelihood(p_alpha,V)
        #Priors
        #eta: mu=0.5,sigma=0.05
        #a: uniform so N/A
        #b: mu=0.7,sigma=0.07
        #P1= normal(test[0],0.5,0.05)*normal(test[1],0.5,0.05)*normal(test[2],0.5,0.05)*uniform(test[3])*uniform(test[4])*normal(test[5],0.7,0.07)*normal(test[6],0.7,0.07) #Prior for p+q
        P1=uniform(test[3])*uniform(test[4])
        #P2= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
        P2=uniform(p_alpha[3])*uniform(p_alpha[4])
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

names=["eta1","eta2","eta3","a1","a2","b1","b2"]

def Plot(chain): #Chain should contain all necessary markov chain data
    """
    Custom plot function to generate the standard traceplot format where there is a smoothed
    histogram for each parameter in the left hand column and the markov chain plot in the right hand column.
    e.g. https://python.arviz.org/en/stable/examples/plot_trace.html
    """
    fig,axs=plt.subplots(len(p_conv),2) #Can use sharex,sharey for further polish if wanted
    for i in range(len(p_conv)):
        axs[i,0].hist(chain[:,i],bins=30)
        #Add axs polish like axes labelling
        axs[i,0].set_ylabel(str(names[i])) #Add label to show which parameter is which
        axs[i,1].plot(chain[:,i])
        axs[i,1].set_xlabel("Markov chain State Number") #Aid understanding of Markov chain plot
    plt.show()

chain=np.array(chain)
Plot(chain)

for i in range(len(p_conv)): #step 4
    par_array=chain[:,i]
    #Plot markov chain plot
    print(names[i])
    #plt.plot(par_array)
    print("Mean is {}".format(np.mean(par_array)))
    print("Standard deviation is {}".format(np.std(par_array)))

