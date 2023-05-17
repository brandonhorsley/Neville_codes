"""
Code to implement the main parameter estimation procedure from Alex Neville's code which is described in 
the entirety of chapter 4.3.4.

I will keep it specific to the toy example case first and then generalise later.
"""

import numpy as np
from Aux_Nev import *

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
    if unif>=p:
        return False
    else:
        return True

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
N_iters=100000

I=[2,500,50,50,500,100,100,100000] #Determines iteration number for each algorithm call

###Burn in###

p_alpha=[0.5,0.5,0.5,0,0,0.5,0.5] #step 2.1

"""
Defining Algorithms from thesis. Have done other documents that implement them but i 
shall use those to define these functions which should give a specific output as wanted.

Starting w/ stub code for now though.
"""
#Procedure seems to refer to modifying this algorithm to target a specific p, perhaps i may need a target arg.
#Either that or define separate funcs for each variant but i don't think that will be necessary.
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
                    p_prime=p
                    p_prime[i]=new_element
                    #Likelihood
                    L1=Likelihood(p,V)
                    L2=Likelihood(p_prime,V)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
                    P2= normal(p_prime[0],0.5,0.05)* normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
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
                    p_prime=p
                    p_prime[i]=new_element
                    #Likelihood
                    L1=Likelihood(p,V)
                    L2=Likelihood(p_prime,V)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
                    P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
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
                    p_prime=p
                    p_prime[i]=new_element
                    #Likelihood
                    L1=Likelihood(p,V)
                    L2=Likelihood(p_prime,V)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
                    P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
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
        p_prime=[np.random.uniform(low=-np.pi,high=np.pi) for i in range(7)]
        if (0<=p_prime[0]<=1) and (0<=p_prime[1]<=1) and (0<=p_prime[2]<=1):
            #Likelihood
            L1=Likelihood(p_alpha,V)
            L2=Likelihood(p_prime,V)
            #Priors
            #eta: mu=0.5,sigma=0.05
            #a: uniform so N/A
            #b: mu=0.7,sigma=0.07
            P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
            P2= normal(p_prime[0],0.5,0.05)* normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
            elem=(np.exp(L1)*P1)/(np.exp(L2)*P2)
            print(elem)
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
    for n in range(N_iters):
        for i in range(len(p_alpha)):
            if i in [0,1,2]: #If it is eta's
                new_element=np.random.uniform(low=-np.pi,high=np.pi)
                if new_element>=0 and new_element<=1: #Check if value is legitimate
                    p_prime=p_alpha
                    p_prime[i]=new_element
                    #Likelihood
                    L1=Likelihood(p_alpha,V)
                    L2=Likelihood(p_prime,V)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                    P2= normal(p_prime[0],0.5,0.05)* normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                    elem=(np.exp(L1)*P1)/(np.exp(L2)*P2)
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p_alpha=p_prime

            if i in [3,4]: #If it is a's
                new_element=np.random.uniform(low=-np.pi,high=np.pi)
                p_prime=p_alpha
                p_prime[i]=new_element
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
                p_prime=p_alpha
                p_prime[i]=new_element
                #Likelihood
                L1=Likelihood(p,V)
                L2=Likelihood(p_prime,V)
                #Priors
                #eta: mu=0.5,sigma=0.05
                #a: uniform so N/A
                #b: mu=0.7,sigma=0.07
                P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                numerator=L1*P1
                denominator=L2*P2
                elem=numerator/denominator
                T=min(1,elem)
                move_prob=random_coin(T)
                if move_prob:
                    p=p_prime

    return p_alpha

def Alg7(p_alpha,Niters):
    """
    This Algorithm is the stochastic π kick search algorithm that is described 
    at the bottom of page 98 in Alex Neville's thesis.
    """
    for n in range(Niters):
        x=np.random.choice(3,len(p))
        q=[(x[i]-1)*np.pi for i in range(len(x))]
        #Likelihood
        L1=Likelihood(np.add(p_alpha,q),V)
        L2=Likelihood(p_alpha,V)
        #Priors
        #eta: mu=0.5,sigma=0.05
        #a: uniform so N/A
        #b: mu=0.7,sigma=0.07
        P1= normal(p_alpha[0]+q[0],0.5,0.05)*normal(p_alpha[1]+q[1],0.5,0.05)*normal(p_alpha[2]+q[2],0.5,0.05)*uniform(p_alpha[3]+q[3])*uniform(p_alpha[4]+q[4])*normal(p_alpha[5]+q[5],0.7,0.07)*normal(p_alpha[6]+q[6],0.7,0.07) #Prior for p+q
        P2= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
        if (np.exp(L1)*P1)>(np.exp(L2)*P2):
            p_alpha=np.add(p_alpha,q)
    return p_alpha

for i in range(I[0]): #step 2.2
    #step 2.2i
    p_alpha=Alg5(p_alpha,I[1])
    #step 2.2ii
    p_alpha=Alg6(p_alpha,I[2])
    #step 2.2iii
    p_alpha=Alg4(p_alpha, I[3],Markov=False)
    #step 2.2iv (and 2.2v)
    p_alpha=Alg7(p_alpha,I[4])

p_beta=[0.5,0.5,0.5,p_alpha[3],p_alpha[4],b_est,b_est] #step 2.3

#step 2.4
p_beta=Alg4(p_beta, I[5],Markov=False)

p_zero=[0.5,0.5,0.5,p_beta[3],p_beta[4],p_beta[5],p_beta[6]] #step 2.5

#step 2.6
p_zero=Alg4(p_zero,I[6], Markov=False)

p_conv=p_zero #step 2.7

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
        axs[i,0].hist(chain[:,i],normed=1,bins=30)
        #Add axs polish like axes labelling
        axs[i,0].set_ylabel(str(names[i])) #Add label to show which parameter is which
        axs[i,1].plot(chain[:,i])
        axs[i,1].set_xlabel("Markov chain State Number") #Aid understanding of Markov chain plot
    plt.show()

for i in range(len(p_conv)): #step 4
    par_array=chain[:,i]
    #Plot markov chain plot
    print(names[i])
    plt.plot(par_array)
    print("Mean is {}".format(np.mean(par_array)))
    print("Standard deviation is {}".format(np.std(par_array)))

