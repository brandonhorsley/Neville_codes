"""
Code to implement the main parameter estimation procedure from Alex Neville's code which is described in 
the entirety of chapter 4.3.4.

This code is specific to the toy example case first, gen_MainCode's are generalised variants.

Current stage of troubleshooting means i have imposed proper bounds on parameter values but the MCMC exploration 
seems to not work well (high serial autocorrelation, see: 
https://www.statlect.com/fundamentals-of-statistics/Markov-Chain-Monte-Carlo-diagnostics#:~:text=The%20simplest%20way%20to%20diagnose,results%20on%20all%20the%20chunks.)
This needs to be done via either increasing number of chain samples and/or tune standard deviation of proposal 
distributions for a better random walk. But in order to do this i shall have to change the recommended values from 
his thesis which seems strange but i can't find any error in the implementation of the subalgorithms...

For further work it sounds like in the thesis he refers to randomly calibrating circuit using randomly 
selecting eta values and a values and b values. But as my code is i am just using predefined true values for 
eta, a and b that doesn't change. I can edit the code to implement this as a benchmarking visual.

Alg4 where the Markov chain is being construted takes the most time but it seems very nontrivial to speedup. 
So am making do for the time being.
"""

import numpy as np
from Aux_Nev import * 
import matplotlib.pyplot as plt

"""
This early version of my code has me making use of the Aux_Nev file that has some of the useful functions and 
values defined so changing them needs to be done in Aux_Nev.py, inconvenient but also helped stop the code running 
on too long and being confusing. This was dealt with when i went with my gen_MainCode developments.

Aux_Nev defines the following functions:
Construct_BS
Construct_PS
ConstructU
DataGen
Likelihood -> Note that this returns a sum of log probabilities such that the transition rules work and likelihoods aren't so small that they get rounded to zero


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
"""

#Data generation (step 1 of Alex's protocol) stuff is all contained in Aux_Nev so i will leave it that way for now for readability 
#but if i need differed functionality then i can bring the code into this code document.

###Preliminaries###
"""
Here i define a bunch of functions defining the distributions used in the subalgorithms later.

normal gets the probability for a value 'x' under a normal distribution of mean 'mu' and standard deviation 'sigma'.
uniform gets the probability of value 'p' where the uniform distribution is bounded between 'lower' and 'upper'. This 
    function checks that p is between upper and lower otherwise the probability will be zero.
random_coin is used for the acceptance rule in the subalgorithms, where we have a probability of transitioning to 
    the new state and this function is that act of rolling that die to determine whether you actually make that move.
"""

def normal(x,mu,sigma):
    numerator = np.exp((-(x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator

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

#eq 4.11: g_i(p',p)=Normal(p_i,sigma_i) #proposal distribution
#sigma_i=pi/200 for a, b_est for b, 0.005 for eta

#CAVEAT
# Note with regard to the above standard deviations for the proposal distributions for the different parameter 
# types is that there is some confusion with b_sigma. On page 94, it suggests that b_sigma should be b_est for 
# which given my choice of 'true values' based around the example true value shown in Figure 4.7 which is ~0.7. 
# However, on page 104 it says that the sigma for b's proposal distribution is 0.07. So it could just be a typo 
# but there is ambiguity there. Switching between b_sigma being 0.7 or 0.07 is something i think i checked but 
# haven't recently.

b_est=0.7
eta_sigma=0.005
a_sigma=np.pi/200
b_sigma=b_est #Based around true values from Neville_thesis_8.py

#I=[2,500,50,50,500,100,100,100000] #Determines iteration number for each algorithm call
I=[2,500,50,50,500,100,100,1000] #Smaller MCMC chain for troubleshooting

#Runtime estimate
runtime=(I[-1]/10)+60
print("runtime in seconds is around {}s".format(runtime))
print("runtime in minutes is around {}min".format(runtime/60))
print("runtime in hours is around {}hr".format(runtime/(3600)))

###Burn in###

p_alpha=[0.5,0.5,0.5,0,0,0.7,0.7] #step 2.1
print("p_alpha initial is {}".format(p_alpha))

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
                if new_element>-np.pi and new_element<np.pi: #enforcing bound on 'a' value
                    p_prime=list(p_alpha) 
                    p_prime[i]=new_element #new proposed state
                    #Likelihood
                    L1=Likelihood(p_alpha,V1,V2)
                    L2=Likelihood(p_prime,V1,V2)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    #P1= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
                    P1=uniform(p_alpha[i],-np.pi,np.pi) #since all other prior terms wind up cancelling out in the transition rule
                    P2=uniform(p_prime[i],-np.pi,np.pi) #since all other prior terms wind up cancelling out in the transition rule
                    #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                    #Candidates
                    g1= np.random.normal(p_alpha[i],a_sigma) #proposal distribution for current state
                    g2=np.random.normal(p_prime[i],a_sigma) #proposal distribution for proposed state
                    numerator=L1*P1*g1 #numerator for transition rule
                    denominator=L2*P2*g2 #denominator for transition rule
                    elem=numerator/denominator #fraction for transition rule
                    T=min(1,elem) #transition rule
                    move_prob=random_coin(T) #to move or not to move that is the question
                    if move_prob: #if move_prob is True then move happens so new proposed state is accepted
                        p_alpha=p_prime
                else: #otherwise proposed state is rejected and stays at current state
                    p_alpha=p_alpha

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
                if new_element>-np.pi and new_element<np.pi: #enforcing bound on 'a' value
                    p_prime=list(p_beta)
                    p_prime[i]=new_element #new proposed state
                    #Likelihood
                    L1=Likelihood(p_beta,V1,V2)
                    L2=Likelihood(p_prime,V1,V2)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    #P1= normal(p_beta[0],0.5,0.05)*normal(p_beta[1],0.5,0.05)*normal(p_beta[2],0.5,0.05)*uniform(p_beta[3])*uniform(p_beta[4])*normal(p_beta[5],0.7,0.07)*normal(p_beta[6],0.7,0.07) #Prior for p
                    P1=uniform(p_beta[i],-np.pi,np.pi)
                    P2=uniform(p_prime[i],-np.pi,np.pi)
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
                else:
                    p_beta=p_beta
            if i in [5,6]: #If it is b's
                new_element=np.random.normal(loc=p_beta[i],scale=b_sigma) #draw random sample from proposal distribution
                p_prime=list(p_beta)
                p_prime[i]=new_element #new proposed state
                #Likelihood
                L1=Likelihood(p_beta,V1,V2)
                L2=Likelihood(p_prime,V1,V2)
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

"""
Alg4 (MH-within-Gibbs) for p shows up twice in Alex's protocol, the first is for burn-in so we just run the 
algorithm and we only care about the p that we get at the end, and the other is using it for MCMC chain generation 
so we want to track the p at each sampling step. As such i have been a clever clogs and made one function be used 
for both and it just gets called differently for each method of use. Upon reflection i don't think i necessarily 
need to have both the Markov and ReturnAll function but either way it still works fine. 

So for the first case, as can be seen when i get to the actual algorithm further down the code, when not used for 
MCMC i just call it with the argument Markov=False which means it enters the 'else' bracket of code with the 
default argument for ReturnAll being False then that means it won't append the p result at each step into the MCMC 
array and at the end of sampling it will just return the final p rather than the MCMC array. Meanwhile of course 
when Alg4 is used for MCMC chain generation then i use Markov=True which means it enters the 'if' bracket which 
basically just sets ReturnAll=True and Markov=False as it calls the function again and so this time enters 
the 'else' bracket but because ReturnAll is set to True it aggregates all the p's into the MCMC array and because 
ReturnAll is true it returns the MCMC array as an output.
"""
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
        #AP below is to track if on each step the move is accepted or not so that the total acceptance rate can 
        # be found since some works point to an optimal total acceptance rate for metropolis-hastings being ~30%, 
        # although this is MH-within-Gibbs...
        AP=[]
        for n in range(Niters):
            for i in range(len(p)):
                if i in [0,1,2]: #If it is eta's
                    new_element=np.random.normal(loc=p[i],scale=eta_sigma) #draw random sample from proposal distribution
                    if new_element>0 and new_element<1: #enforcing bound on 'eta' value
                        p_prime=list(p)
                        p_prime[i]=new_element #new proposed state
                        #Likelihood
                        L1=Likelihood(p,V1,V2)
                        L2=Likelihood(p_prime,V1,V2)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        #P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
                        P1=normal(p[i],0.5,eta_sigma)
                        #P2= normal(p_prime[0],0.5,0.05)* normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                        P2=normal(p_prime[i],0.5,eta_sigma)
                        #Candidates
                        g1= np.random.normal(p[i],eta_sigma)
                        g2=np.random.normal(p_prime[i],eta_sigma)
                        elem=(np.exp(L1)*P1*g1)/(np.exp(L2)*P2*g2)
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            AP.append(move_prob)
                            p=p_prime
                    else: #AKA if eta value is out of bounds
                        AP.append(False)
                        p=p
                if i in [3,4]: #If it is a's
                    new_element=np.random.normal(loc=p[i],scale=a_sigma) #draw random sample from proposal distribution
                    if new_element>-np.pi and new_element<np.pi: #enforcing bound on 'a' value
                        p_prime=list(p)
                        p_prime[i]=new_element #new proposed state
                        #Likelihood
                        L1=Likelihood(p,V1,V2)
                        L2=Likelihood(p_prime,V1,V2)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        #P1= normal(p[0],0.5,0.05)*normal(p[1],0.5,0.05)*normal(p[2],0.5,0.05)*uniform(p[3])*uniform(p[4])*normal(p[5],0.7,0.07)*normal(p[6],0.7,0.07) #Prior for p
                        P1=uniform(p[i],-np.pi,np.pi)
                        #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                        P2=uniform(p_prime[i],-np.pi,np.pi)
                        #Candidates
                        g1= np.random.normal(p[i],a_sigma)
                        g2=np.random.normal(p_prime[i],a_sigma)
                        numerator=L1*P1*g1
                        denominator=L2*P2*g2
                        elem=numerator/denominator
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            AP.append(move_prob)
                            p=p_prime
                    else:
                        AP.append(False)
                        p=p
                if i in [5,6]: #If it is b's
                    new_element=np.random.normal(loc=p[i],scale=b_sigma) #draw random sample from proposal distribution
                    p_prime=list(p)
                    p_prime[i]=new_element #new proposed state
                    #Likelihood
                    L1=Likelihood(p,V1,V2)
                    L2=Likelihood(p_prime,V1,V2)
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
                        AP.append(move_prob)
                        p=p_prime
                    else:
                        AP.append(move_prob)
                        p=p
            if ReturnAll:
                MCMC.append(p)
        print("AP is")
        print(sum(AP)/len(AP)) #calculating the final acceptance probability
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
        p_prime=list(p_alpha) #new proposed state
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
                #P2= normal(p_prime[0],0.5,0.05)*normal(p_prime[1],0.5,0.05)*normal(p_prime[2],0.5,0.05)*uniform(p_prime[3])*uniform(p_prime[4])*normal(p_prime[5],0.7,0.07)*normal(p_prime[6],0.7,0.07) #prior for p'
                P2=uniform(p_prime[i],-np.pi,np.pi)
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
        L1=Likelihood(test,V1,V2)
        L2=Likelihood(p_alpha,V1,V2)
        #Priors
        #eta: mu=0.5,sigma=0.05
        #a: uniform so N/A
        #b: mu=0.7,sigma=0.07
        #P1= normal(test[0],0.5,0.05)*normal(test[1],0.5,0.05)*normal(test[2],0.5,0.05)*uniform(test[3])*uniform(test[4])*normal(test[5],0.7,0.07)*normal(test[6],0.7,0.07) #Prior for p+q
        P1=uniform(test[3],-np.pi,np.pi)*uniform(test[4],-np.pi,np.pi)
        #P2= normal(p_alpha[0],0.5,0.05)*normal(p_alpha[1],0.5,0.05)*normal(p_alpha[2],0.5,0.05)*uniform(p_alpha[3])*uniform(p_alpha[4])*normal(p_alpha[5],0.7,0.07)*normal(p_alpha[6],0.7,0.07) #Prior for p
        P2=uniform(p_alpha[3],-np.pi,np.pi)*uniform(p_alpha[4],-np.pi,np.pi)
        if (L1+np.log(P1))>(L2+np.log(P2)):
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

p_conv=list(p_zero) #step 2.7
print("p_conv is: {}".format(p_conv))


###Main Markov Chain Generation###

#Step 3
chain=Alg4(p_conv,I[7], Markov=True)

###Parameter estimation###

"""
This chain object should contain all the values for each parameter at each 
markov chain state number (i.e. I[7] by 7 matrix).
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
Plot(chain)

for i in range(len(p_conv)): #step 4
    par_array=chain[:,i]
    #Plot markov chain plot
    print(names[i])
    #plt.plot(par_array)
    print("Mean is {}".format(np.mean(par_array)))
    print("Standard deviation is {}".format(np.std(par_array)))
    
###########################################
#Testing, not included in Alex's protocol


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