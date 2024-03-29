"""
This code file i have opened as a seperate version to gen_MainCode where i will generalise to more output modes. 
Some parts should be easy like defining the unitaries but the automation of input will be more awkward since i 
would need beamsplitters as well as what modes they will be across as well as perhaps imposing simple checks like 
not allowing beamsplitters to be over non-adjacent modes.

I have decided that i will reflect the input description of the real circuit i am trying to simulate by having an 
array of strings where each string element reads out the circuit top to down and left to right like how quandela's 
perceval language gets the stuff inputted. The string array should look as follows:
elements=["BS,1,2","BS,3,4","PS,1","PS,3"]
    ====
---|    |--PS--
   | BS |
---|    |------
    ====
---|    |--PS--
   | BS |
---|    |------
    ====

I would put the terms in brackets for visual distinguishability but the way i see it i can 'destringify' each element 
of the array and pass that 'destringified' version to a function as an argument.
So then the next thing is that destringifying ain't that simple, regex split() functionality should be good to 
maybe split up the string element and fortunately the beamsplitter argument doesn't need to be converted since i 
can just use that as a reference point to call e.g. the beamsplitter function and then use int() on the modes to 
get those modes as the arguments.

Changing my procedure to include the true values in elements array since it just makes sense to keep them together 
and i can slice out the relevant parts when it gets to the parameter part of the code.
"""

import numpy as np
import scipy
import time

#####################Generating Data#######################

RANDOM_SEED = 8927 #Seed for numpy random
rng = np.random.default_rng(RANDOM_SEED)

#########################Supporting functions
#Function for splitting strings with elements separated by commas
def extraction(string):
    output_list=[]
    if isinstance(string,str):
        output_list.append(','.join(string.split()))
    return output_list    
#print(extraction("a,b,c"))

#Function for removing dictionary elements
def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


M=4 #Number of modes, won't bother implementing a compatability check with circuit input
"""
The role of the state vectors that should be here are to project from the unitary into each of the output modes 
but the dynamic nature of the number of modes means that i may as well just do that in the data generation part.
"""

from collections import defaultdict

a1_true=0
a2_true=0.1

b1_true=0.75
b2_true=0.77

eta1_true=0.49
eta2_true=0.51


import re

#elements=["BS,1,2","BS,3,4","PS,1","PS,3"] #Old format
#elements=["BS,1,2,{eta1}","BS,3,4,{eta2}","PS,1,{a1},{b1}","PS,3,{a2},{b2}"]
#elements=elements.format(eta1=eta1_true,eta2=eta2_true,a1=a1_true,a2=a2_true,b1=b1_true,b2=b2_true) #new format where true values are included
elements=["BS,1,2,0.49","BS,3,4,0.51","PS,1,0,0.75","PS,3,0.1,0.77"]
#print(elements)

"""
proper=[]
for elem in elements:
    #split into substrings
    word_list=re.split(r",",elem)
    if word_list[0]=="BS":
        #intifying mode values
        word_list[1]=int(word_list[1])
        word_list[2]=int(word_list[2])
        #check for adjacency and thus validity
        _=word_list[1]-word_list[2]
        if _==1 or _==-1:
            pass
        else:
            print("One of your beamsplitter modes aren't defined adjacently. Exiting.")
            break
        #print(word_list[2])
        proper.append(word_list)
    if word_list[0]=="PS":
        #intifying mode value
        word_list[1]=int(word_list[1])
        #Adjacency doesn't need to be done since it is only one mode
        proper.append(word_list)
"""

proper=[]
for elem in elements:
    #split into substrings
    word_list=re.split(r",",elem)
    if word_list[0]=="BS":
        #intifying mode values
        word_list[1]=int(word_list[1])
        word_list[2]=int(word_list[2])
        #floatifying true value
        word_list[3]=float(word_list[3]) #eta
        #check for adjacency and thus validity
        _=word_list[1]-word_list[2]
        if _==1 or _==-1:
            pass
        else:
            print("One of your beamsplitter modes aren't defined adjacently. Exiting.")
            break
        #print(word_list[2])
        proper.append(word_list)
    if word_list[0]=="PS":
        #intifying mode value
        word_list[1]=int(word_list[1])
        #floatifying true values
        word_list[2]=float(word_list[2]) #a
        word_list[3]=float(word_list[3]) #b
        #Adjacency doesn't need to be done since it is only one mode
        proper.append(word_list)
#print("###########################")
#print(proper)
#print("###########################")

#Yay!!!

"""
now comes the tricky bit since i need to keep a general function and now how i am passing the arguments to it since with more modes and need to track mode number as well as element type as well as relevant values.

Actually now i think of it i don't really need to keep it general, i can use the current existence of the circuit as an out of function thing and then just plug in the values after the fact.
"""

#Unitary definition
#https://github.com/clementsw/interferometer/blob/master/interferometer/main.py

def constructU_from_p(etas,phis):
    U=np.eye(M)
    BS_counter=-1
    PS_counter=-1
    for i in range(len(proper)):
        if proper[-(i+1)][0]=='BS':
            T=np.zeros((M,M),dtype=np.complex128) #for some reason i needed to explicitly state dtype, throws an error otherwise
            #print(proper[-(i+1)][1])
            #print(proper[-(i+1)][2])
            #print(etas)
            #print(phis)
            #print(np.sqrt(etas[BS_counter])) #list of length 2
            x=np.sqrt(etas[BS_counter])
            y=1j*np.sqrt(1-etas[BS_counter])
            T[proper[-(i+1)][1]-1,proper[-(i+1)][2]-1]=y
            T[proper[-(i+1)][2]-1,proper[-(i+1)][1]-1]=y
            T[proper[-(i+1)][1]-1,proper[-(i+1)][1]-1]=x
            T[proper[-(i+1)][2]-1,proper[-(i+1)][2]-1]=x
            U=np.matmul(T,U)
            BS_counter-=1
        if proper[-(i+1)][0]=='PS':
            T=np.zeros((M,M))
            #print(proper[-(i+1)])
            #print(etas)
            #print(phis)
            #print(np.exp(1j*phis[PS_counter]))
            T[proper[-(i+1)][1]-1,proper[-(i+1)][1]-1]=np.exp(1j*phis[PS_counter])
            U=np.matmul(T,U)
            PS_counter-=1
    return U

Vmax=5
N=100 #Top of page 108 ->N=number of experiments

#print(proper)

circuit_list1=[]
circuit_list2=[]
for q in range(len(proper)):
    circuit_list1.append(proper[q][0])
    if proper[q][0]=="BS":
        test=proper[q][-1]
    if proper[q][0]=="PS":
        test=[proper[q][2],proper[q][3]]
    #test=[proper[q][2],proper[q][2]]
    circuit_list2.append(test)
#print(circuit_list1)
#print(circuit_list2)

circuit_list=list(zip(circuit_list1,circuit_list2))
#print(circuit_list) #wrong

circuit = defaultdict(list)
totalorder=[]
for k, v in circuit_list:
    circuit[k].append(v)
    totalorder.append(k)

#print(circuit)
#print(totalorder)

V=[]
for _ in range(len(circuit['PS'])):
    Velem=V1=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter
    #Velem=Velem+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
    V.append(list(Velem))

expanded_dict= circuit.copy()
expanded_dict['V'].append(V)

"""
I will need to figure out how to proceed with P_clicks, in my previous variants i defined
P_click1 and P_click2, i reckon i shall have to just make an empty array of length same as the number of modes
"""

def DataGen(InputNumber,poissonian=False, **expanded_dict):
    data=np.empty((N,M))
    C=np.empty(N)

    for i in range(N):
        phis=[]
        for j in range(len(expanded_dict['PS'])):
            phis.append(expanded_dict['PS'][j][0]+expanded_dict['PS'][j][1]*expanded_dict['V'][0][j][i]**2)
        U_true=constructU_from_p(expanded_dict['BS'],phis)        
        #P_click1_true=abs(top_bra@U_true@top_ket)**2 #Probability of click in top
        #P_click1_true=P_click1_true[0][0]
        #P_click2_true=abs(bottom_bra@U_true@top_ket)**2 #Probability of click in bottom
        #P_click2_true=P_click2_true[0][0]
        #P_true=[P_click1_true,P_click2_true]
        #For this generalised approach i'm just going to use selection of elements from the unitary matrix instead of hard defining the state vectors
        #top_ket slices out leftmost column of unitary
        P_true=[]
        for k in range(M):
            P_click_true=abs(U_true[k][0]) #This is assuming input into top mode
            P_true.append(P_click_true)
        #n=C,p=P,x=array of clicks
        data[i]=scipy.stats.multinomial.rvs(n=InputNumber,p=P_true)
        #Need to add poissonian noise
        if poissonian==True:
            data[i]+=rng.poisson(size=len(data[i]))
        C[i]=np.sum(data[i])

    return data,C

data,C=DataGen(InputNumber=1000,poissonian=False, **expanded_dict)
#print(np.shape(data))
#print(data) #Correct
#print(C)

def Likelihood(**p_V):
    #To be called after data generation
    P=np.empty((N,M))
    prob=np.empty(N)

    for i in range(N):
        phis=[]
        for j in range(len(p_V['a'])):
            phis.append(p_V['a'][j]+p_V['b'][j]*p_V['V'][0][j][i]**2)
        etas=p_V['eta']
        U=constructU_from_p(etas,phis)
        #P_click1=np.abs(top_bra@U@top_ket)**2 #Probability of click in top
        #P_click1=P_click1[0][0]
        #P_click2=np.abs(bottom_bra@U@top_ket)**2 #Probability of click in bottom
        #P_click2=P_click2[0][0]
        #P[i]=[P_click1,P_click2]
        #For this generalised approach i'm just going to use selection of elements from the unitary matrix instead of hard defining the state vectors
        #top_ket slices out leftmost column of unitary
        P_true=[]
        for k in range(M):
            P_click_true=abs(U[k][0]) #This is assuming input into top mode
            P_true.append(P_click_true)
        P[i]=P_true
        #n=C,p=P,x=array of clicks
        prob[i]=np.log(scipy.stats.multinomial.pmf(x=data[i],n=C[i],p=P[i]))
        if np.isinf(prob[i]):
            prob[i]=0 #To bypass -inf ruining likelihood calculations.
    logsum=np.sum(prob)
    return logsum

##########################################################################################
#End of Aux Nev component
##########################################################################################

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

#I=[2,500,50,50,500,100,100,100_000] #Determines iteration number for each algorithm call, 100_000 allowed since python 3.6
I=[2,500,50,50,500,100,100,100]


###Burn in###

p_alpha_list=[]
for elem in totalorder:
    if elem=='BS':
        newterm=('eta',0.5)
        p_alpha_list.append(newterm)
    if elem=='PS':
        newterm1=('a',0)
        newterm2=('b',0.5)
        p_alpha_list.append(newterm1)
        p_alpha_list.append(newterm2)

p_alpha = defaultdict(list)
for k, v in p_alpha_list:
    p_alpha[k].append(v)

p_alpha['V'].append(V)


def Alg4_alpha(Niters,**p_alpha):
    """
    Algorithm variant of Algorithm 4 for estimating the alpha model (which involves 
    only learning the a values).
    """
    for n in range(Niters):
        for k,v in p_alpha.items():
            if k == 'a': #If it is a's
                for i in range(len(v)):
                    #new_element=(np.random.normal(loc=p_alpha['a'][i],scale=a_sigma))%(2*np.pi) #draw random sample from proposal distribution
                    new_element=np.random.normal(loc=p_alpha['a'][i],scale=a_sigma) #draw random sample from proposal distribution
                    p_prime=p_alpha.copy()
                    p_prime['a'][i]=new_element
                    #Likelihood
                    L1=Likelihood(**p_alpha)
                    L2=Likelihood(**p_prime)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    P1=uniform(p_alpha['a'][i])
                    P2=uniform(p_prime['a'][i])
                    #Candidates
                    g1= np.random.normal(p_alpha['a'][i],a_sigma)
                    #print(p_alpha['a'][i])
                    #print(a_sigma)
                    #print("--------------")
                    g2=np.random.normal(p_prime['a'][i],a_sigma)
                    #print(p_prime['a'][i])
                    #print(a_sigma)
                    #numerator=L1*P1*g1
                    #denominator=L2*P2*g2
                    #elem=numerator/denominator
                    #elem=(L1+np.log(P1)+np.log(g1))-(L2+np.log(P2)+np.log(g2))
                    elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                    #print(L1)
                    #print(np.log(P1))
                    #print("--------------")
                    #print(g1)
                    #print(np.log(g1)) #This seems to throw errors more often by yielding nan
                    #print(L1+np.log(P1)+np.log(g1))
                    #print("-------------------")
                    #print(L2+np.log(P2)+np.log(g2))
                    #print("-------------------")
                    #print(elem)
                    #elem=np.exp(elem)
                    #print(elem)
                    #print("##############")
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p_alpha=p_prime

    return p_alpha

def Alg4_beta(Niters, **p_beta):
    """
    Algorithm variant of Algorithm 4 for estimating the beta model (which involves 
    learning the a and b values).
    """
    for n in range(Niters):
        for k,v in p_beta.items():
            if k == 'a': #If it is a's
                for i in range(len(v)):
                    #new_element=(np.random.normal(loc=p_beta['a'][i],scale=a_sigma))%(2*np.pi) #draw random sample from proposal distribution
                    new_element=np.random.normal(loc=p_beta['a'][i],scale=a_sigma) #draw random sample from proposal distribution
                    p_prime=p_beta.copy()
                    p_prime['a'][i]=new_element
                    #Likelihood
                    L1=Likelihood(**p_beta)
                    L2=Likelihood(**p_prime)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    P1=uniform(p_beta['a'][i])
                    P2=uniform(p_prime['a'][i])                    
                    #Candidates
                    g1= np.random.normal(p_beta['a'][i],a_sigma)
                    g2=np.random.normal(p_prime['a'][i],a_sigma)
                    #numerator=L1*P1*g1
                    #denominator=L2*P2*g2
                    #elem=numerator/denominator
                    #elem=(L1+np.log(P1)+np.log(g1))-(L2+np.log(P2)+np.log(g2))
                    #print(elem)
                    #elem=np.exp(elem)
                    #print(elem)
                    elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                    #print(elem)
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p_beta=p_prime
            if k == 'b': #If it is b's
                for i in range(len(v)):
                    new_element=np.random.normal(loc=p_beta['b'][i],scale=b_sigma) #draw random sample from proposal distribution
                    p_prime=p_beta.copy()
                    p_prime['b'][i]=new_element
                    #Likelihood
                    L1=Likelihood(**p_alpha)
                    L2=Likelihood(**p_prime)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    P1=uniform(p_beta['b'][i])
                    P2=uniform(p_prime['b'][i])
                    #Candidates
                    g1= np.random.normal(p_beta['b'][i],b_sigma)
                    g2=np.random.normal(p_prime['b'][i],b_sigma)
                    #numerator=L1*P1*g1
                    #denominator=L2*P2*g2
                    #elem=numerator/denominator
                    #elem=(L1+np.log(P1)+np.log(g1))-(L2+np.log(P2)+np.log(g2))
                    #print(elem)
                    #elem=np.exp(elem)
                    #print(elem)
                    elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                    #print(elem)
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p_beta=p_prime
    return p_beta

"""
def Alg4(Niters,Markov=False,ReturnAll=False,**p):
    
    #This Algorithm is the Metropolis-Hastings within Gibbs sampling algorithm that is 
    #described on the middle of page 94 in Alex Neville's thesis.
    
    if Markov: #i.e. if Markov==True
        chain=Alg4(Niters, Markov=False, ReturnAll=True,**p)
        #print(MCMC) #This is what generates the (wrong) repeated result
        return chain
    else: #If Markov==False
        #MCMC=[]
        #print(MCMC)
        blank1=[]
        blank2=[]
        #print("start of Alg 4 p is: {}".format(p))
        for n in range(Niters):
            for k,v in p.items():
                if k == 'eta': #If it is eta's
                    for i in range(len(v)):
                        new_element=np.random.normal(loc=p['eta'][i],scale=eta_sigma) #draw random sample from proposal distribution
                        p_prime=p.copy()
                        p_prime['eta'][i]=new_element
                        #Likelihood
                        L1=Likelihood(**p)
                        L2=Likelihood(**p_prime)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        P1=uniform(p['eta'][i])
                        P2=uniform(p_prime['eta'][i])
                        #Candidates
                        g1= np.random.normal(p['eta'][i],eta_sigma)
                        g2=np.random.normal(p_prime['eta'][i],eta_sigma)
                        #numerator=L1*P1*g1
                        #denominator=L2*P2*g2
                        #elem=numerator/denominator
                        #elem=(L1+np.log(P1)+np.log(g1))-(L2+np.log(P2)+np.log(g2))
                        #print(elem)
                        #elem=np.exp(elem)
                        #print(elem)
                        elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                        #print(elem)
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            p=p_prime
                if k == 'a': #If it is a's
                    for i in range(len(v)):
                        new_element=np.random.normal(loc=p['a'][i],scale=a_sigma) #draw random sample from proposal distribution
                        p_prime=p.copy()
                        p_prime['a'][i]=new_element
                        #Likelihood
                        L1=Likelihood(**p)
                        L2=Likelihood(**p_prime)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        P1=uniform(p['a'][i])
                        P2=uniform(p_prime['a'][i])
                        #Candidates
                        g1= np.random.normal(p['a'][i],a_sigma)
                        g2=np.random.normal(p_prime['a'][i],a_sigma)
                        #numerator=L1*P1*g1
                        #denominator=L2*P2*g2
                        #elem=numerator/denominator
                        #elem=(L1+np.log(P1)+np.log(g1))-(L2+np.log(P2)+np.log(g2))
                        #print(elem)
                        #elem=np.exp(elem)
                        #print(elem)
                        elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                        #print(elem)
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            p=p_prime
                if k == 'b': #If it is b's
                    for i in range(len(v)):
                        new_element=np.random.normal(loc=p['b'][i],scale=b_sigma) #draw random sample from proposal distribution
                        p_prime=p_alpha.copy()
                        p_prime['b'][i]=new_element
                        #Likelihood
                        L1=Likelihood(**p)
                        L2=Likelihood(**p_prime)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        P1=uniform(p['b'][i])
                        P2=uniform(p_prime['b'][i])                        
                        #Candidates
                        g1= np.random.normal(p['b'][i],b_sigma)
                        g2=np.random.normal(p_prime['b'][i],b_sigma)
                        #numerator=L1*P1*g1
                        #denominator=L2*P2*g2
                        #elem=numerator/denominator
                        #elem=(L1+np.log(P1)+np.log(g1))-(L2+np.log(P2)+np.log(g2))
                        #print(elem)
                        #elem=np.exp(elem)
                        #print(elem)
                        elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                        #print(elem)
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            p=p_prime
            #MCMC.append(removekey(p,'V'))
            #print(MCMC)
            #print("p on iteration {value1} is {value2}".format(value1=n,value2=p))
            #MCMC.append(removekey(p,'V'))
            #dic={str(n):n}
            #MCMC.append(dic)
            #print(p)
            #blank.append(p)
            ordering=[('a',n),('b',n+1),('a',n+2),('b',n+3)]
            #print(ordering)
            circuit = defaultdict(list)
            for k, v in ordering:
                circuit[k].append(v)
            #print(circuit)
            blank1.append(circuit)
            #print(blank1)
            #print("#################################")
            #print(p)
            #_=removekey(p,'V')
            #blank2.append(_)
            #print(blank2)
            #print("#################################")
            #print(type(circuit))
            #print(type(p))
            #print("----------------")
            #if ReturnAll:
            #    _=removekey(p,'V')
            #    MCMC.append(_)
            #    #print("MCMC 1 is")
                #print(MCMC) #So this generates the right result
            #print(MCMC)
        #print("#############################")
        #print(MCMC)
        #print("final p is {}".format(p))
        if ReturnAll:
            return MCMC
        else:
            return p

"""
#In this variant the MCMC functionality returns an array due to some strange error otherwise.
def Alg4(Niters,Markov=False,ReturnAll=False,**p):
    
    #This Algorithm is the Metropolis-Hastings within Gibbs sampling algorithm that is 
    #described on the middle of page 94 in Alex Neville's thesis.
    
    if Markov: #i.e. if Markov==True
        chain=Alg4(Niters, Markov=False, ReturnAll=True,**p)
        #print(MCMC) #This is what generates the (wrong) repeated result
        chain=np.stack(chain)
        return chain
    else: #If Markov==False
        MCMC=[]
        #print(MCMC)
        #print("start of Alg 4 p is: {}".format(p))
        for n in range(Niters):
            for k,v in p.items():
                if k == 'eta': #If it is eta's
                    for i in range(len(v)):
                        new_element=np.random.normal(loc=p['eta'][i],scale=eta_sigma) #draw random sample from proposal distribution
                        p_prime=p.copy()
                        p_prime['eta'][i]=new_element
                        #Likelihood
                        L1=Likelihood(**p)
                        L2=Likelihood(**p_prime)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        P1=uniform(p['eta'][i])
                        P2=uniform(p_prime['eta'][i])
                        #Candidates
                        g1= np.random.normal(p['eta'][i],eta_sigma)
                        g2=np.random.normal(p_prime['eta'][i],eta_sigma)
                        #numerator=L1*P1*g1
                        #denominator=L2*P2*g2
                        #elem=numerator/denominator
                        #elem=(L1+np.log(P1)+np.log(g1))-(L2+np.log(P2)+np.log(g2))
                        #print(elem)
                        #elem=np.exp(elem)
                        #print(elem)
                        elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                        #print(elem)
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            p=p_prime
                if k == 'a': #If it is a's
                    for i in range(len(v)):
                        #new_element=(np.random.normal(loc=p['a'][i],scale=a_sigma))%(2*np.pi) #draw random sample from proposal distribution
                        new_element=np.random.normal(loc=p['a'][i],scale=a_sigma) #draw random sample from proposal distribution
                        p_prime=p.copy()
                        p_prime['a'][i]=new_element
                        #Likelihood
                        L1=Likelihood(**p)
                        L2=Likelihood(**p_prime)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        P1=uniform(p['a'][i])
                        P2=uniform(p_prime['a'][i])
                        #Candidates
                        g1= np.random.normal(p['a'][i],a_sigma)
                        g2=np.random.normal(p_prime['a'][i],a_sigma)
                        #numerator=L1*P1*g1
                        #denominator=L2*P2*g2
                        #elem=numerator/denominator
                        #elem=(L1+np.log(P1)+np.log(g1))-(L2+np.log(P2)+np.log(g2))
                        #print(elem)
                        #elem=np.exp(elem)
                        #print(elem)
                        elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                        #print(elem)
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            p=p_prime
                if k == 'b': #If it is b's
                    for i in range(len(v)):
                        new_element=np.random.normal(loc=p['b'][i],scale=b_sigma) #draw random sample from proposal distribution
                        p_prime=p_alpha.copy()
                        p_prime['b'][i]=new_element
                        #Likelihood
                        L1=Likelihood(**p)
                        L2=Likelihood(**p_prime)
                        #Priors
                        #eta: mu=0.5,sigma=0.05
                        #a: uniform so N/A
                        #b: mu=0.7,sigma=0.07
                        P1=uniform(p['b'][i])
                        P2=uniform(p_prime['b'][i])                        
                        #Candidates
                        g1= np.random.normal(p['b'][i],b_sigma)
                        g2=np.random.normal(p_prime['b'][i],b_sigma)
                        #numerator=L1*P1*g1
                        #denominator=L2*P2*g2
                        #elem=numerator/denominator
                        #elem=(L1+np.log(P1)+np.log(g1))-(L2+np.log(P2)+np.log(g2))
                        #print(elem)
                        #elem=np.exp(elem)
                        #print(elem)
                        elem=np.exp(L1+np.log(P1)-L2-np.log(P2))*g1/g2
                        #print(elem)
                        T=min(1,elem)
                        move_prob=random_coin(T)
                        if move_prob:
                            p=p_prime
            
            #print("final p is {}".format(p))
            _=removekey(p,'V')
            #print(dictionary)
            etaarray=np.array(_['eta'])
            aarray=np.array(_['a'])
            barray=np.array(_['b'])

            array=np.concatenate((etaarray,aarray,barray))
            #array=np.array(array)
            MCMC.append(array)
            #print(p)
            #print(MCMC)
        #Next turn the reduced dictionary into a simple array for now

        if ReturnAll:
            return MCMC
        else:
            return p
        
def Alg5(Niters,**p_alpha):
    """
    This Algorithm is the Metropolised Independence Sampling (MIS) algorithm that is
    described at the top of page 98 in Alex Neville's thesis.
    """
    for n in range(Niters):
        p_prime=p_alpha.copy() #Need to use list, lesson learned after 4 years of coding
        new=[np.random.uniform(low=-np.pi,high=np.pi) for i in range(len(p_alpha['a']))]
        p_prime['a']=new
        #Likelihood
        L1=Likelihood(**p_alpha)
        L2=Likelihood(**p_prime)
        #Priors
        #eta: mu=0.5,sigma=0.05
        #a: uniform so N/A
        #b: mu=0.7,sigma=0.07
        P1=1
        for i in range(len(p_alpha['a'])):
            P1*=uniform(p_alpha['a'][i])
        P2=1
        for i in range(len(p_prime['a'])):
            P2*=uniform(p_prime['a'][i])
        #elem=(np.exp(L1)*P1)/(np.exp(L2)*P2)
        #elem=np.exp((L1-L2))*(P1/P2)
        #print(L1)
        #print(L2)
        #print(L1-L2)
        elem=(L1+np.log(P1))-(L2+np.log(P2))
        #print(elem)
        elem=np.exp(elem)
        #print((L1+np.log(P1)))
        #print((L2+np.log(P2)))
        #print((L1+np.log(P1))-(L2+np.log(P2)))
        #print(elem)
        #print('##################')
        T=min(1,elem)
        #np.log(1)=0,np.log(0)=-inf#
        #If negative then logT will be the value,otherwise logT will be zero so T will be 1
        #could run a modification so it is accept/reject rather than the most formal version of transition probabilities
        move_prob=random_coin(T)
        if move_prob:
            p_alpha=p_prime
    return p_alpha

def Alg6(Niters,**p_alpha):
    """
    This Algorithm is the Metropolised Independence Sampling (MIS) within Gibbs algorithm 
    that is described in the middle of page 98 in Alex Neville's thesis.
    """
    for n in range(Niters):
        for k,v in p_alpha.items():
            if k == 'a': #If it is a's
                for i in range(len(v)):
                    new_element=np.random.uniform(low=-np.pi,high=np.pi)
                    p_prime=p_alpha.copy()
                    p_prime['a'][i]=new_element
                    #Likelihood
                    L1=Likelihood(**p_alpha)
                    L2=Likelihood(**p_prime)
                    #Priors
                    #eta: mu=0.5,sigma=0.05
                    #a: uniform so N/A
                    #b: mu=0.7,sigma=0.07
                    P1=uniform(p_alpha['a'][i])
                    P2=uniform(p_prime['a'][i])
                    #numerator=L1*P1
                    #denominator=L2*P2
                    #elem=numerator/denominator
                    elem=(L1+np.log(P1))-(L2+np.log(P2))
                    #print(elem)
                    elem=np.exp(elem)
                    #print(elem)
                    T=min(1,elem)
                    move_prob=random_coin(T)
                    if move_prob:
                        p_alpha=p_prime
    return p_alpha

def Alg7(Niters, **p_alpha):
    """
    This Algorithm is the stochastic π kick search algorithm that is described 
    at the bottom of page 98 in Alex Neville's thesis.
    """
    for n in range(Niters):
        x=np.random.choice(3,len(p_alpha['a']))
        q=[(x[i]-1)*np.pi for i in range(len(x))]
        test=p_alpha.copy()
        for i in range(len(test['a'])):
            test['a'][i]+=q[i]
            #test['a'][i]=test['a'][i]%(2*np.pi)
        #Likelihood
        L1=Likelihood(**test)
        L2=Likelihood(**p_alpha)
        #Priors
        #eta: mu=0.5,sigma=0.05
        #a: uniform so N/A
        #b: mu=0.7,sigma=0.07
        P1=1
        for i in range(len(test['a'])):
            P1*=uniform(test['a'][i])
        P2=1
        for i in range(len(p_alpha['a'])):
            P2*=uniform(p_alpha['a'][i])
        #print(L1)
        #print(np.log(P1))
        #print(L2)
        #print(np.log(P2))
        #Am reworking the inequality since np.exp(likelihood) takes it to zero so i am logging the priors instead since it doesn't alter the inequality
        #print((L1+np.log(P1))-(L2+np.log(P2)))
        if (L1+np.log(P1))>(L2+np.log(P2)):
        #if (np.exp(L1)*P1)>(np.exp(L2)*P2):
            p_alpha=test
    return p_alpha

#Main code bulk

start=time.time()

for i in range(I[0]): #step 2.2
    #step 2.2i
    p_alpha=Alg5(I[1],**p_alpha)
    #print(p_alpha)
    #print("###step 2.2i done###")
    #step 2.2ii
    p_alpha=Alg6(I[2],**p_alpha)
    #print(p_alpha)
    #print("###step 2.2ii done###")
    #step 2.2iii
    p_alpha=Alg4_alpha(I[3],**p_alpha)
    #print(p_alpha)
    #print("###step 2.2iii done###")
    #step 2.2iv (and 2.2v)
    p_alpha=Alg7(I[4],**p_alpha)
    #print(p_alpha)
    #print("###step 2.2iv done###")

p_beta=p_alpha.copy()
for i in range(len(p_beta['eta'])):
    p_beta['eta'][i]=0.5
for i in range(len(p_beta['b'])):
    p_beta['b'][i]=0.7
#print("p_beta initial is: {}".format(p_beta))

#step 2.4
p_beta=Alg4_beta(I[5],**p_beta)
#print(p_beta)
#print("###step 2.4 done###")

#step 2.5
p_zero=p_beta.copy()
for i in range(len(p_zero['eta'])):
    p_zero['eta'][i]=0.5
#print("p_zero is: {}".format(p_zero))

#step 2.6
#p_zero=Alg4(I[6], Markov=False, ReturnAll=False, **p_zero)
p_zero=Alg4(I[6], Markov=False, **p_zero)
#print(p_zero)
#print("###step 2.6 done###")
#step 2.7
p_conv=p_zero.copy()
#print("p_conv is: {}".format(p_conv))

###Main Markov Chain Generation###

#Step 3
chain=Alg4(I[7], Markov=True,ReturnAll=True,**p_conv)

end=time.time()

print("runtime is {}".format(end-start))

###Parameter estimation###

"""
I imagine this chain object should contain all the values for each parameter at each 
markov chain state number (i.e. I[7] by 7 matrix).
To start with i shall just generate the markov chain plot and comment out the histogram 
plot and polish can later be applied to get the standard plot with the smoothed histogram in the left column and
markov state number plot in the right column with a Plot() function.
"""
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

#names=["eta1","eta2","eta3","a1","a2","b1","b2"]

#difficulty in setting names since i don't think i can use formatting to set a variable name :(
#so for now i will leave it
#i reckon for now i shall have to use the totalorder and count which occurrence 'BS' or 'PS' it is and then use that to format a string.

names=[]
for _ in range(len(p_conv['eta'])):
    names.append("eta{}".format(_+1))
for _ in range(len(p_conv['a'])):
    names.append("a{}".format(_+1))
for _ in range(len(p_conv['b'])):
    names.append("b{}".format(_+1))
#print(chain) #chain seems to be just a repetition of the same default dicts

def Plot(chain): #Chain should contain all necessary markov chain data
    """
    Custom plot function to generate the standard traceplot format where there is a smoothed
    histogram for each parameter in the left hand column and the markov chain plot in the right hand column.
    e.g. https://python.arviz.org/en/stable/examples/plot_trace.html
    """
    #nitems=len(p_alpha['eta'])+len(p_alpha['a'])+len(p_alpha['b'])
    #fig,axs=plt.subplots(nitems,2,constrained_layout=True) #Can use sharex,sharey for further polish if wanted
    fig,axs=plt.subplots(len(chain[0]),2,constrained_layout=True) #Can use sharex,sharey for further polish if wanted
    #for i in range(nitems):
    for i in range(len(chain[0])):
        #histogram
        #axs[i,0].hist(chain[:,i],bins=30)
        #smoothed kde with scott's rule bandwidth selection (bw selection is important consideration in kde)
        eval_points = np.linspace(np.min(chain[:,i]), np.max(chain[:,i]),len(chain[:,i]))
        kde=gaussian_kde(chain[:,i])
        evaluated=kde.evaluate(eval_points)
        evaluated/=sum(evaluated) #For normalisation
        axs[i,0].plot(eval_points,evaluated)
        #Add axs polish like axes labelling
        axs[i,0].set_ylabel(str(names[i])) #Add label to show which parameter is which
        axs[i,1].plot(chain[:,i])
        axs[i,1].set_xlabel("Markov chain State Number") #Aid understanding of Markov chain plot
    #fig.tight_layout()
    plt.show()

#print(chain)
"""
chain contains an array of default dictionaries so i sgould figure out how to handle this output.
"""

chain=np.array(chain)
#print(chain)
Plot(chain)

for i in range(len(chain[0])): #step 4
    par_array=chain[:,i]
    #Plot markov chain plot
    print(names[i])
    #plt.plot(par_array)
    print("Mean is {}".format(np.mean(par_array)))
    print("Standard deviation is {}".format(np.std(par_array)))