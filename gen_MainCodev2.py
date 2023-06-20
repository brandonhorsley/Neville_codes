"""
This code file i have opened as a seperate version to gen_MainCode where i will generalise to more output modes. 
Some parts should be easy like defining the unitaries but the automation of input will be more awkward since i 
would need beamsplitters as well as what modes they will be across as well as perhaps imposing simple checks like 
not allowing beamsplitters to be over non-adjacent modes.

I have decided that i will reflect the input description of the real circuit i am trying to simulate by having an 
array of strings where each string element reads out the circuit top to down and left to right like how quandela's 
perceval language gets the stuff inputted. The string array should look as follows:
elements=["BS,1,2","BS,3,4","PS,1","PS,3"]
I would put the terms in brackets for visual distinguishability but the way i see it i can destringify each element 
of the array and pass that destringified version to a function as an argument.
So then the next thing is that destringifying ain't that simple, regex split() functionality should be good to 
maybe split up the string element and fortunately the beamsplitter argument doesn't need to be converted since i 
can just use that as a reference point to call e.g. the beamsplitter function and then use int() on the modes to 
get those modes as the arguments.

Does BS(1,2)=BS(2,1), I imagine it should be right?

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

#Define formula for Phase shifter unitary, will define just dim=2 for now
def construct_PS(phi):
    mat=np.array(([np.exp(1j*phi/2),0],[0,np.exp(-1j*phi/2)]))
    return mat

def construct_BS(eta):
    mat=np.array(([np.sqrt(eta), 1j*np.sqrt(1-eta)],[1j*np.sqrt(1-eta),np.sqrt(eta)]))
    return mat


N=4 #Number of modes, won't bother implementing a compatability check with circuit input
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
elements=["BS,1,2,{eta1}","BS,3,4,{eta2}","PS,1,{a1},{b1}","PS,3,{a2},{b2}".format(eta1=eta1_true,eta2=eta2_true,a1=a1_true,a2=a2_true,b1=b1_true,b2=b2_true)] #new format where true values are included
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

#print(proper)

#Yay!!!

"""
now comes the tricky bit since i need to keep a general function and now how i am passing the arguments to it since with more modes and need to track mode number as well as element type as well as relevant values.

Actually now i think of it i don't really need to keep it general, i can use the current existence of the circuit as an out of function thing and then just plug in the values after the fact.
"""

#Unitary definition
#https://github.com/clementsw/interferometer/blob/master/interferometer/main.py

def constructU_from_p(etas,phis):
    U=np.eye(N)
    BS_counter=-1
    PS_counter=-1
    for i in range(len(proper)):
        if proper[-(i+1)][0]=='BS':
            T=np.zeros((N,N))
            T[proper[-(i+1)][1]-1,proper[-(i+1)][1]-1]=np.sqrt(etas[BS_counter])
            T[proper[-(i+1)][1]-1,proper[-(i+1)][2]-1]=1j*np.sqrt(1-etas[BS_counter])
            T[proper[-(i+1)][2]-1,proper[-(i+1)][1]-1]=1j*np.sqrt(1-etas[BS_counter])
            T[proper[-(i+1)][2]-1,proper[-(i+1)][2]-1]=np.sqrt(etas[BS_counter])
            U=np.matmul(T,U)
            BS_counter-=1
        if proper[-(i+1)][0]=='PS':
            T=np.zeros((N,N))
            T[proper[-(i+1)][1]-1,proper[-(i+1)][1]-1]=np.exp(1j*phis[PS_counter]/2)
            T[proper[-(i+1)][2]-1,proper[-(i+1)][2]-1]=np.exp(-1j*phis[PS_counter]/2)
            U=np.matmul(T,U)
            PS_counter-=1
    return U
    