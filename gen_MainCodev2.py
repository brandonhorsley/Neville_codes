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
So then the next thing is that destringifying ain't that simple, regex split() functionality should be good to maybe split up the string element and fortunately the beamsplitter argument doesn't need to be converted since i can just use that as a reference point to call e.g. the beamsplitter function and then use int() on the modes to get those modes as the arguments
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

top_ket=np.array([1,0])
top_ket.shape=(2,1)

top_bra=np.array([1,0])
top_bra.shape=(1,2)

bottom_ket=np.array([0,1])
bottom_ket.shape=(2,1)

bottom_bra=np.array([0,1])
bottom_bra.shape=(1,2)

from collections import defaultdict

a1_true=0
a2_true=0

b1_true=0.788
b2_true=0.711

eta1_true=0.447
eta2_true=0.548
eta3_true=0.479

import re

elements=["BS,1,2","BS,3,4","PS,1","PS,3"]

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

#print(proper)

#Yay!!!


