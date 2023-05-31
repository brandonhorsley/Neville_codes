"""
This code will be dedicated to generalising Aux_Nev.py so it isn't specifically built around the toy example.
"""

import numpy as np
import scipy
from collections import OrderedDict
#####################Generating Data#######################

RANDOM_SEED = 8927 #Seed for numpy random
rng = np.random.default_rng(RANDOM_SEED)

#########################Supporting functions

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

"""
Unitary construction is where there is a bit of a hitch since the unitary depends on the actual 
circuit so my model for this would need to account for positioning maybe?
I can add sophistication later so for now i will just settle for doing a chain of beamsplitters
and phase shifters like with the toy example.

A case i could pose could be an ordered dictionary? e.g. BS3@PS2@BS2@PS1@BS1 circuit with values 
0.51,-pi,0.49,pi,0.5 respectively then **kwargs dictionary would be:
{BS:0.51,PS: -pi, BS:0.49,PS: pi, BS:0.5}.
N.B. that dictionary generation should be the reverse of how the circuit reads left to right.
"""
"""
def constructU(*eta,*phi):
    #U=construct_BS(eta3)@construct_PS(phi2)@construct_BS(eta2)@construct_PS(phi1)@construct_BS(eta1)
    U=np.eye(2)
    for i, j in zip(eta, phi):
        U=U@construct_BS(i)@construct_PS(j)
    U=U@construct_BS(eta[-1]) #To get the last beamsplitter since zip excludes it.
"""

"""
#Commented out this code block since i have discovered that python dictionaries 
#don't support duplicate keys so i will need to find another way.

def constructU(**kwargs):
    #*kwargs should be ordered appropriately
    U=np.eye(2)
    for key,value in kwargs.items():
        if key=='BS':
            U=U@construct_BS(value)
        if key=='PS':
            U=U@construct_PS(value)
        print(value)
    return U
#Ordered circuit reading left to right
circuit={'BS':0.5,'PS':np.pi,'BS':0.49,'PS':-np.pi,'BS':0.51} 
#Reversed circuit ordering since unitary is a product of circuit terms from right to left
circuit_rev = OrderedDict(reversed(list(circuit.items())))
print(constructU(**circuit_rev))
"""

#https://stackoverflow.com/questions/10664856/how-can-one-make-a-dictionary-with-duplicate-keys-in-python

"""
So making a list in a dictionary is useful but then i need to think about how will i preserve 
some sense of total order, perhaps i could create a separate list of total order, such as
[BS,PS,PS,BS,BS,PS] and then use that to steer how i address the circuit dictionary, naturally remembering to reverse the
circuit ordering in my constructU function
"""

from collections import defaultdict

a1_true=0
a2_true=0

b1_true=0.788
b2_true=0.711

eta1_true=0.447
eta2_true=0.548
eta3_true=0.479

#circuit_ordering=[('BS',0.5),('PS',np.pi),('BS',0.49),('PS',-np.pi),('BS',0.51)]
#Phase shifter stores a,b as [a,b], i will need to remember this in future components.
#Additionally i will have to remember that circuit ordering and total ordering is reading from left to right
circuit_ordering=[('BS',eta1_true),('PS',[a1_true,b1_true]),('BS',eta2_true),('PS',[a2_true,b2_true]),('BS',eta3_true)]
circuit = defaultdict(list)
totalorder=[]
for k, v in circuit_ordering:
    circuit[k].append(v)
    totalorder.append(k)

#print(circuit)
#print(totalorder)

#constructU will take calculated phi value from datagen so i can keep constructU generic
#I will build my function like i pass a unitary dictionary to it where it accepts eta values and phi values
#e.g.U_dict={'BS':[BS3,BS2,BS1],'PS':[PS2,PS1]}
def constructU(**kwargs):
    U=np.eye(2)
    BS_counter=1
    PS_counter=1
    for i in range(len(totalorder)):
        if i=='BS':
            U=U@construct_BS(circuit[-BS_counter])
            BS_counter+=1
        if i=='BS':
            U=U@construct_PS(circuit[-PS_counter])
            PS_counter+=1
    return U

Vmax=5
N=100 #Top of page 108 ->N=number of experiments
M=2 #Number of modes
V1=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter 1
#V1=V1+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
V2=np.random.uniform(low=0, high=Vmax,size=N) #random voltage between 1 and 5 for phase shifter 2
#V2=V2+rng.normal(scale=0.02, size=N) #Adding gaussian noise, top of page 108 says 2% voltage noise
#Voltage array being passed
V=[]
V.append(V1)
V.append(V2)

#*V should be length same as number of phase shifters
"""
but then to generate phi values i need to relate a,b values as well as the voltage across each phase shifter for a given experiment so what may be necessary is another default_dict
like {'BS':[BSarray],'PS':[[a1,b1],[a2,b2]],V:[[V1array],[V2array]]}.
"""

def DataGen(InputNumber,*V,poissonian=False):
    data=np.empty((N,M))
    C=np.empty(N)

    for i in range(N):
        PS_length=len(V) #should equal 2 in this case
        PS=[]
        for _ in range(PS_length):
            PS.append()
