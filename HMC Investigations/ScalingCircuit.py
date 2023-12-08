"""
This code will be for practicing building a larger circuit (follow Reck 
scheme for simplicity and scaling number of modes) so that it can be 
assessed how the protocol performs as circuit size scales. As my measure 
of performance i will follow what NEville does by randomly setting true 
values for the circuits and measuring average error across multiple runs 
and i can use this to build a plot of average error vs circuit size. 
Alongside this perhaps a runtime analysis could be warranted for comparison.

Active notes:

Automatic definitions will be tricky but maybe an unnecessary problem to 
solve since pymc needs variable name assignment but you can't really 
dynamically create variables as far as I am aware. Maybe I could find 
alternative means. How about saving time for myself by making generic code 
for Reck and finding p functions that work as long as there are the right 
number of eta's,a's and b's? Or maybe I can save hassle and just have 
separate codes that are hardcoded for each, e.g. Scheme_modes (so reck 
scheme on a 4 mode system would be reck_4.py). And I can still make a sympy 
calculator code that tells me the unitary and/or p equation to use so i don't 
have to figure it out by hand.

In this case then I will spend this code practicing by drafting up the 
necessary functions. Remember that for p i have to ensure that it remains 
real so alternative expression is key.

Made a start on Reck code, actually somewhat nontrivial, am decomposing eta's 
and phi's into AMZI but i need to track AMZI ordering too since when we get to 
4 modes, the top and bottom middle amzi are both at the same place which throws 
off the pattern so maybe i will be better off just doing it by hand rather 
than making a python function that finds the unitary I want, though it could 
still be a useful exercise for sympy stuff in the classical sampling 
investigation. So in summary I will either open up a separate sympy file to 
calculate the unitaries for a given number of modes case explicitly or just 
do it and my derivation of p by hand, i guess it depends how high i wish to 
go, for the longevity a computer way of doing it is going to be far better 
though. Which will then point to me still needing to write this python 
function.

What if my way of using code is to say use the component calculator to determine number of AMZIs, then make ordered dictionary of {AMZI1:[params],AMZI2:[params]...etc}, then it is a case of just needing to figure out which AMZI to go to next, could I maybe encode an ordering within to tell me which one would be next. Since the rule is right to left and down to up so to encode it's position in the circuit? Or if i just opt for upside down pyramid reck scheme then the actual assignment of the AMZIs to make the Reck scheme in the first place should then be my ordering? I just need to figure out how to navigate the pyramid in accordance with the rule. Or what if i just move AMZI to AMZI in sequence then the figuring out of which is which is just done in interpretation of results? This is still a pain in the ass since really logical numbering would be just reading components left to right starting from the top mode and working down to the bottom mode like is done with Alex Neville's figures.But this numbering is a lot less trivial if i do it at the last step of interpreting results. Maybe for ease of interpretation can i generate an associated latex diagram? This is quickly becoming very confusing, perhaps for ease it is just simpler to encode the specific pattern for r and c for up to some number of modes.
"""

import numpy as np
#import sympy as smp

def BS(eta):
    mat=np.array(([np.sqrt(eta), 1j*np.sqrt(1-eta)],[1j*np.sqrt(1-eta),np.sqrt(eta)]))
    return mat

def PS(phi):
    mat=np.array(([np.exp(1j*phi/2),0],[0,np.exp(-1j*phi/2)]))
    return mat

def AMZI(eta1,eta2,phi1,phi2):
    mat=PS(phi2)@BS(eta2)@PS(phi1)@BS(eta1)
    return mat

def ReckComponentCalculator(n:int):
    #Proposed by Reck et al. (1994)
    print("You have {} modes".format(n))
    AMZIs=n*(n-1)/2
    print("You will have {} AMZIs".format(AMZIs))
    print("Number of eta's: {}".format(AMZIs*2))
    print("Number of a's: {}".format(AMZIs*2))
    print("Number of b's: {}".format(AMZIs*2))

def ClementsComponentCalculator(n:int):
    #Proposed by Clements et al. (2016)
    print("You have {} modes".format(n))
    AMZIs=n*(n-1)/2
    print("You will have {} AMZIs".format(AMZIs))
    print("Number of eta's: {}".format(AMZIs*2))
    print("Number of a's: {}".format(AMZIs*2))
    print("Number of b's: {}".format(AMZIs*2))

def DiamondComponentCalculator(n:int):
    #For the Diamond scheme by Shokraneh et al. (2020)
    print("You have {} modes".format(n))
    AMZIs=(n-1)**2
    print("You will have {} AMZIs".format(AMZIs))
    print("Number of eta's: {}".format(AMZIs*2))
    print("Number of a's: {}".format(AMZIs*2))
    print("Number of b's: {}".format(AMZIs*2))

def Reck(*arg):
    """
    Reck means number of AMZI will be triangular number, so 2 modes will be 1 AMZI, 3 modes will be 3 AMZI's, 4 modes will be 6 AMZI's.
    sequence for triangular number is n(n+1)/2 where n is number of modes-1.
    Each AMZI will have 4 components and so given each phase shifter has a and b then AMZI_num=total comp /6 (assuming the number of eta's and a's and b's are appropriate).
    """
    AMZI_num=len(arg)/6 #number of AMZIs in circuit, if not an int then throw error
    n=np.roots([1,1,2*AMZI_num])+1 #number of modes calculated by solving for n in triangular sequence formula then adding 1
    U=np.eye(n)
    T=np.zeros((n,n))

    """
    need to find my way to index the args correctly and get the appropriate row and columns to index, maybe make arrays for r and c?
    """
    mat=AMZI()
    r,c=0,0 #this should be to index the appropriate mode numbers for where the AMZI is
    T[r:r+mat.shape[0], c:c+mat.shape[1]] += mat

def p_calculator(): #Maybe argument should be unitary and taking the input mode(s) and output mode(s) to index
    pass