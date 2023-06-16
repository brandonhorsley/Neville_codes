"""
This document is being opened for the sake of my sanity. In my generalised 
main code i try to get it to append p from each MCMC iteration to an empty list 
called MCMC and it only seems to happen with appending p which is a defaultdict, 
the error doesn't replicate with dummy variables or even normal dictionaries so i 
am distilling the problem down to that basic element of generating a defaultdict 
and then appending it to an empty array and seeing that overwrite clear as day, and 
then perhaps i can move forward in knowing it simply has to be because it is a 
default dictionary and nothing to do with my Alg4 function.
"""
from collections import defaultdict
"""
ordering1=[('a',1),('b',2),('a',2),('b',4)]
circuit1 = defaultdict(list)

for k, v in ordering1:
    circuit1[k].append(v)


ordering2=[('a',5),('b',6),('a',7),('b',8)]
circuit2 = defaultdict(list)

for k, v in ordering2:
    circuit2[k].append(v)

blank=[]
#print(blank)
#blank.append(circuit1)
#print(blank)
#blank.append(circuit2)
#print(blank)
for n in range(2):
    ordering=[('a',n),('b',n+1),('a',n+2),('b',n+3)]
    print(ordering)
    circuit = defaultdict(list)
    for k, v in ordering:
        circuit[k].append(v)
    print(circuit)
    blank.append(circuit)
    print(blank)
"""
"""
Okay,so why doesn't my other code do this!?
"""

a1=[0,1]
b1=[2,3]

a2=[4,5]
b2=[6,7]

p1={'a':a1,'b':b1}
p2={'a':a2,'b':b2}

_=[]
print(_)
_.append(p1)
print(_)
p1=p2
_.append(p1)
print(_)