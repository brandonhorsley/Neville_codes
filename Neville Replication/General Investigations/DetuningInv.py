"""
Investigation to gauge the sensitivity of changing eta values and a values and b values. My initial 
investigation will be to just individually plot detuning from true against trace distance.

It occurs to me that it will actually be quite tough to incorporate things, i kinda got close with Tunable_4_3.py
where i investigated how changing etas affects the phi1 v phi2 plot which seemed to generate a particular relationship.
But then finding phi is awkward because it is a function of a and b and V. a will be linear with fixing b and V since 
it is just an offset term. So all that is left is thinking about b and V which in our model which is essentially
y=ax**2. So from simple parabolas, a sharpens the parabola so you need a smaller x to get a given value.
y=ax**2 so dy/dx=2ax

I will focus on investigating the relationship between y, a and x first and then this should complete my 
intuition around a,b,V affecting phi values, and my knowledge of eta values and phi values from 
Tunable_4_3.py

Active Notes:

- Plot completed, the key thing seems to be that increasing V means distance between peaks (where modulo kicks in) 
gets shorter and shorter, and lower b values means shallower rise so less distance sharp impact on phi. So then 
i think perhaps really sensitivity of the landscape is proportional to both b and V so such a procedure becomes 
less wise to do with larger b and V. a is just a matter of proportionality since it is an offset term. Each eta 
value in the toy example has a unique relationship to phi and the smearing that is seen interestingly seems to 
mean that more detuning of eta leads to less of an impact as phi values change...

- So all this intuition is only really directed towards the toy example but it could be worth thinking then about 
impact on performance as needing stricter reparameterisation and an even more precise estimate for larger b (and V). 
a is pretty agnostic. I think eta smearing is pretty reasonable for the toy example, i am also curious though about 
how eta sensitivity depends on number of elements, i would wager more components means more sensitivity...

- Primitive test on more components leads me to think maybe the dependence is actually the same 'individually', although
along some dimensions it doesn't change as much but all in all seems fairly reasonable. The phi1,phi2 peaks are still 
generally present sometimes they are just one peak or sometimes two.
"""

"""

#investigating b and V sensitivity

import numpy as np
import matplotlib.pyplot as plt
b=[0.1,0.2,0.5,1]
V=np.linspace(0,2*np.pi,100)

for _ in range(len(a)):
    phi=(b[_]*V**2)%(2*np.pi)
    plt.plot(V,phi)

plt.show()
"""

#Investigating sensitivity of more components using fig 4.10 toy example

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#Data generation
phi1=np.linspace(0,2*np.pi,10)
phi2=np.linspace(0,2*np.pi,10)
phi3=np.linspace(0,2*np.pi,10)
phi4=np.linspace(0,2*np.pi,10)
phi5=np.linspace(0,2*np.pi,10)

eta1=0.5
eta2=0.5
eta3=0.5
eta4=0.5
eta5=0.5
eta6=0.5

#Define formula for Phase shifter unitary, will define just dim=2 for now
def construct_PS(phi):
    mat=np.array(([np.exp(1j*phi/2),0],[0,np.exp(-1j*phi/2)]))
    return mat

def construct_BS(eta):
    mat=np.array(([np.sqrt(eta), 1j*np.sqrt(1-eta)],[1j*np.sqrt(1-eta),np.sqrt(eta)]))
    return mat

results=np.empty((len(phi1),len(phi2),len(phi3),len(phi4),len(phi5)))

a=np.array([1,0])
a.shape=(2,1)

b=np.array([1,0])
b.shape=(1,2)


for i in range(len(phi1)):
    for j in range(len(phi2)):
        for k in range(len(phi3)):
            for l in range(len(phi4)):
                for m in range(len(phi5)):
                    unitary_toy=construct_BS(eta6)@construct_PS(phi5[m])@construct_BS(eta5)@construct_PS(phi4[l])@construct_BS(eta4)@construct_PS(phi3[k])@construct_BS(eta3)@construct_PS(phi2[j])@construct_BS(eta2)@construct_PS(phi1[i])@construct_BS(eta1)
                    results[i][j][k][l][m]=abs(b@unitary_toy@a)**2
        

test=results[:][:][0][-1][5]
fig,ax=plt.subplots()
im = ax.imshow(test, cmap='turbo', interpolation='nearest', extent=[0,2*np.pi,0,2*np.pi])
fig.colorbar(im, ax=ax)
plt.xlabel(r'$\phi_1$')
plt.ylabel(r'$\phi_2$')
plt.show()
