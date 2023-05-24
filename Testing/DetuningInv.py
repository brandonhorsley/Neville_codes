"""
Investigation to gauge the sensitivity of changing eta values and a values and b values. My initial 
investigation will be to just individually plot detuning from true against trace distance.

It occurs to me that it will actually be quite tough to incorporate things, i kinda got close with Neville_thesis_3.py
where i investigated how changing etas affects the phi1 v phi2 plot which seemed to generate a particular relationship.
But then finding phi is awkward because it is a function of a and b and V. a will be linear with fixing b and V since 
it is just an offset term. So all that is left is thinking about b and V which in our model which is essentially
y=ax**2. So from simple parabolas, a sharpens the parabola so you need a smaller x to get a given value.
y=ax**2 so dy/dx=2ax

I will focus on investigating the relationship between y, a and x first and then this should complete my 
intuition around a,b,V affecting phi values, and my knowledge of eta values and phi values from 
Neville_thesis_3.py

Plot completed, the key thing seems to be that increasing V means distance between peaks (where modulo kicks in) 
gets shorter and shorter, and lower b values means shallower rise so less distance sharp impact on phi. So then 
i think perhaps really sensitivity of the landscape is proportional to both b and V so such a procedure becomes 
less wise to do with larger b and V. a is just a matter of proportionality since it is an offset term. Each eta 
value in the toy example has a unique relationship to phi and the smearing that is seen interestingly seems to 
mean that more detuning of eta leads to less of an impact as phi values change...

So all this intuition is only really directed towards the toy example but it could be worth thinking then about 
impact on performance as needing stricter reparameterisation and an even more precise estimate for larger b (and V). 
a is pretty agnostic. I think eta smearing is pretty reasonable for the toy example, i am also curious though about 
how eta sensitivity depends on number of elements, i would wager more components means more sensitivity...
"""

import numpy as np
import matplotlib.pyplot as plt
a=[0.1,0.2,0.5,1]
x=np.linspace(0,2*np.pi,100)

for _ in range(len(a)):
    y=(a[_]*x**2)%(2*np.pi)
    plt.plot(x,y)

plt.show()