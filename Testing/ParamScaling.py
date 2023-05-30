"""
This code file will be dedicating to investigating how MCMC generation scales with number of parameters 
to be estimated. 

The first roadblock that becomes apparent is that my current codes are very specific in both Alg4 and 
the likelihood function from Aux_Nev is also dependent on the specifics of modelling a circuit, so first 
i will review Alex's thesis to see if he reviews runtime considerations, otherwise i will focus on generalising 
the code/model building component.

Alex refers to reaching infeasibility once a certain size of circuit has been reached but he never states any specifics.
This 'infeasibility is related both to possible runtime of a given method but also that given method's probability
of success which also decreases as time progresses and one approach he suggests is using restarts or parallelising 
multiple burn-ins and picking the burn in with the best mean likelihood. So it seems that he was very aware that
his protocol worked well but by no means was guaranteed to be optimal.

So then in building some intuition first around runtime scaling with model scaling i can likely estimate the 
runtime scaling by profiling the model building component, the likelihood component, and the Alg4 component and 
then put these parts together. I could just hardcode a bunch of different cases but i think it would be optimal
for me to generalise my code first.
"""