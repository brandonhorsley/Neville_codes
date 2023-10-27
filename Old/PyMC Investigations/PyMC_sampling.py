"""
Opening this file to practice with custom sampling procedures since Alex's procedure does sampling to figure out 
the alpha model (just a's), then the beta model (a's and b's) then the whole model (eta's,a's and b's). So in 
PyMC where custom sampling methods is done on the specific parameters then that will be the situation here and 
could hold the key to stopping the uniform distribution for a's and wider normal distributions. Could also be good 
to ask the pyMC community how they handle cases where there the state space is very sensitive to multiple strong 
solutions.

Revisiting PyMC1 though shows that defining a fixed eta and b values in the current code still leads to a basically 
uniform distribution so first off i would need to sort out my PyMC implementation.
"""
