"""
This code file is being opened to brainstorm different subalgorithms that could be created.
Alg. 4 (MH within Gibbs) is for findingparameter estimates (so could maybe just be interpreted as an exploration)
Alg. 5 (MIS) to get an estimate that is in the rough vicinity of optimal value
Alg 6. (MIS within Gibbs) has one value change at a time so is better for fine tuning.
Alg. 7 (pi kick search) was devised based on the observation that local maxima in the posterior occur at
intervals which look like random strings with elements from {0, pi}, has a much higher probability of 
transitioning to the global maximum than either of algorithms 5 and 6.

########################################################################################
Ideas for alternative algorithms:
https://www.pymc.io/projects/docs/en/stable/api/samplers.html
https://www.pymc.io/projects/docs/en/stable/api/smc.html

NUTS (need to do with distributions)
Differential Evolution Metropolis sampling step.
Adaptive Differential Evolution Metropolis (DREAM) sampling step that uses the past to inform jumps.
SMC, SMC kernel?
What about general optimisation algorithms (Nelder-Mead,krylov,...etc), if so how would i define a good acquisition function,...etc

################################################################################################
Alg 7 (pi kick search) seems to chronically plateau on the log likelihood plot seen in fig 4.8 so that could be
one to try and really deal with.
"""

