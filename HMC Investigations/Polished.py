"""
Code file to clear up messy previous code writing to make it effective so that I can then pivot to actually just trying more advanced modelling methods (regularising priors, additional info sources..etc)

Things to tidy up:
PyMC model context box should calculate like matrix vector calculations rather than plugging in the expression I have drawn from outside the code
Eliminate trying to model ('external') phase shifters at end of circuit since they aren't captured by the model
Function to calculate unitary done in the code file (ties into solving point 1)
Add further results stuff like closeness between inferred U and true U
"""