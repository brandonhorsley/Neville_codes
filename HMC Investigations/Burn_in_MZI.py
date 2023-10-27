"""
This code will be for implementing a custom sampling method akin to what 
Alex Neville does in his thesis in order to converge to a good 'a' and then 
a good 'a' and 'b'. Unfortunately raw HMC does not seem to meet this end and 
so we get solid eta estimates while 'a' and 'b' estimates are rubbish. Alex 
implements a hierarchical approach of using sampling to determine alpha model 
(only changing 'a') then determine beta model (starting from 'a' values from 
end of alpha model, then change 'a' and 'b' together) before finally changing 
all eta's,a's and b's together.
"""