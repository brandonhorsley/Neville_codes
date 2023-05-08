"""
Algorithm 1 in Alex Neville thesis. Rejection sampling. Page 35 in thesis.

For i = 1 to i = N
1. Sample x_(i) ~ q(x) and u ~ unif(0, 1).
2. If u < p(x^(i))/Mq(x^(i)):
      Accept x^(i) and set i to i + 1.
   Else:
      Reject x^(i)
"""

