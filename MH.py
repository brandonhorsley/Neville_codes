"""
Algorithm 2 in Alex Neville thesis. Metropolis-Hastings algorithm. Middle of page 38.

1. Pick an initial state x^(0)
2. For i = 0 to i = N-1
   2.1. Sample u ~ unif (0, 1)
   2.2. Sample x' ~ g(x'|x^(i))
   2.3. If u < A(x'|x^(i))
           x^(i+1) = x'
        else:
           x^(i+1) = x^(i)
"""