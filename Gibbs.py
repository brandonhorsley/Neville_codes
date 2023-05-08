"""
Algorithm 3 from Alex Neville thesis. Gibbs sampling algorithm. Bottom of page 40.

1. Pick an initial state vec(x)^(0)
2. For i = 0 to i = N-1
   2.1. Sample x'_1 ~ p(x'_1 | x_2,...,x_n)
   2.2. Sample x'_2 ~ p(x'_2 | x_1,x_3,...,x_n)
   2.j. Sample x'_j ~ p(x'_1 | x_1,...,x_j-1,x_j+1,x_n)
   2.n. Sample x'_n ~ p(x'_n | x_1,...,x_n-1)
   2.n+1. Set vec(x)^(i+1) = (x_1,...x_n)
"""
