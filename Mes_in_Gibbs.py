"""
Algorithm 4 in Alex Neville thesis. Metropolis within Gibbs sampling algorithm. Middle of page 94.

1. Label the current state vec(p).
2. For each component p_i ∈ vec(p):
   2.1. Propose a new state vec(p)' with components p'_k = p_k for all k not equal to i and p'_α,i 
        is picked randomly from the proposal distribution g_i(vec(p')|vec(p)). 
   2.2. Accept the proposed state and transition from vec(p) to vec(p') with probability
        T(vec(p)|vec(p')) = min(1,(P(D|vec(p),V)P(vec(p))g_i(vec(p')|vec(p)))/(P(D|vec(p'),V)P(vec(p'))g_i(vec(p)|vec(p')))
3. Go to 1
"""
