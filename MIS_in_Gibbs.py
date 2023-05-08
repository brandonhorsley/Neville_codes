"""
Algorithm 6 in Alex Neville thesis. Metropolised Independence Sampling (MIS) within Gibbs algorithm. Middle of page 98.

1. Label the current state vec(p_α).
2. For each component p_i ∈ vec(p_α):
   2.1. Propose a new state vec(p)' with components p'_α,k = p_α,k for all k not equal to i and random
        p'_α,i ~ unif(-π,π).
   2.2. Accept the proposed state and transition from vec(p_α) to vec(p'_α) with probability
        T(vec(p_α)|vec(p'_α)) = min(1,(P(D|vec(p_α),V)P(vec(p_α)))/(P(D|vec(p'_α),V)P(vec(p'_α))
3. Go to 1
"""