"""
Algorithm 5 in Alex Neville thesis. Metropolised Independence Sampling (MIS) algorithm. Top of page 98.

1. Label the current state vec(p_α).
2. Propose a new state vec(p'_α) by picking each component from p'_α,i ~ unif(-π,π).
3. Accept the proposed state and transition from vec(p_α) to vec(p'_α) with probability
   T(vec(p_α)|vec(p'_α)) = min(1,(P(D|vec(p_α),V)P(vec(p_α)))/(P(D|vec(p'_α),V)P(vec(p'_α))
4. Go to 1.
"""
