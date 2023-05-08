"""
Algorithm 7 from Alex Neville thesis. Stochastic π kick search algorithm. Bottom of page 98.

1. Label the current state vec(p_α)
2. Generate vec(q_α) = (q_1,...,q_k) where each q_i ~ unif{−π, 0, π}.
3. If P (D|vec(p_α) + vec(q_α),V)P(vec(p)+vec(q_α)) > P(D|vec(p_α),V)P(vec(p_α)) then 
   transition from vec(p_α) to vec(p_α) + vec(q_α).
4. Go to 1.
"""