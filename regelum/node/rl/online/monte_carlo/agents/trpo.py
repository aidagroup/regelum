"""Trust Region Policy Optimization (TRPO) agent implementation.

This module implements the TRPO algorithm, a policy optimization method that
uses trust regions to ensure stable policy updates while maximizing expected
return. TRPO maintains monotonic improvement in policy performance by
constraining the size of policy updates using KL divergence.
"""
