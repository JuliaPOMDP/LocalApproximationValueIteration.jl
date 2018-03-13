"""
This module implements Value Iteration for large/continuous state spaces by solving for some 
subspace of the state space and interpolating the value function over the rest of the state space
"""
module LocalApproximationValueIteration

using POMDPs
using POMDPToolbox

import POMDPs: Solver, solve, Policy, action, value 

export
    LocalApproxValueIterationPolicy,
    LocalApproxValueIterationSolver,
    create_policy,
    solve,
    action,
    value,
    locals

end # module
