"""
This module implements Value Iteration for large/continuous state spaces by solving for some 
subspace of the state space and interpolating the value function over the rest of the state space
"""
__precompile__()
module LocalApproximationValueIteration

using POMDPs
using POMDPToolbox

import POMDPs: Solver, solve, Policy, action, value 

# Exports related to solver
export
    LocalApproximationValueIterationPolicy,
    LocalApproximationValueIterationSolver,
    solve,
    action,
    value

# Exports related to approximator
export
    LocalValueFunctionApproximator,
    LocalGIValueFunctionApproximator,
    n_interpolants,
    interpolating_states,
    get_interpolants,
    evaluate,
    batchUpdate


include("localApproximationVI.jl")
include("localValueFunctionApproximator.jl")
include("localGIValueFunctionApproximator.jl")

end # module