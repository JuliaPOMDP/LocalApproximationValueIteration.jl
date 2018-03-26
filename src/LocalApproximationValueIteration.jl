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
    LocalNNValueFunctionApproximator,
    n_interpolants,
    get_all_interpolating_states,
    get_all_interpolants,
    get_interpolating_nbrs_idxs_wts,
    evaluate,
    batchUpdate

export
    LocalSubMDPGenerator,
    generate

include("localApproximationVI.jl")
include("localValueFunctionApproximator.jl")
include("localGIValueFunctionApproximator.jl")
include("localNNValueFunctionApproximator.jl")
include("localSubMDPGenerator.jl")
    
end # module