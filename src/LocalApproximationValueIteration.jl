"""
This module implements Value Iteration for large/continuous state spaces by solving for some 
subspace of the state space and interpolating the value function over the rest of the state space
"""
module LocalApproximationValueIteration

using Random
using Printf
using POMDPs
using POMDPTools
using LocalFunctionApproximation
using POMDPLinter: @POMDP_require, @req, @subreq, @warn_requirements

import POMDPs: Solver, solve, Policy, action, value


# Exports related to solver
export
    LocalApproximationValueIterationPolicy,
    LocalApproximationValueIterationSolver

include("local_approximation_vi.jl")
    
end # module
