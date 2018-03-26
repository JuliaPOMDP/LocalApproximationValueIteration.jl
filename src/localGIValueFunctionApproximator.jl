using GridInterpolations
include("localValueFunctionApproximator.jl")

# TODO : Should I template by state-type here? (And below - wherever I do it)
# I want to 'require' that the S type has the conversion methods defined
# Where do I put those requirements
mutable struct LocalGIValueFunctionApproximator{S,G<:AbstractGrid} <: LocalValueFunctionApproximator
  grid::G
  gvalues::Vector{Float64}
  gstates::Vector{S}
end

# TODO : So no outer constructor for this right?
# TODO : Should we define an 'initialize' method that just uses the grid and initializes other stuff to zero?


################ INTERFACE FUNCTIONS ################

# Return the number of vertices in the grid
function n_interpolants(gifa::LocalGIValueFunctionApproximator)
  return length(gifa.grid)
end

# Return the vector of states 
function get_all_interpolating_states(gifa::LocalGIValueFunctionApproximator)
  return gifa.gstates
end

function get_all_interpolants(gifa::LocalGIValueFunctionApproximator)
  return gifa.gvalues
end

function get_interpolating_nbrs_idxs_wts{S}(gifa::LocalGIValueFunctionApproximator, s::S, mdp::Union{MDP,POMDP})
  state_vector = convert_s(AbstractVector{Float64},s,mdp)
  return interpolants(gifa.grid, state_vector)
end


function evaluate{S}(gifa::LocalGIValueFunctionApproximator, s::S, mdp::Union{MDP,POMDP})
  state_vector = convert_s(AbstractVector{Float64},s,mdp)
  value = interpolate(gifa.grid,gifa.gvalues,state_vector)
  return value
end

function batchUpdate(gifa::LocalGIValueFunctionApproximator, gvalues::Vector{Float64})
  gifa.gvalues = deepcopy(gvalues)
end