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

# Constructor where grid is passed
# function LocalGIValueFunctionApproximator{G<:AbstractGrid}(grid::G)
#   self.grid = grid
#   self.gvalues = zeros(length(grid))

#   # TODO : Convert each vertex to state and put in gstates. Can I do this?
#   state_vectors = vertices(grid)
#   for i = 1 : length(grid)
#     push!(self.gstates,convertVectorToState(state_vectors[i]))
#   end

#   return self
# end

# TODO : Should we define an 'initialize' method that just uses the grid and initializes other stuff to zero?


################ INTERFACE FUNCTIONS ################

# Return the number of vertices in the grid
function n_interpolants(gifa::LocalGIValueFunctionApproximator)
  return length(gifa.grid)
end

# Return the vector of states 
function interpolating_states(gifa::LocalGIValueFunctionApproximator)
  return gifa.gstates
end

function get_interpolants(gifa::LocalGIValueFunctionApproximator)
  return gifa.gvalues
end

function evaluate{S}(gifa::LocalGIValueFunctionApproximator, s::S, mdp::Union{MDP,POMDP})
  state_vector = convert_s(AbstractVector{Float64},s,mdp)
  value = interpolate(gifa.grid,gifa.gvalues,state_vector)
  return value
end

function batchUpdate(gifa::LocalGIValueFunctionApproximator, gvalues::Vector{Float64})
  gifa.gvalues = deepcopy(gvalues)
end