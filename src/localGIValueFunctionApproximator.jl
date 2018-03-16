using GridInterpolations

# TODO : Should I template by state-type here? (And below - wherever I do it)
# I want to 'require' that the S type has the conversion methods defined
# Where do I put those requirements
mutable struct localGIValueFunctionApproximator{S,G<:AbstractGrid} <: localGIValueFunctionApproximator
  grid::G
  gvalues::Vector{Float64}
  gstates::Vector{S}
end

# Constructor where grid is passed
function localGIValueFunctionApproximator{S,G<:AbstractGrid}(grid::G)
  self.grid = grid
  self.gvalues = zeros(length(grid))

  # TODO : Convert each vertex to state and put in gstates. Can I do this?
  self.gstates = Vector{S}(length(grid))
  state_vectors = vertices(grid)
  for i = 1 : length(grid)
    self.gstates[i] = convertVectorToState{S}(state_vectors[i])
  end

  return self
end


################ INTERFACE FUNCTIONS ################

# Return the number of vertices in the grid
function n_interpolants(gifa::localGIValueFunctionApproximator)
  return length(gifa.grid)
end

# Return the vector of states 
function interpolating_states(gifa::localGIValueFunctionApproximator)
  return gifa.gstates
end

function interpolants(gifa::localGIValueFunctionApproximator)
  return gifa.gvalues
end

function evaluate{S}(gifa::localGIValueFunctionApproximator, s::S)
  state_vector = convertStateToVector{S}(s)
  value = interpolate(gifa.grid,gifa.gvalues,state_vector)
  return value
end

function batchUpdate(gifa::localGIValueFunctionApproximator, gvalues::Vector{Float64})
  gifa.gvalues = deepcopy(gvalues)
end