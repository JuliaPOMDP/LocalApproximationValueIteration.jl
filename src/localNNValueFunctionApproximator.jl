using NearestNeighbors
using Distances
include("localValueFunctionApproximator.jl")

mutable struct LocalNNValueFunctionApproximator{S, V <: AbstractVector, M <: Metric} <: LocalValueFunctionApproximator
  nntree::NNTree{V,M}
  nnvalues::Vector{Float64}
  nnstates::Vector{S}
  knnK::Int64
  rnnR::Float64
end

################ INTERFACE FUNCTIONS ################
function n_interpolants(nnfa::LocalNNValueFunctionApproximator)
  return length(nnfa.nntree.indices)
end

function interpolating_states(nnfa::LocalNNValueFunctionApproximator)
  return nnfa.nnstates
end

function get_interpolants(nnfa::LocalNNValueFunctionApproximator)
  return nnfa.nnvalues
end

function evaluate{S}(nnfa::LocalNNValueFunctionApproximator, s::S, mdp::Union{MDP,POMDP})
  state_vector = convert_s(AbstractVector{Float64}, s, mdp)

  # Depending on parameter, run knn or inrange
  @assert (nnfa.knnK > 0 || nnfa.rnnR > 0.0)
  if nnfa.knnK > 0
    # Do k-NN lookup to get data and distances
    idxs, dists = knn(nnfa.nntree, state_vector, nnfa.knnK)
  else
    # Do inrange lookup to get data
    # Then use metric to get dists between query and each nearest neighbor
    idxs = inrange(nnfa.nntree, state_vector, nnfa.rnnR)
    dists = zeros(length(idxs))
    for (i,idx) in enumerate(idxs)
      dists[i] = Distances.evaluate(nnfa.nntree.metric, state_vector, nnfa.nntree.data[idx])
    end
  end

  # If EXACTLY equal to a value then return that value
  # TODO : Is this the right way to handle this?
  if minimum(dists) < eps(Float64)
    same_idx = idxs[indmin(dists)]
    return nnfa.nnvalues[same_idx]
  end

  # Now do a weighted average of values with weights inverse to distances
  value::Float64 = 0.0
  wtsum::Float64 = 0.0
  for i = 1 : length(idxs)
    wt = 1.0/dists[i]
    value += wt*nnfa.nnvalues[idxs[i]]
    wtsum += wt
  end
  value /= wtsum


  return value
end

function batchUpdate(nnfa::LocalGIValueFunctionApproximator, nnvalues::AbstractVector{Float64})
  nnfa.gvalues = deepcopy(gvalues)
end