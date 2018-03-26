include("localValueFunctionApproximator.jl")

# This requires the MDP to approximate via a sub-MDP and an approximation scheme (simplex/multilinear/kNN)
# The states of the sub-MDP are the interpolating states according to the approx. scheme
# The transitions to other sub-MDP states are based on the interpolation weights for neighbour
# states of a given state in the true MDP. The one-step reward for a (sub-MDP state, action) pair is the
# weighted average of rewards for that action on that sub-MDP state in the true MDP
mutable struct LocalSubMDPGenerator{RNG<:AbstractRNG}
  interp::LocalValueFunctionApproximator
  rng::RNG
  is_mdp_generative::Bool
  n_generative_samples::Int64
end

function LocalSubMDPGenerator{RNG<:AbstractRNG}(interp::LocalValueFunctionApproximator; rng::RNG=Base.GLOBAL_RNG,
                              is_mdp_generative::Bool=false, n_generative_samples::Int64=0)
  return LocalSubMDPGenerator(interp,rng,is_mdp_generative,n_generative_samples)
end


function generate(generator::LocalSubMDPGenerator, mdp::Union{MDP,POMDP})

  if generator.is_mdp_generative
    @assert generator.n_generative_samples > 0
  end

  # Get data needed for states, transition and reward
  all_interp_states::Vector = get_all_interpolating_states(generator.interp)
  num_states::Int64 = length(all_interp_states)
  num_actions::Int64 = n_actions(mdp)

  reward_matrix = zeros(num_states, num_actions)
  transition_matrix = Array{SparseVector{Float64},2}(num_states, num_actions)

  for (istate,s) in enumerate(all_interp_states)

    sub_aspace = actions(mdp,s)

    for a in iterator(sub_aspace)
      iaction = action_index(mdp,a)


      if isterminal(mdp,s)
        # One step reward is 0 and so is sp transition
        reward_matrix[istate,iaction] = 0.0
        transition_matrix[istate,iaction] = sparsevec(zeros(num_states))
      else
        avg_reward::Float64 = 0.0
        next_state_dist = zeros(num_states)

        if generator.is_mdp_generative
          
          # TODO : Is this the right way to do this?
          for j in 1:generator.n_generative_samples
            sp, r = generate_sr(mdp, s, a, sol.rng)
            avg_reward += r
            idxs,wts = get_interpolating_nbrs_idxs_wts(gifa, sp, mdp)
            for (i,w) in zip(idxs,wts)
              next_state_dist[i] += w
            end
          end

          avg_reward /= generator.n_generative_samples
          next_state_dist /= generator.n_generative_samples
          next_state_dist /= sum(next_state_dist)

        else
          dist = transition(mdp,s,a)
          for (sp, p) in weighted_iterator(dist)
            p == 0.0 ? continue : nothing
            r = reward(mdp, s, a, sp)
            
            # Weighted average of one-step rewards, wt sums to 1
            avg_reward += p*r

            idxs,wts = get_interpolating_nbrs_idxs_wts(gifa, sp, mdp)
            for (i,w) in zip(idxs,wts)
              next_state_dist[i] += w
            end
          end

          next_state_dist /= sum(next_state_dist)
        end

        reward_matrix[istate, iaction] = avg_reward
        transition_matrix[istate, iaction] = sparsevec(next_state_dist)
      end # if !isterminal(mdp,)
    end # For a in iterator(sub_space)
  end # For (istate,s) in enumerate


  # TODO: Now need to save the states, the reward and transition matrices in the appropriate format

end
