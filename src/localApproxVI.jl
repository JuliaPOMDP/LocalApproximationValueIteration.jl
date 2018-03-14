# The solver type
mutable struct LocalApproxValueIterationSolver <: Solver
    max_iterations::Int64 # max number of iterations
    belres::Float64 # the Bellman Residual
    interp::LocalValueFnApproximator # Will be copied over to each policy
    verbose::Bool # Whether to print while solving or not
end
# Default constructor
function LocalApproxValueIterationSolver(;max_iterations::Int64=100, belres::Float64=1e-3)
    return LocalApproxValueIterationSolver(max_iterations, belres)
end

# The policy type
# TODO : For now, we work directly with value function
# TODO : And extract actions at the end from the interp object
mutable struct LocalApproxValueIterationPolicy <: Policy
    interp::LocalValueFnApproximator # General approximator to be used in VI 
    action_map::Vector # Maps the action index to the concrete action type
    mdp::Union{MDP,POMDP} # uses the model for indexing in the action function
end

# Constructor with interpolator initialized
function LocalApproxValueIterationPolicy(mdp::Union{MDP,POMDP},
                                         solver::LocalApproxValueIterationSolver)
    self.interp = deepcopy(solver.interp) # So that different policies (with different T,R) for same solver can be used
    self.action_map = ordered_actions(mdp)
    self.mdp = mdp
    return self
end


@POMDP_require solve(solver::LocalApproxValueIterationSolver, mdp::Union{MDP,POMDP}) begin
    
    P = typeof(mdp)
    S = state_type(P)
    A = action_type(P)
    @req discount(::P)
    @req n_actions(::P)
    @subreq ordered_actions(mdp)
    @req transition(::P,::S,::A)
    @req reward(::P,::S,::A,::S)
    @req action_index(::P, ::A)
    @req actions(::P, ::S)
    as = actions(mdp)
    @req iterator(::typeof(as))
    a = first(iterator(as))
    dist = transition(mdp, s, a)
    D = typeof(dist)
    @req iterator(::D)
    @req pdf(::D,::S)
end


function solve(solver::LocalApproxValueIterationSolver, mdp::Union{MDP,POMDP})

    @warn_requirements solve(solver,mdp)

    # solver parameters
    max_iterations = solver.max_iterations
    belres = solver.belres
    discount_factor = discount(mdp)

    # Initialize the policy
    policy = LocalApproxValueIterationPolicy(mdp,solver)

    total_time::Float64 = 0.0
    iter_time::Float64 = 0.0

    # Get attributes of interpolator
    num_interps::Int = n_interpolants(policy.interp)
    interp_states::Vector = interpolating_states(policy.interp)
    interp_values = interpolants(policy.interp)
    
    # Main loop
    for i = 1 : max_iterations
        residual::Float64 = 0.0
        tic()

        for (istate,s) in enumerate(interp_states)

            # TODO : Assume that interpolator's state values can be directly
            # used with T and R functions - the converters from state to vector 
            # and vice versa are called inside the interpolator
            sub_aspace = actions(mdp,s)

            if is_terminal(mdp, s)
                interp_values[istate] = 0.0
            else
                old_util = interp_values[istate]
                max_util = -Inf

                for a in iterator(sub_aspace)
                    iaction = action_index(mdp,a)
                    dist = transition(mdp,s,a)
                    u::Float64 = 0.0
                    for (sp, p) in weighted_iterator(dist)
                        p = 0.0 ? continue : nothing
                        r = reward(mdp, s, a, sp)
                        u += p * (r + discount_factor*evaluate(policy.interp, sp))
                    end # next-states
                    max_util = (u > max_util) ? u : max_util
                end #action

                # Update this interpolant value
                interp_values[istate] = max_util
                util_diff = abs(max_util - old_util)
                util_diff > residual ? (residual = util_diff) : nothing
            end
        end #state

        # Update all interpolant values
        batchUpdate(policy.interp, interp_values)

        iter_time = toq()
        total_time += iter_time
        solver.verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual, iter_time*1000.0, total_time) : nothing
        residual < belres ? break : nothing

    end #main
    return policy
end


function value(policy::LocalApproxValueIterationPolicy, s::S) where S

    # Again, assume that state-to-vector converter called by interpolator
    val = evaluate(policy.interp,s)
    return val
end

# Not explicitly stored in policy - extract from value function interpolation
function action(policy::LocalApproxValueIterationPolicy, s::S) where S
    
    mdp = policy.mdp
    best_a_idx = -1
    max_util = -Inf
    sub_aspace = actions(mdp,s)

    for a in iterator(sub_aspace)
        iaction = action_index(mdp)
        dist = transition(mdp, s, a) # creates distribution over neighbors
        u::Float64 = 0.0
        for (sp, p) in weighted_iterator(dist)
            p == 0.0 ? continue : nothing
            r = reward(mdp, s, a, sp)
            u += p * (r + discount_factor*evaluate(policy.interp, sp))
        end
        if u > max_util
            max_util = u
            best_a_idx = iaction
        end
    end

    return policy.action_map[best_a_idx]
end