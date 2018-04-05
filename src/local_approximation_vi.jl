# The solver type
mutable struct LocalApproximationValueIterationSolver{RNG<:AbstractRNG} <: Solver
    interp::LocalFunctionApproximator # Will be copied over by value to each policy
    max_iterations::Int64 # max number of iterations
    belres::Float64 # the Bellman Residual
    verbose::Bool # Whether to print while solving or not
    rng::RNG # Seed if req'd
    is_mdp_generative::Bool # Whether to treat underlying MDP model as generative
    n_generative_samples::Int64 # If underlying model generative, how many samples to use
    terminal_costs_set::Bool
end

# Default constructor
function LocalApproximationValueIterationSolver{RNG<:AbstractRNG}(interp::LocalFunctionApproximator;
                                                                  max_iterations::Int64=100, belres::Float64=1e-3,
                                                                  verbose::Bool=false, rng::RNG=Base.GLOBAL_RNG,
                                                                  is_mdp_generative::Bool=false, n_generative_samples::Int64=0,
                                                                  terminal_costs_set::Bool=false)
    # TODO : Will this copy the interp object by reference?
    return LocalApproximationValueIterationSolver(interp,max_iterations, belres, verbose, rng, is_mdp_generative, n_generative_samples, terminal_costs_set)
end

# The policy type
# NOTE : For now, we work directly with value function
# And extract actions at the end from the interp object
mutable struct LocalApproximationValueIterationPolicy <: Policy
    interp::LocalFunctionApproximator # General approximator to be used in VI 
    action_map::Vector # Maps the action index to the concrete action type
    mdp::Union{MDP,POMDP} # uses the model for indexing in the action function
    is_mdp_generative::Bool
    n_generative_samples::Bool
end

# Constructor with interpolator initialized
function LocalApproximationValueIterationPolicy(mdp::Union{MDP,POMDP},
                                                solver::LocalApproximationValueIterationSolver)
    return LocalApproximationValueIterationPolicy(deepcopy(solver.interp),ordered_actions(mdp),mdp,solver.is_mdp_generative,solver.n_generative_samples)
end


@POMDP_require solve(solver::LocalApproximationValueIterationSolver, mdp::Union{MDP,POMDP}) begin
    
    P = typeof(mdp)
    S = state_type(P)
    A = action_type(P)
    @req discount(::P)
    @req n_actions(::P)
    @subreq ordered_actions(mdp)
    
    # TODO : Can we specify EITHER requiring the below OR requiring generate_sr
    # @req transition(::P,::S,::A)
    # dist = transition(mdp, s, a)
    # D = typeof(dist)
    # @req iterator(::D)

    @req reward(::P,::S,::A,::S)
    @req action_index(::P, ::A)
    @req actions(::P, ::S)
    as = actions(mdp)
    @req iterator(::typeof(as))
    a = first(iterator(as))
    
end


function solve(solver::LocalApproximationValueIterationSolver, mdp::Union{MDP,POMDP})

    @warn_requirements solve(solver,mdp)

    # Ensure that generative model has a non-zero number of samples
    if solver.is_mdp_generative
        @assert solver.n_generative_samples > 0
    end

    # solver parameters
    max_iterations = solver.max_iterations
    belres = solver.belres
    discount_factor = discount(mdp)

    # Initialize the policy
    policy = LocalApproximationValueIterationPolicy(mdp,solver)

    total_time::Float64 = 0.0
    iter_time::Float64 = 0.0

    # Get attributes of interpolator
    num_interps::Int = n_interpolants(policy.interp)
    interp_points::Vector = get_all_interpolating_points(policy.interp)
    interp_values::Vector = get_all_interpolating_values(policy.interp)

    # Obtain the vector of states
    S = state_type(typeof(mdp))
    interp_states = Vector{S}(num_interps)
    for (i,pt) in enumerate(interp_points)
        interp_states[i] = POMDPs.convert_s(S, pt, mdp)
    end

    
    # Main loop
    for i = 1 : max_iterations
        residual::Float64 = 0.0
        tic()

        for (istate,s) in enumerate(interp_states)

            # NOTE : Assume that interpolator's state values can be directly
            # used with T and R functions - the converters from state to vector 
            # and vice versa are called inside the interpolator
            sub_aspace = actions(mdp,s)

            if isterminal(mdp, s)
                if !solver.terminal_costs_set
                    interp_values[istate] = 0.0
                end
            else
                old_util = interp_values[istate]
                max_util = -Inf

                for a in iterator(sub_aspace)
                    iaction = action_index(mdp,a)
                    u::Float64 = 0.0

                    # Do bellman backup based on generative / explicit
                    if solver.is_mdp_generative
                        # Generative Model
                        for j in 1:solver.n_generative_samples
                            sp, r = generate_sr(mdp, s, a, sol.rng)
                            sp_point = POMDPs.convert_s(Vector{Float64}, sp, mdp)
                            u += r + discount_factor*evaluate(policy.interp, sp_point)
                        end
                        u = u / solver.n_generative_samples
                    else
                        # Explicit Model
                        dist = transition(mdp,s,a)
                        for (sp, p) in weighted_iterator(dist)
                            p == 0.0 ? continue : nothing
                            r = reward(mdp, s, a, sp)
                            sp_point = POMDPs.convert_s(Vector{Float64}, sp, mdp)
                            u += p * (r + discount_factor*evaluate(policy.interp, sp_point))
                        end # next-states
                    end
                    
                    max_util = (u > max_util) ? u : max_util
                end #action

                # Update this interpolant value
                interp_values[istate] = max_util
                util_diff = abs(max_util - old_util)
                util_diff > residual ? (residual = util_diff) : nothing
            end
        end #state

        # TODO : interp_values directly edits the values in place
        # Is that acceptable?
        iter_time = toq()
        total_time += iter_time
        solver.verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual, iter_time*1000.0, total_time) : nothing
        residual < belres ? break : nothing

    end #main
    return policy
end


function value(policy::LocalApproximationValueIterationPolicy, s::S) where S

    # Again, assume that state-to-vector converter called by interpolator
    s_point = POMDPs.convert_s(Vector{Float64}, s, policy.mdp)
    val = evaluate(policy.interp, s_point)
    return val
end

# Not explicitly stored in policy - extract from value function interpolation
function action(policy::LocalApproximationValueIterationPolicy, s::S) where S
    
    mdp = policy.mdp
    best_a_idx = -1
    max_util = -Inf
    sub_aspace = actions(mdp,s)
    discount_factor = discount(mdp)


    for a in iterator(sub_aspace)
        iaction = action_index(mdp, a)
        u::Float64 = 0.0

        # Similar to what is done above
        if policy.is_mdp_generative
            for j in 1:policy.n_generative_samples
                sp, r = generate_sr(mdp, s, a, sol.rng)
                sp_point = POMDPs.convert_s(Vector{Float64}, sp, mdp)
                u += r + discount_factor*evaluate(policy.interp, sp_point)
            end
            u = u / policy.n_generative_samples
        else
            # Do for explicit
            dist = transition(mdp,s,a)
            for (sp, p) in weighted_iterator(dist)
                p == 0.0 ? continue : nothing
                r = reward(mdp, s, a, sp)
                sp_point = POMDPs.convert_s(Vector{Float64}, sp, mdp)
                u += p * (r + discount_factor*evaluate(policy.interp, sp_point))
            end # next-states
        end

        if u > max_util
            max_util = u
            best_a_idx = iaction
        end
    end

    return policy.action_map[best_a_idx]
end