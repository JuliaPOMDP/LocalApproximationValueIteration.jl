# Construct a full grid of 100 x 100 and a discrete grid of 10 x 10
# Put reward states of all the same value from (40,40) to (60,60)
# Run discrete VI and localapproxVI with two different resolutions on each grid
# Then randomly sample several points on full grid and compare the average
# difference from the full value function.
# The approximation with higher resolution should have lower avg. error
function test_against_full_grid()

    # Generate reward states and set to reward 10.0
    rewards = Dict{GWPos, Float64}()
    for x = 40:60
        for y = 40:60
            rewards[GWPos(x, y)] = 10
        end
    end

    # Create full MDP - to be used by both!
    mdp = SimpleGridWorld(size=(100, 100), rewards=rewards)

    # Solve with discrete VI
    solver = ValueIterationSolver(max_iterations=1000, verbose=true)
    policy = solve(solver, mdp)

    # Set up two different grids with different step sizes and ensure that the
    # grid with higher resolution has lower error
    STEP_SIZE_LOW = 5
    STEP_SIZE_HI = 10
    
    grid_low = RectangleGrid(range(1,step=STEP_SIZE_LOW,stop=100), range(1,step=STEP_SIZE_LOW,stop=100))
    grid_hi = RectangleGrid(range(1,step=STEP_SIZE_HI,stop=100), range(1,step=STEP_SIZE_HI,stop=100))

    interp_low = LocalGIFunctionApproximator(grid_low)
    interp_hi = LocalGIFunctionApproximator(grid_hi)

    solver_low = LocalApproximationValueIterationSolver(interp_low, verbose=true, max_iterations = 1000)
    solver_hi = LocalApproximationValueIterationSolver(interp_hi, verbose=true, max_iterations = 1000)

    policy_low = solve(solver_low, mdp)
    policy_hi = solve(solver_hi, mdp)

    total_err_low = 0.0
    total_err_hi = 0.0

    for state in states(mdp)

        full_val = value(policy, state)
        
        approx_val_low = value(policy_low, state)
        approx_val_hi = value(policy_hi, state)   

        total_err_low += abs(full_val-approx_val_low)
        total_err_hi += abs(full_val-approx_val_hi)
    end
    
    avg_err_low = total_err_low / length(states(mdp))
    avg_err_hi = total_err_hi / length(states(mdp))

    return (avg_err_low < avg_err_hi)
end


@test test_against_full_grid() == true
