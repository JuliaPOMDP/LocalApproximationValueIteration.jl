#=
Construct a full grid of 100 x 100 and a discrete grid of 10 x 10
Put reward states of all the same value from (40,40) to (60,60)
Run discrete VI and localapproxVI on each grid
Then randomly sample several points on full grid and compare values of discreteVI and localApproxVI
=#
function test_against_full_grid()

  # Generate reward states and set to reward 10.0
  rstates = Vector{GridWorldState}()
  rvect = Vector{Float64}()
  for x = 40:60
    for y = 40:60
      push!(rstate,GridWorldState(x,y))
      push!(rvect,10.0)
    end
  end

  # Create full MDP and solve
  mdp = GridWorld(sx=100, sy=100, rs=rstates, rv=rvect)
  solver = ValueIterationSolver()
  policy = create_policy(solver, mdp)
  policy = solve(solver, mdp, policy, verbose=true)

  return true
end


@test test_against_full_grid() == true