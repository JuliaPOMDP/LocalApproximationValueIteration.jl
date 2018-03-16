#=
Construct a full grid of 100 x 100 and a discrete grid of 10 x 10
Put reward states of all the same value from (40,40) to (60,60)
Run discrete VI and localapproxVI on each grid
Then randomly sample several points on full grid and compare values of discreteVI and localApproxVI
=#

# State conversion functions
function convertStateToVector(s::GridWorldState)
  vect = [convert(Float64,s.x),convert(Float64,s.y)]
  return vect
end

function convertVectorToState(vect::Vector{Float64})
  s = GridWorldState(convert(Int64,vect[1]), convert(Int64,vect[2]))
  return s
end

# TODO : We should define this for standard types somewhere
function is_generative(::GridWorld)
  return false
end


function test_against_full_grid()

  # Generate reward states and set to reward 10.0
  rstates = Vector{GridWorldState}()
  rvect = Vector{Float64}()
  for x = 40:60
    for y = 40:60
      push!(rstates,GridWorldState(x,y))
      push!(rvect,10.0)
    end
  end

  # Create full MDP - to be used by both!
  mdp = GridWorld(sx=100, sy=100, rs=rstates, rv=rvect)

  # Solve with discrete VI
  # solver = ValueIterationSolver()
  # policy = create_policy(solver, mdp)
  # policy = solve(solver, mdp, policy, verbose=true)


  # Setup grid with 0.1 resolution
  grid = RectangleGrid(linspace(1,100,10), linspace(1,100,10))
  interp = LocalGIValueFunctionApproximator{RectangleGrid}(grid)
  approx_solver = LocalApproximationValueIterationSolver(interp)
  approx_policy = solve(approx_solver, mdp, verbose=True)
  return true
end


@test test_against_full_grid() == true