#=
Construct a full grid of 100 x 100 and a discrete grid of 10 x 10
Put reward states of all the same value from (40,40) to (60,60)
Run discrete VI and localapproxVI on each grid
Then randomly sample several points on full grid and compare values of discreteVI and localApproxVI
=#

# State conversion functions
function POMDPs.convert_s(::Type{AbstractVector{Float64}}, s::GridWorldState, mdp::GridWorld)
  v = SVector{2,Float64}(s.x, s.y)
  return v
end

function POMDPs.convert_s(::Type{GridWorldState}, v::AbstractVector{Float64}, mdp::GridWorld)
  s = GridWorldState(convert(Int64,v[1]), convert(Int64, v[2]))
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

  # Create the interpolation object
  gvalues = zeros(length(grid))
  state_vectors = vertices(grid)
  gstates = Vector{GridWorldState}(length(grid))
  for (i,sv) in enumerate(state_vectors)
    gstates[i] = convert_s(GridWorldState, sv, mdp)
  end
  interp = LocalGIFunctionApproximator(grid,gvalues,gstates)

  # Try out some interp stuff
  println(n_interpolants(interp))

  dummy_state = convert_s(GridWorldState, SVector{2,Float64}(40,50), mdp)
  println(evaluate(interp,dummy_state,mdp))


  approx_solver = LocalApproximationValueIterationSolver(interp, verbose=true, max_iterations = 1000)
  approx_policy = solve(approx_solver, mdp)
  return true
end


@test test_against_full_grid() == true