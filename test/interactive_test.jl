using POMDPModels
using POMDPs
using DiscreteValueIteration
using GridInterpolations
using LocalApproximationValueIteration
using StaticArrays
using Distances
using NearestNeighbors

function POMDPs.convert_s(::Type{AbstractVector{Float64}}, s::GridWorldState, mdp::GridWorld)
  v = SVector{2,Float64}(s.x, s.y)
  return v
end

function POMDPs.convert_s(::Type{GridWorldState}, v::AbstractVector{Float64}, mdp::GridWorld)
  s = GridWorldState(convert(Int64,v[1]), convert(Int64, v[2]))
end



rstates = Vector{GridWorldState}()
rvect = Vector{Float64}()
for x = 40:50
  for y = 50:70
    push!(rstates,GridWorldState(x,y))
    push!(rvect,10.0)
  end
end

# Create full MDP - to be used by both!
mdp = GridWorld(sx=100, sy=100, rs=rstates, rv=rvect)

# Solve with discrete VI
solver = ValueIterationSolver()
policy = create_policy(solver, mdp)
policy = solve(solver, mdp, policy, verbose=false)


#Setup grid with 0.1 resolution
grid = RectangleGrid(linspace(1,100,10), linspace(1,100,10))

# Create the interpolation object
gvalues = zeros(length(grid))
state_vectors = vertices(grid)
gstates = Vector{GridWorldState}(length(grid))
for (i,sv) in enumerate(state_vectors)
  gstates[i] = convert_s(GridWorldState, sv, mdp)
end
interp = LocalGIValueFunctionApproximator(grid,gvalues,gstates)

# Try out some interp stuff
println(n_interpolants(interp))

dummy_state = convert_s(GridWorldState, SVector{2,Float64}(40,50), mdp)
println(LocalApproximationValueIteration.evaluate(interp,dummy_state,mdp))


approx_gi_solver = LocalApproximationValueIterationSolver(interp, verbose=false, max_iterations = 1000)
approx_gi_policy = solve(approx_gi_solver, mdp)

# Now construct nearest neighbor tree
nn_samples = 100
nnvalues = zeros(100)
nnstates = Vector{GridWorldState}(100)
nndata = Vector{SVector{2,Float64}}(100)
idx = 1
for i in linspace(1,100,10)
  for j in linspace(1,100,10)
    #datapt = SVector{2,Float64}(rand(1:100), rand(1:100))
    datapt = SVector{2,Float64}(i,j)
    nndata[idx] = datapt
    nnstates[idx] = convert_s(GridWorldState, datapt, mdp)
    idx += 1
  end
end
nntree = KDTree(nndata)
nnfa = LocalNNValueFunctionApproximator(nntree, nnvalues, nnstates, 4, 0.0)


approx_nn_solver = LocalApproximationValueIterationSolver(nnfa, verbose=false, max_iterations = 1000)
approx_nn_policy = solve(approx_nn_solver, mdp)




