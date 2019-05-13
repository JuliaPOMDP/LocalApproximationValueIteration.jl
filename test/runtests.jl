using POMDPModels
using POMDPs
using POMDPModelTools
using StaticArrays
using Random
using DiscreteValueIteration
using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using Test

@testset "all" begin

    println("Testing Requirements")
    gifa = LocalGIFunctionApproximator(RectangleGrid([0.0, 1.0], [0.0, 1.0]))
    @requirements_info LocalApproximationValueIterationSolver(gifa) SimpleGridWorld()

    @testset "integration" begin
        include("runtests_versus_discrete_vi.jl")
    end
end