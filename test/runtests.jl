using POMDPModels
using POMDPs
using POMDPTools
using StaticArrays
using Random
using DiscreteValueIteration
using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using POMDPLinter: show_requirements, get_requirements
using Test

@testset "all" begin

    println("Testing Requirements")
    gifa = LocalGIFunctionApproximator(RectangleGrid([0.0, 1.0], [0.0, 1.0]))
    @test_skip @requirements_info LocalApproximationValueIterationSolver(gifa) SimpleGridWorld()
    show_requirements(get_requirements(POMDPs.solve, (LocalApproximationValueIterationSolver(gifa), SimpleGridWorld())))

    @testset "integration" begin
        include("runtests_versus_discrete_vi.jl")
    end
end
