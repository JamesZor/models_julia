using Test
using DataFrames
include(joinpath(@__DIR__, "..", "current_development", "scottish_lower", "_protocol", "ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol
struct FakeAdapter <: AbstractSLModelAdapter end
ScottishLowerProtocol.sl_capabilities(::FakeAdapter) = (; posterior_schema=(; varinfo_sites=[:scale, :effects], chain_columns=n -> [:scale, Symbol("effects[1]"), Symbol("effects[2]")], parameter_count=n -> n + 1), funnel_sites=[:scale], expected_score_dispatch=nothing)
ScottishLowerProtocol.sl_referee_eval(::FakeAdapter, ::Symbol, args...) = NamedTuple[]
ScottishLowerProtocol.sl_model(::FakeAdapter) = (; name=:fake)
ScottishLowerProtocol.sl_model_name(::FakeAdapter) = "fake"
ScottishLowerProtocol.sl_required_features(::FakeAdapter) = [:fixture]
ScottishLowerProtocol.sl_assert_model_contract(::FakeAdapter) = true
@testset "shared protocol contract" begin
 a=FakeAdapter(); c=sl_contract()
 @test sl_posterior_schema(a).parameter_count(2) == 3
 @test sl_artifact_hash(a,c) == sl_artifact_hash(a,c)
 @test sl_artifact_dir(a,c) != sl_legacy_artifact_dir(a,c)
 @test !only(sl_gate_score_grid(a, DataFrame(), c)).pass
 @test !only(sl_gate_contract((; matches=DataFrame(tournament_id=Int[])), SLFold[], c)[3:3]).pass
end
