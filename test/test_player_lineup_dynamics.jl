using Test
using BayesianFootball
using DataFrames
using Dates
using Distributions
using DynamicPPL
using ForwardDiff
using LinearAlgebra
using LogDensityProblems
using MCMCChains
using Random
using ReverseDiff
using Statistics
using Turing

const PLD_PG = BayesianFootball.Models.PreGame
const PLD_API = PLD_PG.Builder
const PLD_F = BayesianFootball.Features

function pld_feature_set(n::Int=12)
    x = collect(range(-0.8, 0.8; length=n))
    return FeatureSet(Dict{Symbol,Any}(
        :flat_home_ids => ones(Int, n),
        :flat_away_ids => fill(2, n),
        :season_indices => ones(Int, n),
        :flat_months => Int[mod1(i, 12) for i in 1:n],
        :flat_home_goals => Int[round(Int, 1.4 + 0.5 * x[i]) for i in 1:n],
        :flat_away_goals => Int[round(Int, 1.1 - 0.4 * x[i]) for i in 1:n],
        :dates => collect(0:(n - 1)),
        :n_teams => 2,
        :n_seasons => 1,
        :team_map => Dict("home" => 1, "away" => 2),
        :flat_home_outfield_rating => x .+ 0.3,
        :flat_away_outfield_rating => .-x .+ 0.1,
        :flat_home_bench_rating => 0.3 .* x,
        :flat_away_bench_rating => .-0.2 .* x,
        :flat_home_D_rating => 0.2 .* x,
        :flat_home_M_rating => 0.4 .* x,
        :flat_home_F_rating => 0.4 .* x .+ 0.3,
        :flat_away_D_rating => .-0.3 .* x,
        :flat_away_M_rating => .-0.4 .* x,
        :flat_away_F_rating => .-0.3 .* x .+ 0.1,
        :flat_home_bench_D_rating => 0.05 .* x,
        :flat_home_bench_M_rating => 0.10 .* x,
        :flat_home_bench_F_rating => 0.15 .* x,
        :flat_away_bench_D_rating => .-0.04 .* x,
        :flat_away_bench_M_rating => .-0.08 .* x,
        :flat_away_bench_F_rating => .-0.08 .* x,
        :flat_home_minute_weighted_rating => 0.85 .* x .+ 0.2,
        :flat_away_minute_weighted_rating => .-0.75 .* x .+ 0.1,
        :flat_pxg_home => fill(1.4, n),
        :flat_pxg_away => fill(1.1, n),
        :flat_pxg_obs_available => ones(Float64, n),
        :flat_delta_production_wealth => zeros(Float64, n),
        :production_wealth_oos_bridge_by_match_id => Dict{Int,Float64}(),
        :flat_distance => zeros(Float64, n),
    ))
end

function pld_model(aggregation, observation=PoissonObservation(); bench_prior=nothing)
    return build_count_model(
        :player_lineup_test,
        GlobalInterception(μ=Normal(0.2, 0.1)),
        TimeDecayDynamics(
            days_half_life=180.0,
            σ_att=Gamma(2.0, 0.15),
            σ_def=Gamma(2.0, 0.15),
        ),
        GlobalHomeAdvantage(γ_global=Normal(0.2, 0.2)),
        PlayerLineupPillar(
            feature=PLD_F.XGPlusMinusFeature(),
            aggregation=aggregation,
            w_att_prior=Normal(0.0, 0.3),
            w_def_prior=Normal(0.0, 0.3),
            w_bench_prior=bench_prior,
        ),
        observation,
        ClampGuard(),
    )
end

function pld_joint_contract_models()
    base() = (
        GlobalInterception(μ=Normal(0.2, 0.1)),
        TimeDecayDynamics(days_half_life=180.0),
        GlobalHomeAdvantage(γ_global=Normal(0.2, 0.2)),
    )
    lineup() = PlayerLineupPillar(
        feature=PLD_F.XGPlusMinusFeature(),
        aggregation=OutfieldPlayerAggregation(),
    )
    wealth() = ProductionWealthCovariate()
    distance() = DistanceCovariate()
    observation() = JointGammaPoissonObservation()
    return Dict(
        :m05 => build_count_model(:m05, base()..., wealth(), observation()),
        :m09 => build_count_model(:m09, base()..., lineup(), observation()),
        :m10 => build_count_model(:m10, base()..., lineup(), observation()),
        :m11 => build_count_model(:m11, base()..., lineup(), observation()),
        :m12 => build_count_model(:m12, base()..., lineup(), wealth(), observation()),
        :m13 => build_count_model(
            :m13, base()..., lineup(), wealth(), distance(), observation()),
    )
end

pld_expected_params(name::Symbol, n_teams::Int) =
    name === :m05 ? 2 * n_teams + 7 :
    name in (:m09, :m10, :m11) ? 2 * n_teams + 8 :
    name === :m12 ? 2 * n_teams + 9 :
    name === :m13 ? 2 * n_teams + 10 :
    error("no structural parameter contract for $name")

function pld_density(model, fs; seed=711)
    turing_model = PLD_PG.build_turing_model(model, fs)
    Random.seed!(seed)
    varinfo = DynamicPPL.VarInfo(turing_model)
    turing_model(varinfo)
    theta = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model)
    f = x -> LogDensityProblems.logdensity(density, x)
    return (; turing_model, varinfo, theta, f)
end

@testset "Player lineup feature aggregation contract" begin
    matches = DataFrame(match_id=[1, 2], match_date=[Date(2024, 1, 1), Date(2024, 1, 8)])
    lineups = DataFrame(
        match_id=[1, 1, 1, 2, 2],
        player_id=[10, 11, 20, 10, 20],
        team_side=["home", "home", "away", "home", "away"],
        position=["D", "F", "M", "D", "M"],
        is_substitute=[false, true, false, false, false],
        minutes_played=Union{Missing,Float64}[90.0, 20.0, 80.0, missing, missing],
    )
    ratings = Dict(10 => 1.0, 11 => 2.0, 20 => -0.5)
    aggregates = PLD_F.pm_lineup_aggregates(lineups, matches, ratings)
    @test aggregates[1].home_outfield == 1.0
    @test aggregates[1].home_bench == 2.0
    @test aggregates[1].away_M == -0.5
    @test aggregates[1].home_minute == 1.0
    @test aggregates[2].home_minute == 1.0
    @test aggregates[2].away_minute ≈ -0.5 * (80.0 / 90.0)

    data = Dict{Symbol,Any}()
    PLD_F._emit_pm_lineup_vectors!(data, [1, 2], aggregates)
    for key in (
        :flat_home_outfield_rating, :flat_away_outfield_rating,
        :flat_home_bench_rating, :flat_away_bench_rating,
        :flat_home_D_rating, :flat_home_M_rating, :flat_home_F_rating,
        :flat_away_D_rating, :flat_away_M_rating, :flat_away_F_rating,
        :flat_home_minute_weighted_rating, :flat_away_minute_weighted_rating,
    )
        @test haskey(data, key)
        @test data[key] isa Vector{Float64}
        @test length(data[key]) == 2
    end
end

@testset "CountModelBuilder accepts all player aggregation strategies" begin
    fs = pld_feature_set()
    strategies = (
        OutfieldPlayerAggregation(),
        BenchWeightedPlayerAggregation(w_bench=0.25),
        PositionalPlayerAggregation(),
        MinuteWeightedPlayerAggregation(),
    )
    for strategy in strategies
        model = pld_model(strategy)
        @test model isa PoissonCountModel
        @test model.dynamics isa TimeDecayDynamics
        @test first(cb_predictor_terms(model)) isa PlayerLineupPillar
        @test any(f -> f isa PLD_F.XGPlusMinusFeature, PLD_F.required_features(model))
        design = @inferred predictor_design(first(cb_predictor_terms(model)), fs, 12)
        @test design isa PLD_API.AbstractPlayerLineupDesign
        artifacts = pld_density(model, fs)
        @test isfinite(artifacts.f(artifacts.theta))
    end

    learned_bench = pld_model(
        BenchWeightedPlayerAggregation(),
        PoissonObservation();
        bench_prior=truncated(Normal(0.25, 0.10), 0.0, 1.0),
    )
    @test Symbol("lineup.w_bench") in cb_varinfo_sites(learned_bench)
end

@testset "Hybrid structural model contract" begin
    n_teams = 14
    models = pld_joint_contract_models()
    team_sites = Set(Symbol.(("dyn.raw_a", "dyn.raw_d", "dyn.σ_a", "dyn.σ_d")))
    lineup_sites = Set(Symbol.(("lineup.w_att", "lineup.w_def")))

    for (name, model) in models
        n_params = cb_parameter_count(model, n_teams)
        @test n_params == pld_expected_params(name, n_teams)
        sites = Set(cb_varinfo_sites(model))
        columns = Set(cb_chain_columns(model, n_teams))
        @test team_sites ⊆ sites
        @test "dyn.σ_a" in columns
        @test "dyn.σ_d" in columns
        @test "dyn.raw_a[1]" in columns
        @test "dyn.raw_d[1]" in columns
        if name !== :m05
            @test lineup_sites ⊆ sites
            @test "lineup.w_att" in columns
            @test "lineup.w_def" in columns
        end
        if name in (:m05, :m12, :m13)
            @test Symbol("production_wealth.w") in sites
            @test "production_wealth.w" in columns
        end
    end

    hybrid_design = PLD_API.cb_design(models[:m12], pld_feature_set(12))
    @test hybrid_design.match_weights ==
          0.5 .^ (collect(0.0:11.0) ./ 180.0)
end

@testset "Player lineup dynamics supports every wired observation" begin
    @test NegBinObservation === NegativeBinomialObservation
    fs = pld_feature_set()
    observations = (
        PoissonObservation(),
        NegBinObservation(),
        JointGammaPoissonObservation(),
    )
    for observation in observations
        model = pld_model(OutfieldPlayerAggregation(), observation)
        artifacts = pld_density(model, fs; seed=913)
        @test isfinite(artifacts.f(artifacts.theta))
        @test model isa (observation isa NegativeBinomialObservation ?
                         NegBinCountModel : PoissonCountModel)
    end
end

@testset "Player lineup OOS extraction uses the lineup bridge" begin
    model = pld_model(OutfieldPlayerAggregation())
    fs = pld_feature_set(8)
    aggregate = merge(PLD_F._pm_empty_lineup_aggregate(),
                      (home_outfield=0.7, away_outfield=-0.2))
    fs[:player_lineup_ratings_map] = Dict(99 => aggregate)

    columns = cb_chain_columns(model, 2)
    values = zeros(Float64, 3, length(columns), 1)
    values[:, findfirst(==("inter.μ"), columns), 1] .= 0.2
    values[:, findfirst(==("ha.γ_global"), columns), 1] .= 0.1
    values[:, findfirst(==("dyn.σ_a"), columns), 1] .= 0.2
    values[:, findfirst(==("dyn.σ_d"), columns), 1] .= 0.2
    values[:, findfirst(==("dyn.raw_a[1]"), columns), 1] .= 1.0
    values[:, findfirst(==("dyn.raw_a[2]"), columns), 1] .= -1.0
    values[:, findfirst(==("lineup.w_att"), columns), 1] .= 0.25
    values[:, findfirst(==("lineup.w_def"), columns), 1] .= 0.15
    chain = Chains(values, Symbol.(columns))
    fixtures = DataFrame(
        match_id=[99, 100, 101],
        home_team=["home", "home", "away"],
        away_team=["away", "away", "home"],
        match_date=fill(Date(2025, 1, 4), 3),
        season_idx=ones(Int, 3),
    )
    extracted = PLD_PG.extract_parameters(model, fixtures, fs, chain)
    @test all(isfinite, extracted[99].λ_h)
    @test all(isfinite, extracted[99].λ_a)
    @test extracted[99].λ_h != extracted[99].λ_a

    # 100 and 101 have identical neutral (missing) lineup inputs. Their rates
    # must still differ because dyn.α/dyn.β preserve team identity OOS.
    @test extracted[100].λ_h != extracted[101].λ_h
end

@testset "Player lineup ReverseDiff tape is fast and gradient-correct" begin
    fs = pld_feature_set(700)
    artifacts = pld_density(pld_model(OutfieldPlayerAggregation()), fs; seed=112)
    raw = ReverseDiff.GradientTape(artifacts.f, artifacts.theta)
    tape = ReverseDiff.compile(raw)
    gradient = similar(artifacts.theta)
    for _ in 1:40
        ReverseDiff.gradient!(gradient, tape, artifacts.theta)
    end
    elapsed = minimum([
        @elapsed ReverseDiff.gradient!(gradient, tape, artifacts.theta) for _ in 1:200
    ])
    fresh = ReverseDiff.gradient(artifacts.f, artifacts.theta)
    forward = ForwardDiff.gradient(artifacts.f, artifacts.theta)
    relerr(a, b) = norm(a - b) / max(norm(a), norm(b), 1.0)

    @test length(raw.tape) < 250
    @test elapsed < 1.0e-4
    @test relerr(gradient, fresh) <= 1e-8
    @test relerr(gradient, forward) <= 1e-6

    perturbed = artifacts.theta .+ 0.002 .* sin.(collect(eachindex(artifacts.theta)))
    compiled_perturbed = similar(perturbed)
    ReverseDiff.gradient!(compiled_perturbed, tape, perturbed)
    @test relerr(compiled_perturbed, ReverseDiff.gradient(artifacts.f, perturbed)) <= 1e-8
end

@testset "Single-fold mock inference samples non-zero lineup weights" begin
    Random.seed!(20260901)
    model = pld_model(OutfieldPlayerAggregation())
    turing_model = PLD_PG.build_turing_model(model, pld_feature_set(24))
    chain = sample(turing_model, NUTS(5, 0.65), 12; progress=false)
    w_att = vec(Array(chain[Symbol("lineup.w_att")]))
    w_def = vec(Array(chain[Symbol("lineup.w_def")]))
    @test length(w_att) == 12
    @test length(w_def) == 12
    @test all(isfinite, w_att)
    @test all(isfinite, w_def)
    @test any(!iszero, w_att)
    @test any(!iszero, w_def)
    @test std(w_att) > 0.0
    @test std(w_def) > 0.0
end
