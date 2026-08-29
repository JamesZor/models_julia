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

const BuilderPG = BayesianFootball.Models.PreGame
const BuilderFeatures = BayesianFootball.Features
const BuilderAPI = BuilderPG.Builder

function builder_feature_set(n::Int=6)
    home = Int[isodd(i) ? 1 : 2 for i in 1:n]
    away = Int[isodd(i) ? 2 : 1 for i in 1:n]
    return BayesianFootball.FeatureSet(Dict{Symbol,Any}(
        :flat_home_ids => home,
        :flat_away_ids => away,
        :season_indices => ones(Int, n),
        :time_indices => ones(Int, n),
        :flat_months => Int[mod1(i, 12) for i in 1:n],
        :flat_home_goals => Int[mod(i, 3) for i in 1:n],
        :flat_away_goals => Int[mod(i + 1, 3) for i in 1:n],
        :dates => collect(0:(n - 1)),
        :flat_delta_wealth_logsum => collect(range(-0.4, 0.4; length=n)),
        :flat_distance => collect(range(-1.0, 1.0; length=n)),
        :n_teams => 2,
        :n_seasons => 1,
        :n_rounds => 1,
        :team_map => Dict("home" => 1, "away" => 2),
    ))
end

function builder_base_components()
    return (
        BuilderPG.GlobalInterception(μ=Normal(0.2, 0.1)),
        BuilderPG.TimeDecayDynamics(
            days_half_life=180.0,
            σ_att=Gamma(2.0, 0.15),
            σ_def=Gamma(2.0, 0.15),
        ),
        BuilderPG.GlobalHomeAdvantage(γ_global=Normal(0.2, 0.2)),
    )
end

function builder_density_artifacts(model, feature_set; seed::Int=20260828)
    turing_model = BuilderPG.build_turing_model(model, feature_set)
    Random.seed!(seed)
    varinfo = DynamicPPL.VarInfo(turing_model)
    turing_model(varinfo)
    θ = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model)
    f = x -> LogDensityProblems.logdensity(density, x)
    return (; turing_model, varinfo, θ, f)
end

function builder_site_vector(value)
    value isa AbstractArray && return vec(Float64.(value))
    return Float64[Float64(value)]
end

function builder_site_range(varinfo, target::AbstractString)
    offset = 1
    for site in keys(varinfo)
        width = length(builder_site_vector(varinfo[site]))
        string(site) == target && return offset:(offset + width - 1)
        offset += width
    end
    error("site $target not found")
end

function builder_synthetic_chain(model; n_teams::Int=2, n_seasons::Int=1,
                                 n_draws::Int=4)
    columns = cb_chain_columns(model, n_teams; n_seasons)
    values = zeros(Float64, n_draws, length(columns), 1)
    for (j, name) in enumerate(columns)
        value = startswith(name, "inter.") ? 0.2 :
                startswith(name, "ha.") ? 0.15 :
                occursin("dyn.σ_", name) ? 0.2 :
                startswith(name, "disp.log_r") ? 3.1 :
                endswith(name, ".w") ? 0.1 : 0.0
        values[:, j, 1] .= value
    end
    return Chains(values, Symbol.(columns))
end

@testset "Composable count-model builder" begin
    @test CountModelBuilder === BayesianFootball.Models.CountModelBuilder
    @test CountModelBuilder === BuilderPG.CountModelBuilder
    @test WealthCovariate().feature isa LogSumWealthFeature
    @test DistanceCovariate().feature isa BuilderFeatures.DistanceFeature

    base = builder_base_components()
    builder = CountModelBuilder(:construction)
    @test add!(builder, base...) === builder
    @test add!(builder, WealthCovariate(), DistanceCovariate(), NoGuard()) === builder
    @test all(row.pass for row in validate(builder))

    model = build_count_model(builder)
    @test model isa PoissonCountModel
    @test model isa BayesianFootball.TypesInterfaces.AbstractPoissonModel
    @test model.covariates isa Tuple
    @test length(model.covariates) == 2
    @test model.guard isa NoGuard
    @test cb_parameter_count(model, 2) == 10
    @test cb_varinfo_sites(model) == [
        Symbol("inter.μ"), Symbol("ha.γ_global"), Symbol("dyn.σ_a"),
        Symbol("dyn.σ_d"), Symbol("dyn.raw_a"), Symbol("dyn.raw_d"),
        Symbol("wealth.w"), Symbol("distance.w"),
    ]
    required = BuilderFeatures.required_features(model)
    @test any(f -> f isa LogSumWealthFeature, required)
    @test any(f -> f isa BuilderFeatures.DistanceFeature, required)
    @test BayesianFootball.Models.latent_family(model) isa
          BayesianFootball.Models.PoissonCountFamily

    direct = build_count_model(
        :direct, base..., BuilderPG.GlobalDispersion(), ClampGuard())
    @test direct isa NegBinCountModel
    @test direct isa BayesianFootball.TypesInterfaces.AbstractNegBinModel
    @test BayesianFootball.Models.latent_family(direct) isa
          BayesianFootball.Models.NegBinCountFamily

    @test_throws ErrorException build_count_model(CountModelBuilder(:empty))
    duplicate = CountModelBuilder(:duplicate)
    add!(duplicate, base...)
    @test_throws ErrorException add!(duplicate, BuilderPG.StaticZeroDynamics())
    @test replace!(duplicate, BuilderPG.StaticZeroDynamics()) === duplicate
    @test duplicate.dynamics isa BuilderPG.StaticZeroDynamics
    @test_throws ErrorException build_count_model(
        :bad_half_life,
        base[1], BuilderPG.TimeDecayDynamics(days_half_life=0.0), base[3])
    @test_throws ErrorException build_count_model(
        :unwired, base..., BuilderPG.GlobalDixonColesConfig())
    @test_throws ErrorException build_count_model(
        :player_dynamics, base[1], BuilderPG.PositionalPlayerDynamics(), base[3])
    @test_throws ErrorException build_count_model(
        :league_home_advantage, base[1], base[2],
        BuilderPG.HierarchicalLeagueHomeAdvantage())
    @test_throws ErrorException build_count_model(
        :discrete_covariate, base..., WealthCovariate(prior=Bernoulli()))
    @test_throws ErrorException build_count_model(
        :negbin_without_floor_guard, base..., BuilderPG.GlobalDispersion(), NoGuard())
    @test_throws Exception WealthCovariate(feature=BuilderFeatures.DistanceFeature())
    @test_throws Exception DistanceCovariate(feature=BuilderFeatures.SquadWealthFeature())

    seasonal = build_count_model(
        :seasonal_schema, BuilderPG.SeasonalInterception(), base[2], base[3])
    @test cb_chain_columns(seasonal, 2; n_seasons=3)[1:3] ==
          ["inter.μ[1]", "inter.μ[2]", "inter.μ[3]"]
    @test cb_parameter_count(seasonal, 2; n_seasons=3) == 10
    seasonal_fs = builder_feature_set(6)
    seasonal_fs[:n_seasons] = 3
    seasonal_fs[:season_indices] = [1, 2, 3, 1, 2, 3]
    seasonal_artifacts = builder_density_artifacts(seasonal, seasonal_fs; seed=19)
    @test length(seasonal_artifacts.θ) == cb_parameter_count(
        seasonal, 2; n_seasons=3)

    monthly = build_count_model(
        :monthly_schema, BuilderPG.HierarchicalMonthlyInterception(), base[2], base[3])
    monthly_columns = cb_chain_columns(monthly, 2; n_seasons=3)
    @test monthly_columns[1:3] ==
          ["inter.μ_base[1]", "inter.μ_base[2]", "inter.μ_base[3]"]
    @test count(name -> startswith(name, "inter.raw_month["), monthly_columns) == 12
end

@testset "Composable engine has a static, compile-time-unrolled AD kernel" begin
    base = builder_base_components()
    model = build_count_model(
        :ad_kernel, base..., WealthCovariate(), DistanceCovariate(), NoGuard())

    small = builder_density_artifacts(model, builder_feature_set(6))
    large = builder_density_artifacts(model, builder_feature_set(60))
    raw_small = ReverseDiff.GradientTape(small.f, small.θ)
    raw_large = ReverseDiff.GradientTape(large.f, large.θ)

    # A vectorized typed-tuple walk has a tape shape independent of observation count.
    @test length(raw_small.tape) == length(raw_large.tape)
    @test length(raw_small.tape) < 250

    tape = ReverseDiff.compile(raw_small)
    compiled_gradient = similar(small.θ)
    ReverseDiff.gradient!(compiled_gradient, tape, small.θ)
    fresh_gradient = ReverseDiff.gradient(small.f, small.θ)
    forward_gradient = ForwardDiff.gradient(small.f, small.θ)
    relerr(a, b) = norm(a - b) / max(norm(a), norm(b), 1.0)

    @test isfinite(small.f(small.θ))
    @test all(isfinite, compiled_gradient)
    @test relerr(compiled_gradient, fresh_gradient) <= 1e-8
    @test relerr(compiled_gradient, forward_gradient) <= 1e-6

    perturbed = small.θ .+ 0.002 .* sin.(collect(eachindex(small.θ)))
    perturbed_compiled = similar(perturbed)
    ReverseDiff.gradient!(perturbed_compiled, tape, perturbed)
    @test relerr(perturbed_compiled, ReverseDiff.gradient(small.f, perturbed)) <= 1e-8

    # The production engine also agrees with its independent equations implementation.
    equation_data = BuilderAPI.cb_equation_data(model, builder_feature_set(6))
    params = BuilderAPI.cb_params_from_varinfo(model, small.varinfo)
    @test small.f(small.θ) ≈ BuilderAPI.cb_logjoint(model, params, equation_data) atol=1e-9

    # The production default is the explicit ClampGuard, not NoGuard.
    guarded_model = build_count_model(
        :guarded_ad, base..., WealthCovariate(), DistanceCovariate())
    @test guarded_model.guard isa ClampGuard
    guarded = builder_density_artifacts(guarded_model, builder_feature_set(6); seed=42)
    guarded_tape = ReverseDiff.compile(ReverseDiff.GradientTape(guarded.f, guarded.θ))
    for δ in (0.0, 0.001, -0.002, 0.003)
        guarded_point = guarded.θ .+ δ .* sin.(collect(eachindex(guarded.θ)))
        guarded_compiled = similar(guarded_point)
        ReverseDiff.gradient!(guarded_compiled, guarded_tape, guarded_point)
        @test relerr(guarded_compiled,
                     ForwardDiff.gradient(guarded.f, guarded_point)) <= 1e-6
    end
end

@testset "Negative-binomial observations are compiled-ReverseDiff safe" begin
    base = builder_base_components()
    for dispersion in (BuilderPG.GlobalDispersion(), BuilderPG.HomeAwayDispersion())
        model = build_count_model(
            :negbin_ad, base..., dispersion)
        artifacts = builder_density_artifacts(model, builder_feature_set(10); seed=73)
        raw_tape = ReverseDiff.GradientTape(artifacts.f, artifacts.θ)
        tape = ReverseDiff.compile(raw_tape)
        compiled = similar(artifacts.θ)
        ReverseDiff.gradient!(compiled, tape, artifacts.θ)
        forward = ForwardDiff.gradient(artifacts.f, artifacts.θ)
        relerr = norm(compiled - forward) / max(norm(compiled), norm(forward), 1.0)

        @test isfinite(artifacts.f(artifacts.θ))
        @test all(isfinite, compiled)
        @test relerr <= 1e-6
        @test length(raw_tape.tape) < 250
        larger = builder_density_artifacts(model, builder_feature_set(100); seed=73)
        @test length(ReverseDiff.GradientTape(larger.f, larger.θ).tape) ==
              length(raw_tape.tape)

        perturbed = artifacts.θ .+ 0.002 .* sin.(collect(eachindex(artifacts.θ)))
        perturbed_compiled = similar(perturbed)
        ReverseDiff.gradient!(perturbed_compiled, tape, perturbed)
        perturbed_forward = ForwardDiff.gradient(artifacts.f, perturbed)
        perturbed_relerr = norm(perturbed_compiled - perturbed_forward) /
                           max(norm(perturbed_compiled), norm(perturbed_forward), 1.0)
        @test perturbed_relerr <= 1e-6

        # A compiled tape recorded in the typical region must remain valid after
        # crossing both former hard-clamp boundaries.
        log_r_range = builder_site_range(artifacts.varinfo, "disp.log_r")
        for log_r in (-12.0, -10.5, 10.5, 12.0)
            crossed = copy(artifacts.θ)
            crossed[log_r_range] .= log_r
            crossed_compiled = similar(crossed)
            ReverseDiff.gradient!(crossed_compiled, tape, crossed)
            crossed_fresh = ReverseDiff.gradient(artifacts.f, crossed)
            crossed_forward = ForwardDiff.gradient(artifacts.f, crossed)
            fresh_relerr = norm(crossed_compiled - crossed_fresh) /
                           max(norm(crossed_compiled), norm(crossed_fresh), 1.0)
            forward_relerr = norm(crossed_compiled - crossed_forward) /
                             max(norm(crossed_compiled), norm(crossed_forward), 1.0)
            @test fresh_relerr <= 1e-8
            @test forward_relerr <= 1e-6
        end

        # Extraction applies the identical smooth bound used by the likelihood.
        columns = cb_chain_columns(model, 2)
        chain_values = zeros(Float64, 2, length(columns), 1)
        chain_values[:, findfirst(==("disp.log_r"), columns), 1] .= 12.0
        home_log_r = 12.0
        if dispersion isa BuilderPG.HomeAwayDispersion
            chain_values[:, findfirst(==("disp.δ_r_home"), columns), 1] .= 1.0
            home_log_r += 1.0
        end
        chain = Chains(chain_values, Symbol.(columns))
        extracted_dispersion = BuilderAPI._cb_extract_observation(
            model.observation, chain, 2)
        expected_away = exp(BuilderAPI._cb_bound_dispersion_log(12.0))
        expected_home = exp(BuilderAPI._cb_bound_dispersion_log(home_log_r))
        @test extracted_dispersion.a == fill(expected_away, 2)
        @test extracted_dispersion.h == fill(expected_home, 2)
    end
end

@testset "Composable model parity with the legacy PreGame count engine" begin
    inter, dynamics, home_advantage = builder_base_components()
    dispersion = BuilderPG.GlobalDispersion(log_r=Normal(3.1, 0.4))
    feature_set = builder_feature_set(8)

    composable = build_count_model(
        :legacy_parity, inter, dynamics, home_advantage, dispersion, ClampGuard())
    legacy = BuilderPG.DynamicGoalsTimeDecayModel(
        interception_config=inter,
        dynamics_config=dynamics,
        dispersion_config=dispersion,
        homeadvantage_config=home_advantage,
    )

    new_draw = builder_density_artifacts(composable, feature_set; seed=991)
    legacy_artifacts = builder_density_artifacts(legacy, feature_set; seed=7)
    values_by_site = Dict(string(k) => new_draw.varinfo[k] for k in keys(new_draw.varinfo))

    @test Set(string.(keys(new_draw.varinfo))) == Set(string.(keys(legacy_artifacts.varinfo)))
    legacy_θ = Float64[]
    for site in keys(legacy_artifacts.varinfo)
        append!(legacy_θ, builder_site_vector(values_by_site[string(site)]))
    end

    # Same named parameter point, with only the two engines' declaration order remapped.
    @test new_draw.f(new_draw.θ) ≈ legacy_artifacts.f(legacy_θ) atol=1e-9

    # Legacy constructors and feature contracts remain untouched.
    @test legacy isa BuilderPG.AbstractPregameModel
    legacy_required = BuilderFeatures.required_features(legacy)
    @test any(f -> f isa BuilderFeatures.TeamIDsFeature, legacy_required)
    @test isfinite(legacy_artifacts.f(legacy_artifacts.θ))
end

@testset "Builder feature semantics and OOS extraction" begin
    matches = DataFrame(
        match_id=Int32[11, 12],
        start_timestamp=DateTime[DateTime(2024, 1, 2), DateTime(2024, 1, 10)],
        home_team=["home", "home"],
        away_team=["away", "away"],
    )
    lineups = DataFrame(
        match_id=Int32[11, 11, 11, 11, 12, 12],
        team_side=["home", "home", "away", "away", "home", "away"],
        is_substitute=fill(false, 6),
        proposed_market_value=[300.0, 100.0, 100.0, 100.0, 900.0, 100.0],
        valuation_timestamp=DateTime[
            DateTime(2024, 1, 1), DateTime(2024, 1, 1),
            DateTime(2024, 1, 1), DateTime(2024, 1, 1),
            DateTime(2024, 1, 11), DateTime(2024, 1, 11),
        ],
    )
    empty = DataFrame()
    datastore = BayesianFootball.Data.DataStore(
        BayesianFootball.Data.Ireland(), matches,
        empty, empty, lineups, empty, empty, empty, empty)

    config = LogSumWealthFeature(fallback_default=100.0, log_scale=1.0)
    feature_data = Dict{Symbol,Any}()
    BuilderFeatures.add_feature!(
        feature_data, config, [11], Dict("home" => 1, "away" => 2), datastore)
    @test feature_data[:flat_delta_wealth_logsum] ≈ [log(2.0)]
    @test feature_data[:flat_wealth_fallback] == [0]
    # Match 12's only valuations are stamped after kickoff, so its OOS bridge is neutral.
    @test !haskey(feature_data[:wealth_oos_bridge_by_match_id], 12)

    wealth = WealthCovariate(feature=config)
    wealth_oos = covariate_oos(
        wealth, BayesianFootball.FeatureSet(feature_data),
        DataFrame(match_id=[11, 12, 99]))
    @test wealth_oos ≈ [log(2.0), 0.0, 0.0]

    # A materialised distance must match the configured metric; an unrelated
    # `distance_z` column cannot override road mileage.
    distance = DistanceCovariate(
        feature=BuilderFeatures.DistanceFeature(metric=:road_miles))
    distance_df = DataFrame(match_id=[1, 2], road_miles=[25.0, 40.0],
                            distance=[-100.0, -100.0], distance_z=[-9.0, -9.0])
    @test covariate_oos(distance, BayesianFootball.FeatureSet(), distance_df) ==
          [25.0, 40.0]

    base = builder_base_components()
    model = build_count_model(:oos_extract, base..., wealth, NoGuard())
    fs = builder_feature_set(4)
    fixtures = DataFrame(
        match_id=[101], home_team=["home"], away_team=["away"],
        match_date=[Date(2024, 2, 1)], season_idx=[1],
        delta_wealth_logsum=[0.3],
    )
    chain = builder_synthetic_chain(model)
    extracted = BuilderPG.extract_parameters(model, fixtures, fs, chain)
    @test haskey(extracted, 101)
    @test all(isfinite, extracted[101].λ_h)
    @test all(isfinite, extracted[101].λ_a)
    @test extracted[101].λ_h != extracted[101].λ_a
end
