using Test
using BayesianFootball
using DataFrames
using Dates
using Distributions
import DynamicPPL
using LinearAlgebra
import LogDensityProblems
using ReverseDiff
using TimeZones

const PW_PG = BayesianFootball.Models.PreGame
const PW_FEATURES = BayesianFootball.Features
const PW_SECONDS_PER_YEAR = 365.25 * 86_400.0

function pw_datastore(matches::DataFrame, lineups::DataFrame)
    empty = DataFrame()
    return BayesianFootball.Data.DataStore(
        BayesianFootball.Data.ScottishLower(), matches,
        empty, empty, lineups, empty, empty, empty, empty)
end

function pw_model_feature_set(n::Int=8)
    return BayesianFootball.FeatureSet(Dict{Symbol,Any}(
        :flat_home_ids => Int[isodd(i) ? 1 : 2 for i in 1:n],
        :flat_away_ids => Int[isodd(i) ? 2 : 1 for i in 1:n],
        :season_indices => ones(Int, n),
        :time_indices => ones(Int, n),
        :flat_months => Int[mod1(i, 12) for i in 1:n],
        :flat_home_goals => Int[mod(i, 3) for i in 1:n],
        :flat_away_goals => Int[mod(i + 1, 3) for i in 1:n],
        :dates => collect(0:(n - 1)),
        :flat_delta_production_wealth => collect(range(-0.5, 0.5; length=n)),
        :n_teams => 2,
        :n_seasons => 1,
        :n_rounds => 1,
        :team_map => Dict("home" => 1, "away" => 2),
    ))
end

@testset "Production-wealth age curves" begin
    ages = (18.0, 23.0, 27.5, 34.0, 38.0)
    richards = RichardsSigmoid()
    gamma_curve = ShiftedGamma()
    gaussian = GaussianPrime()

    for age in ages
        expected_richards =
            (1.0 + exp(-0.80 * (age - 23.0)))^(-1.0 / 2.0)
        gamma_x = age - 16.0
        gamma_mode = 27.5 - 16.0
        expected_gamma = gamma_x <= 0.0 ? 0.0 :
            (gamma_x / gamma_mode)^(3.5 - 1.0) *
            exp(-((3.5 - 1.0) / gamma_mode) * (gamma_x - gamma_mode))
        expected_gaussian = exp(-((age - 26.5)^2) / (2.0 * 4.5^2))

        @test age_weight(richards, age) ≈ expected_richards rtol=1e-14
        @test age_weight(gamma_curve, age) ≈ expected_gamma rtol=1e-14
        @test age_weight(gaussian, age) ≈ expected_gaussian rtol=1e-14
        @test richards(age) == age_weight(richards, age)
    end

    @test age_weight(gamma_curve, 16.0) == 0.0
    @test age_weight(gamma_curve, gamma_curve.peak) == 1.0
    @test age_weight(gaussian, gaussian.mu) == 1.0
    @test @inferred(age_weight(richards, 27.5)) isa Float64
    @test @inferred(age_weight(gamma_curve, 27.5)) isa Float64
    @test @inferred(age_weight(gaussian, 27.5)) isa Float64

    # Warm each scalar kernel before checking the inner valuation arithmetic.
    age_weight(richards, 27.5)
    age_weight(gamma_curve, 27.5)
    age_weight(gaussian, 27.5)
    @test @allocated(age_weight(richards, 27.5)) == 0
    @test @allocated(age_weight(gamma_curve, 27.5)) == 0
    @test @allocated(age_weight(gaussian, 27.5)) == 0
end

@testset "ProductionWealthFeature lineup, fallback, and PIT semantics" begin
    kickoff = DateTime(2024, 1, 20, 15, 30)
    kickoff_seconds = round(Int, datetime2unix(kickoff))
    prime_dob = kickoff_seconds - round(Int, 26.5 * PW_SECONDS_PER_YEAR)
    age_18_dob = kickoff_seconds - round(Int, 18.0 * PW_SECONDS_PER_YEAR)
    age_34_dob = kickoff_seconds - round(Int, 34.0 * PW_SECONDS_PER_YEAR)

    matches = DataFrame(
        match_id=Int32[11, 12, 13, 14, 15],
        # PostgreSQL timestamptz arrives through LibPQ as ZonedDateTime.
        start_timestamp=fill(ZonedDateTime(kickoff, tz"UTC"), 5),
        home_team=fill("alpha", 5),
        away_team=fill("beta", 5),
    )
    lineups = DataFrame(
        match_id=Int32[11, 11, 11, 12, 12, 14, 14, 15, 15],
        team_side=[
            "home", "away", "home", "home", "away",
            "home", "away", "home", "away",
        ],
        is_substitute=[false, false, true, false, false, false, false, false, false],
        proposed_market_value=Union{Missing,Float64}[
            400.0, 100.0, 1.0e12, 400.0, 100.0, 900.0, 100.0, 100.0, 100.0,
        ],
        date_of_birth_timestamp=Union{Missing,Int64}[
            prime_dob, prime_dob, prime_dob, missing, missing,
            prime_dob, prime_dob, age_18_dob, age_34_dob,
        ],
        valuation_timestamp=Union{Missing,DateTime}[
            missing, missing, missing, missing, missing,
            kickoff + Day(1), kickoff + Day(1), missing, missing,
        ],
    )

    config = ProductionWealthFeature(
        curve=GaussianPrime(), fallback_default=100.0,
        fallback_age=26.5, log_scale=2.0)
    feature_data = Dict{Symbol,Any}()
    PW_FEATURES.add_feature!(
        feature_data, config, [11, 12, 13, 14, 15],
        Dict("alpha" => 1, "beta" => 2), pw_datastore(matches, lineups))

    # The trillion-value bench player is excluded; only the starting XI enters.
    expected_delta = log(4.0) / config.log_scale
    @test feature_data[:flat_delta_production_wealth][1] ≈ expected_delta
    # Missing DOBs use 26.5 years. GaussianPrime(26.5) == 1, so the same
    # valuation ratio is retained exactly.
    @test feature_data[:flat_delta_production_wealth][2] ≈ expected_delta
    # No lineup and all future-stamped valuations both produce the neutral term.
    @test feature_data[:flat_delta_production_wealth][3:4] == [0.0, 0.0]
    # The DOB timestamps are converted to exact ages at this fixture's kickoff.
    expected_age_delta =
        log(age_weight(config.curve, 18.0) / age_weight(config.curve, 34.0)) /
        config.log_scale
    @test feature_data[:flat_delta_production_wealth][5] ≈ expected_age_delta atol=1e-12
    @test feature_data[:flat_production_wealth_fallback] == [0, 0, 1, 1, 0]
    @test eltype(feature_data[:flat_delta_production_wealth]) == Float64
    @test !haskey(feature_data[:production_wealth_oos_bridge_by_match_id], 14)

    expected_column = [
        expected_delta, expected_delta, 0.0, 0.0, expected_age_delta]
    covariate = ProductionWealthCovariate(feature=config)
    feature_set = BayesianFootball.FeatureSet(feature_data)
    @test covariate_column(covariate, feature_set) ≈ expected_column
    @test covariate_oos(
        covariate, feature_set,
        DataFrame(match_id=[11, 12, 13, 14, 15, 999])) ≈
        vcat(expected_column, 0.0)
    @test covariate_oos(
        covariate, feature_set,
        DataFrame(match_id=[999], delta_production_wealth=[0.75])) == [0.75]
end

@testset "Lineup DOB schema" begin
    raw = DataFrame(
        tournament_id=[56, 56], season_id=[1, 1], match_id=[1001, 1001],
        team_side=["home", "away"], player_id=[101, 202],
        date_of_birth_timestamp=Union{Missing,Int64}[631_152_000, missing],
        proposed_market_value=Union{Missing,Int64}[250_000, missing],
        proposed_market_value_currency=Union{Missing,String}["EUR", missing],
    )
    processed = BayesianFootball.Data.process_data(
        raw, BayesianFootball.Data.LineUpsData())
    @test eltype(processed.date_of_birth_timestamp) == Union{Missing,Int64}
    @test isequal(processed.date_of_birth_timestamp,
                  Union{Missing,Int64}[631_152_000, missing])
end

@testset "ProductionWealthCovariate composes into a ReverseDiff tape" begin
    builder = CountModelBuilder(:production_wealth_ad)
    add!(
        builder,
        PW_PG.GlobalInterception(μ=Normal(0.2, 0.1)),
        PW_PG.TimeDecayDynamics(
            days_half_life=180.0,
            σ_att=Gamma(2.0, 0.15),
            σ_def=Gamma(2.0, 0.15),
        ),
        PW_PG.GlobalHomeAdvantage(γ_global=Normal(0.2, 0.2)),
        ProductionWealthCovariate(),
        NoGuard(),
    )
    model = build_count_model(builder)

    @test model isa PW_PG.PoissonCountModel
    @test any(f -> f isa ProductionWealthFeature,
              PW_FEATURES.required_features(model))
    @test ProductionWealthCovariate ===
          BayesianFootball.Models.ProductionWealthCovariate
    @test ProductionWealthCovariate === PW_PG.ProductionWealthCovariate

    turing_model = PW_PG.build_turing_model(model, pw_model_feature_set())
    varinfo = DynamicPPL.VarInfo(turing_model)
    turing_model(varinfo)
    theta = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model)
    objective = x -> LogDensityProblems.logdensity(density, x)

    raw_tape = ReverseDiff.GradientTape(objective, theta)
    compiled_tape = ReverseDiff.compile(raw_tape)
    gradient = similar(theta)
    ReverseDiff.gradient!(gradient, compiled_tape, theta)

    @test length(raw_tape.tape) < 250
    @test isfinite(objective(theta))
    @test all(isfinite, gradient)
    @test gradient ≈ ReverseDiff.gradient(objective, theta) rtol=1e-8 atol=1e-8
end
