# test/test_pxg_rapm_features.jl
#
# The pxG form covariate and the stint-RAPM covariate, from the feature engines up to a compiled
# ReverseDiff tape.
#
# The three properties worth defending, in order of how expensive they are to get wrong:
#   1. POINT-IN-TIME. A fixture must never see itself, its same-kickoff siblings, or anything later.
#      The last of those is tested destructively: a later result is rewritten and every earlier
#      column must come back bit-identical.
#   2. NEUTRAL DEGRADATION. No coverage is a 0.0 column, never an error and never a NaN — the
#      builder rejects a non-finite design vector, and a NaN three hours into a grid run is the
#      expensive failure this contract exists to prevent.
#   3. AD SAFETY. Both covariates must compose into one compiled tape with distinct chain sites.

using Test
using BayesianFootball
using DataFrames
using Dates
using Distributions
using Statistics
import DynamicPPL
import LogDensityProblems
using ReverseDiff

const PXG_PG = BayesianFootball.Models.PreGame
const PXG_FEATURES = BayesianFootball.Features

# ==============================================================================
# FIXTURES
# ==============================================================================

"A DataStore carrying only the domains these features read."
function pxg_store(; matches = DataFrame(), lineups = DataFrame(),
                     incidents = DataFrame(), bbc = DataFrame(),
                     bbc_events = DataFrame())
    empty = DataFrame()
    return BayesianFootball.Data.DataStore(
        BayesianFootball.Data.ScottishLower(), matches,
        empty, empty, lineups, incidents, empty, bbc, bbc_events)
end

"""
Five fixtures with hand-computable goals-rung pxG. Matches 3 and 4 share a calendar day but have
different kickoffs, which is the same-card case the point-in-time walk has to get right.

    m1  2024-01-01  A 2-0 B
    m2  2024-01-08  C 1-1 D
    m3  2024-01-15  A 3-1 C   \\ same kickoff
    m4  2024-01-15  B 0-2 D   /
    m5  2024-01-22  A ? C
"""
function pxg_goal_matches(; m5_home_score = 1.0, m5_away_score = 1.0)
    return DataFrame(
        match_id = Int[1, 2, 3, 4, 5],
        tournament_id = fill(56, 5),
        season = fill("24/25", 5),
        match_date = Date[Date(2024, 1, 1), Date(2024, 1, 8), Date(2024, 1, 15),
                          Date(2024, 1, 15), Date(2024, 1, 22)],
        start_timestamp = DateTime[
            DateTime(2024, 1, 1, 15), DateTime(2024, 1, 8, 15),
            DateTime(2024, 1, 15, 15), DateTime(2024, 1, 15, 18),
            DateTime(2024, 1, 22, 15)],
        home_team = ["alpha", "gamma", "alpha", "beta", "alpha"],
        away_team = ["beta", "delta", "gamma", "delta", "gamma"],
        home_score = Union{Missing,Float64}[2.0, 1.0, 3.0, 0.0, m5_home_score],
        away_score = Union{Missing,Float64}[0.0, 1.0, 1.0, 2.0, m5_away_score],
    )
end

"The exactly-computable configuration: plain means over all history, no shrinkage, unit scale."
pxg_exact_config(; kwargs...) = PxGFeature(;
    lookback = 0, decay = :window, prior_weight = 0.0, min_matches = 1,
    fallback = :goals, scale = 1.0, kwargs...)

function pxg_extract(config, ds; ordered_ids = Int.(ds.matches.match_id), history = nothing)
    F = Dict{Symbol,Any}()
    history === nothing || (F[:history_match_ids] = Set(Int.(history)))
    PXG_FEATURES.add_feature!(F, config, ordered_ids, Dict{String,Int}(), ds)
    return F
end

function pxg_model_feature_set(; n::Int = 8, extra...)
    data = Dict{Symbol,Any}(
        :flat_home_ids => Int[isodd(i) ? 1 : 2 for i in 1:n],
        :flat_away_ids => Int[isodd(i) ? 2 : 1 for i in 1:n],
        :season_indices => ones(Int, n),
        :time_indices => ones(Int, n),
        :flat_months => Int[mod1(i, 12) for i in 1:n],
        :flat_home_goals => Int[mod(i, 3) for i in 1:n],
        :flat_away_goals => Int[mod(i + 1, 3) for i in 1:n],
        :dates => collect(0:(n - 1)),
        :n_teams => 2,
        :n_seasons => 1,
        :n_rounds => 1,
        :team_map => Dict("home" => 1, "away" => 2),
    )
    for (k, v) in pairs(extra)
        data[k] = v
    end
    return BayesianFootball.FeatureSet(data)
end

# ==============================================================================
# 1. CONFIGURATION CONTRACTS
# ==============================================================================

@testset "PxGFeature / PxGRapmFeature configuration" begin
    @test PxGFeature().decay === :exponential
    @test PxGFeature().half_life_matches == 16.0
    @test PxGFeature(lookback=3).lookback_matches == 3
    @test PxGFeature().fallback === :goals
    @test PxGRapmFeature().target === :y_xg
    @test PxGRapmFeature().fit_on === :history
    # y_xg is the least team-loaded target, so it inherits XGPlusMinusFeature's tuned penalty.
    @test PxGRapmFeature().lambda == 200.0
    @test PxGRapmFeature().w_sim == 0.0
    @test PxGRapmFeature().scale === nothing

    ds = pxg_store(matches = pxg_goal_matches())
    for bad in (PxGFeature(decay = :ewma), PxGFeature(fallback = :xg),
                PxGFeature(scale = 0.0), PxGFeature(lookback = -1),
                PxGFeature(lookback_matches = -1),
                PxGFeature(half_life_matches = 0.0), PxGFeature(prior_weight = -1.0))
        @test_throws ErrorException pxg_extract(bad, ds)
    end
    for bad in (PxGRapmFeature(target = :y_points), PxGRapmFeature(fit_on = :everything),
                PxGRapmFeature(lambda = -1.0), PxGRapmFeature(scale = 0.0),
                PxGRapmFeature(half_life_days = 0.0), PxGRapmFeature(shrink_segments = -1.0))
        F = Dict{Symbol,Any}(:history_match_ids => Set([1, 2, 3]))
        @test_throws ErrorException PXG_FEATURES.add_feature!(
            F, bad, [1, 2, 3], Dict{String,Int}(), ds)
    end
end

# ==============================================================================
# 2. THE pxG MEASUREMENT LADDER
# ==============================================================================

@testset "pxg_match_observations measurement ladder" begin
    matches = pxg_goal_matches()
    # Four commentary attempts on match 1, all from the same cell, exactly one converted.
    # fit_shot_xg then shrinks that cell to (1 + 25*0.25)/(4 + 25) = 0.25, so every attempt
    # is worth 0.25 and each side's pxG is exactly 0.5.
    bbc_events = DataFrame(
        match_id = Int[1, 1, 1, 1],
        time = Int[10, 20, 30, 40],
        added_time = Union{Missing,Int}[missing, missing, missing, missing],
        event_type = ["goal", "attempt_saved", "attempt_saved", "attempt_saved"],
        is_home_event = Union{Missing,Bool}[true, true, false, false],
        text = fill("Right footed shot from the centre of the box.", 4),
    )
    # Match 2 has a match page but no live text: 10 v 6 attempts at the league's own pxG-per-shot.
    bbc = DataFrame(
        match_id = Int[2],
        shots_h = Union{Missing,Float64}[10.0],
        shots_a = Union{Missing,Float64}[6.0],
    )
    ds = pxg_store(matches = matches, bbc = bbc, bbc_events = bbc_events)

    obs = PXG_FEATURES.pxg_match_observations(ds, PxGFeature())
    @test obs[1].source === :commentary
    @test obs[1].h ≈ 0.5
    @test obs[1].a ≈ 0.5

    per_shot = (obs[1].h + obs[1].a) / 4
    @test per_shot ≈ 0.25
    @test obs[2].source === :shot_counts
    @test obs[2].h ≈ 10.0 * per_shot
    @test obs[2].a ≈ 6.0 * per_shot

    # Everything else falls through to the scoreline.
    @test obs[3].source === :goals
    @test obs[3].h ≈ 3.0 && obs[3].a ≈ 1.0

    # The ladder stops where the config says it stops.
    only_text = PXG_FEATURES.pxg_match_observations(ds, PxGFeature(fallback = :none))
    @test collect(keys(only_text)) == [1]
    through_shots = PXG_FEATURES.pxg_match_observations(ds, PxGFeature(fallback = :shots))
    @test sort(collect(keys(through_shots))) == [1, 2]

    F = pxg_extract(PxGFeature(), ds)
    @test F[:pxg_source_counts][:commentary] == 1
    @test F[:pxg_source_counts][:shot_counts] == 1
    @test F[:pxg_source_counts][:goals] == 3
end

# ==============================================================================
# 3. POINT-IN-TIME INTEGRITY
# ==============================================================================

@testset "PxGFeature is point-in-time by construction" begin
    ds = pxg_store(matches = pxg_goal_matches())
    F = pxg_extract(pxg_exact_config(), ds)

    sup = F[:flat_pxg_supremacy]
    lev = F[:flat_pxg_level]

    # --- the first fixture of each side has no history at all -----------------------------
    @test sup[1] == 0.0 && lev[1] == 0.0
    @test sup[2] == 0.0 && lev[2] == 0.0
    @test F[:flat_pxg_available][1:2] == [0.0, 0.0]
    @test F[:flat_pxg_form_fallback][1:2] == [1, 1]

    # --- match 3: state is m1 + m2, league baseline 1.0 -----------------------------------
    # alpha scored 2 conceded 0 -> att +1.0, def -1.0. gamma drew 1-1 -> att 0.0, def 0.0.
    @test F[:flat_pxg_att_home][3] ≈ 1.0
    @test F[:flat_pxg_def_home][3] ≈ -1.0
    @test F[:flat_pxg_att_away][3] ≈ 0.0
    @test F[:flat_pxg_def_away][3] ≈ 0.0
    @test sup[3] ≈ 2.0            # (1.0 + 0.0) - (0.0 + -1.0)
    @test lev[3] ≈ 0.0
    @test F[:flat_pxg_available][3] == 1.0

    # --- match 4 SHARES match 3's day and must not have seen it ---------------------------
    # Had m3 landed first the league baseline would have moved to 8/6, so these two values
    # are exactly the discriminator between a correct and an off-by-one-group walk.
    @test F[:flat_pxg_att_home][4] ≈ -1.0     # beta: scored 0, conceded 2
    @test F[:flat_pxg_def_home][4] ≈ 1.0
    @test sup[4] ≈ -2.0
    @test lev[4] ≈ 0.0

    # --- match 5: state is m1..m4, league baseline 10/8 = 1.25 ----------------------------
    # alpha for [2,3] ag [0,1] -> att +1.25, def -0.75.
    # gamma for [1,1] ag [1,3] -> att -0.25, def +0.75.
    @test F[:flat_pxg_att_home][5] ≈ 1.25
    @test F[:flat_pxg_def_home][5] ≈ -0.75
    @test F[:flat_pxg_att_away][5] ≈ -0.25
    @test F[:flat_pxg_def_away][5] ≈ 0.75
    @test sup[5] ≈ 3.0
    @test lev[5] ≈ 1.0

    # --- the destructive test: rewriting the future cannot move the past ------------------
    ds_rewritten = pxg_store(
        matches = pxg_goal_matches(m5_home_score = 9.0, m5_away_score = 0.0))
    G = pxg_extract(pxg_exact_config(), ds_rewritten)
    @test G[:flat_pxg_supremacy][1:5] == sup[1:5]
    @test G[:flat_pxg_level][1:5] == lev[1:5]

    # Every emitted column is finite and Float64 — the builder rejects anything else.
    for key in (:flat_pxg_supremacy, :flat_pxg_level, :flat_pxg_att_home,
                :flat_pxg_att_away, :flat_pxg_def_home, :flat_pxg_def_away)
        @test F[key] isa Vector{Float64}
        @test all(isfinite, F[key])
    end
end

@testset "PxGFeature windowing, shrinkage and neutral degradation" begin
    ds = pxg_store(matches = pxg_goal_matches())

    # A one-match window on match 5 sees only alpha's 3-1 and gamma's 1-3, not the earlier pair.
    win = pxg_extract(pxg_exact_config(lookback = 1), ds)
    @test win[:flat_pxg_att_home][5] ≈ 3.0 - 1.25
    @test win[:flat_pxg_def_home][5] ≈ 1.0 - 1.25

    # Exponential decay with a very short half-life converges on the same one-match answer.
    ewma = pxg_extract(
        pxg_exact_config(decay = :exponential, half_life_matches = 0.02), ds)
    @test ewma[:flat_pxg_att_home][5] ≈ win[:flat_pxg_att_home][5] atol = 1e-8

    # `min_matches` gates a side out entirely rather than trusting a one-match sample.
    gated = pxg_extract(pxg_exact_config(min_matches = 3), ds)
    @test all(iszero, gated[:flat_pxg_supremacy])
    @test all(iszero, gated[:flat_pxg_available])

    # Shrinkage pulls the deviation toward zero without changing its sign.
    shrunk = pxg_extract(pxg_exact_config(prior_weight = 2.0), ds)
    exact = pxg_extract(pxg_exact_config(), ds)
    @test 0.0 < shrunk[:flat_pxg_supremacy][3] < exact[:flat_pxg_supremacy][3]

    # `scale` divides the emitted columns and nothing else.
    scaled = pxg_extract(pxg_exact_config(scale = 4.0), ds)
    @test scaled[:flat_pxg_supremacy] ≈ exact[:flat_pxg_supremacy] ./ 4.0

    # An empty store degrades to a neutral column rather than erroring.
    blank = pxg_extract(pxg_exact_config(), pxg_store(); ordered_ids = Int[7, 8])
    @test blank[:flat_pxg_supremacy] == [0.0, 0.0]
    @test blank[:flat_pxg_form_fallback] == [1, 1]

    # An id absent from the store is neutral, not an error.
    partial = pxg_extract(pxg_exact_config(), ds; ordered_ids = Int[3, 999])
    @test partial[:flat_pxg_supremacy][1] ≈ 2.0
    @test partial[:flat_pxg_supremacy][2] == 0.0
end

# ==============================================================================
# 4. THE pxG COVARIATE
# ==============================================================================

@testset "PxGCovariate role dispatch and OOS bridge" begin
    ds = pxg_store(matches = pxg_goal_matches())
    F = pxg_extract(pxg_exact_config(), ds)
    fs = BayesianFootball.FeatureSet(F)

    supremacy = PxGCovariate(feature = pxg_exact_config())
    level = PxGCovariate(feature = pxg_exact_config(), role = LevelRole())

    @test covariate_name(supremacy) === :pxg
    @test covariate_role(supremacy) isa SupremacyRole
    @test covariate_role(level) isa LevelRole
    @test only(covariate_features(supremacy)) isa PxGFeature

    # The role IS the column: supremacy moves the result, level moves the total.
    @test covariate_column(supremacy, fs) ≈ F[:flat_pxg_supremacy]
    @test covariate_column(level, fs) ≈ F[:flat_pxg_level]
    @test covariate_sides(covariate_role(supremacy), 2.0) == (2.0, -2.0)
    @test covariate_sides(covariate_role(level), 2.0) == (2.0, 2.0)

    # The out-of-sample bridge is the same point-in-time walk, so an in-fold fixture reads back
    # identically through it and an unknown fixture is neutral.
    oos_df = DataFrame(match_id = [3, 5, 12_345])
    @test covariate_oos(supremacy, fs, oos_df) ≈ [2.0, 3.0, 0.0]
    @test covariate_oos(level, fs, oos_df) ≈ [0.0, 1.0, 0.0]

    # A materialised column wins, so a caller can price a hypothetical form state.
    @test covariate_oos(supremacy, fs,
                        DataFrame(match_id = [999], pxg_supremacy = [0.4])) == [0.4]
    @test covariate_oos(level, fs,
                        DataFrame(match_id = [999], pxg_level = [-0.4])) == [-0.4]

    # A feature set built without the matching role's column is an error naming the key.
    @test_throws ErrorException covariate_column(
        level, BayesianFootball.FeatureSet(Dict{Symbol,Any}(
            :flat_pxg_supremacy => [0.0])))
end

# ==============================================================================
# 5. THE RAPM AGGREGATION
# ==============================================================================

@testset "pxg_rapm_deltas: starting XI, shrinkage and gating" begin
    lineups = DataFrame(
        match_id = Int[1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
        player_id = Int[1, 2, 3, 4, 5, 6, 7, 1, 4, 5],
        team_side = ["home", "home", "home", "away", "away", "away", "home",
                     "home", "away", "away"],
        is_substitute = Bool[false, false, false, false, false, false, true,
                             false, false, false],
    )
    ratings = Dict{Int,Float64}(
        1 => 0.5, 2 => 0.3, 3 => 0.1, 4 => -0.2, 5 => 0.4, 6 => 0.0, 7 => 99.0)
    exposure = Dict{Int,Float64}(
        1 => 10.0, 2 => 30.0, 3 => 10.0, 4 => 10.0, 5 => 90.0, 6 => 5.0, 7 => 500.0)
    config = PxGRapmFeature(shrink_segments = 10.0, min_rated_per_side = 2)

    deltas = PXG_FEATURES.pxg_rapm_deltas(lineups, ratings, exposure, config)

    # r * n/(n + shrink): 0.5*0.5 + 0.3*0.75 + 0.1*0.5 = 0.525 against -0.2*0.5 + 0.4*0.9 = 0.26.
    # Player 6 is rated exactly 0.0 (RAPM's neutral) so it neither adds nor counts;
    # player 7 is a substitute and its enormous rating must not reach the sum.
    @test deltas[1].delta ≈ 0.525 - 0.26
    @test deltas[1].home_rated == 3
    @test deltas[1].away_rated == 2
    @test deltas[1].available == 1.0

    # Match 2 has one rated starter at home, below `min_rated_per_side`: neutral, not noisy.
    @test deltas[2].home_rated == 1
    @test deltas[2].delta == 0.0
    @test deltas[2].available == 0.0

    # A player the ridge never saw is shrunk to nothing but still counts as covered.
    no_exposure = PXG_FEATURES.pxg_rapm_deltas(
        lineups, ratings, Dict{Int,Float64}(), config)
    @test no_exposure[1].delta == 0.0
    @test no_exposure[1].home_rated == 3

    # Zero shrinkage recovers the raw sum of ratings.
    raw = PXG_FEATURES.pxg_rapm_deltas(
        lineups, ratings, exposure,
        PxGRapmFeature(shrink_segments = 0.0, min_rated_per_side = 2))
    @test raw[1].delta ≈ (0.5 + 0.3 + 0.1) - (-0.2 + 0.4)

    @test isempty(PXG_FEATURES.pxg_rapm_deltas(
        DataFrame(), ratings, exposure, config))
end

@testset "PxGRapmFeature scale is set by the permitted matches only" begin
    # 12 available fixtures with a known spread, half of them in the fit block. Fixture 100 is a
    # wild outlier OUTSIDE the fit block: if it moved the scale, the fold's own target season
    # would be setting the covariate's units.
    deltas = Dict{Int, NamedTuple{(:delta, :home_rated, :away_rated, :available),
                                  Tuple{Float64, Int, Int, Float64}}}()
    for i in 1:12
        deltas[i] = (delta = Float64(i), home_rated = 11, away_rated = 11, available = 1.0)
    end
    deltas[100] = (delta = 5_000.0, home_rated = 11, away_rated = 11, available = 1.0)
    # An unavailable fixture carries a 0.0 that is a neutral placeholder, not a measurement.
    deltas[101] = (delta = 0.0, home_rated = 0, away_rated = 0, available = 0.0)
    fit_ids = Set(1:12)

    auto = PXG_FEATURES._pxg_rapm_scale(deltas, fit_ids, PxGRapmFeature())
    @test auto ≈ std(1.0:12.0)

    # A fixed scale wins outright.
    @test PXG_FEATURES._pxg_rapm_scale(
        deltas, fit_ids, PxGRapmFeature(scale = 3.0)) == 3.0

    # Too few permitted observations to estimate a spread falls back to 1.0 rather than to a
    # near-zero divisor that would blow the covariate up.
    @test PXG_FEATURES._pxg_rapm_scale(
        deltas, Set([1, 2, 3]), PxGRapmFeature()) == 1.0
    constant = Dict(i => (delta = 2.0, home_rated = 11, away_rated = 11, available = 1.0)
                    for i in 1:12)
    @test PXG_FEATURES._pxg_rapm_scale(constant, fit_ids, PxGRapmFeature()) == 1.0
end

@testset "PxGRapmFeature leak control and neutral degradation" begin
    matches = pxg_goal_matches()
    ds = pxg_store(matches = matches)

    # No segments (no lineups, no incidents) is the every-non-Scottish-league case: zeros.
    F = pxg_extract(PxGRapmFeature(), ds; history = [1, 2])
    @test F[:flat_pxg_rapm] == zeros(5)
    @test F[:flat_pxg_rapm_available] == zeros(5)
    @test F[:flat_pxg_rapm_fallback] == ones(Int, 5)
    @test F[:pxg_rapm_by_match_id] == Dict{Int,Float64}()
    @test F[:pxg_rapm_scale] == 1.0
    # The fit set is recorded even on the degrade path, so a runner can audit what was permitted.
    @test F[:pxg_rapm_fit_match_ids] == [1, 2]

    # Gate 2: the default demands the frozen history block and refuses to guess without it.
    @test_throws ErrorException PXG_FEATURES.add_feature!(
        Dict{Symbol,Any}(), PxGRapmFeature(), Int[1, 2], Dict{String,Int}(), ds)

    # The research override sources its fit ids from the fold instead.
    G = Dict{Symbol,Any}()
    PXG_FEATURES.add_feature!(
        G, PxGRapmFeature(fit_on = :training), Int[3, 1], Dict{String,Int}(), ds)
    @test G[:pxg_rapm_fit_match_ids] == [1, 3]
end

@testset "PxGRapmCovariate contract" begin
    fs = BayesianFootball.FeatureSet(Dict{Symbol,Any}(
        :flat_pxg_rapm => [0.2, -0.4, 0.0],
        :pxg_rapm_by_match_id => Dict{Int,Float64}(1 => 0.2, 2 => -0.4, 3 => 0.0),
    ))
    covariate = PxGRapmCovariate()

    @test covariate_name(covariate) === :pxg_rapm
    @test covariate_role(covariate) isa SupremacyRole
    @test only(covariate_features(covariate)) isa PxGRapmFeature
    @test covariate_prior(covariate) isa UnivariateDistribution
    @test covariate_column(covariate, fs) == [0.2, -0.4, 0.0]
    @test covariate_oos(covariate, fs, DataFrame(match_id = [2, 77])) == [-0.4, 0.0]
    @test covariate_oos(covariate, fs,
                        DataFrame(match_id = [77], pxg_rapm = [1.5])) == [1.5]
    @test_throws ErrorException covariate_column(
        covariate, BayesianFootball.FeatureSet(Dict{Symbol,Any}()))
end

# ==============================================================================
# 6. BUILDER INTEGRATION AND THE ReverseDiff TAPE
# ==============================================================================

@testset "pxG covariates compose into a compiled ReverseDiff tape" begin
    builder = CountModelBuilder(:pxg_rapm_ad)
    add!(
        builder,
        PXG_PG.GlobalInterception(μ = Normal(0.2, 0.1)),
        PXG_PG.TimeDecayDynamics(
            days_half_life = 180.0,
            σ_att = Gamma(2.0, 0.15),
            σ_def = Gamma(2.0, 0.15),
        ),
        PXG_PG.GlobalHomeAdvantage(γ_global = Normal(0.2, 0.2)),
        PxGCovariate(),
        PxGRapmCovariate(),
        NoGuard(),
    )

    report = validate(builder)
    @test all(r -> r[2], report)

    model = build_count_model(builder)
    @test model isa PXG_PG.PoissonCountModel
    @test PXG_PG.cb_covariate_names(model) == (:pxg, :pxg_rapm)

    required = PXG_FEATURES.required_features(model)
    @test any(f -> f isa PxGFeature, required)
    @test any(f -> f isa PxGRapmFeature, required)

    # Both names must survive the export chain unqualified.
    @test PxGCovariate === BayesianFootball.Models.PxGCovariate
    @test PxGRapmCovariate === PXG_PG.PxGRapmCovariate
    @test PxGFeature === PXG_FEATURES.PxGFeature

    n = 8
    fs = pxg_model_feature_set(
        n = n,
        flat_pxg_supremacy = collect(range(-0.6, 0.6; length = n)),
        flat_pxg_level = collect(range(-0.3, 0.3; length = n)),
        flat_pxg_rapm = collect(range(0.4, -0.4; length = n)),
    )

    turing_model = PXG_PG.build_turing_model(model, fs)
    varinfo = DynamicPPL.VarInfo(turing_model)
    turing_model(varinfo)
    theta = copy(varinfo[:])

    density = DynamicPPL.LogDensityFunction(turing_model)
    objective = x -> LogDensityProblems.logdensity(density, x)

    raw_tape = ReverseDiff.GradientTape(objective, theta)
    compiled_tape = ReverseDiff.compile(raw_tape)
    gradient = similar(theta)
    ReverseDiff.gradient!(gradient, compiled_tape, theta)

    @test isfinite(objective(theta))
    @test all(isfinite, gradient)
    @test gradient ≈ ReverseDiff.gradient(objective, theta) rtol = 1e-8 atol = 1e-8

    # The two covariates cost two scalar sites and nothing else; a runtime loop over an abstract
    # covariate vector would show up here as a tape that grows with the design length.
    sites = PXG_PG.cb_varinfo_sites(model)
    @test Symbol("pxg.w") in sites
    @test Symbol("pxg_rapm.w") in sites

    # Recompiling against a longer design must not change the tape's shape.
    long_fs = pxg_model_feature_set(
        n = 32,
        flat_pxg_supremacy = collect(range(-0.6, 0.6; length = 32)),
        flat_pxg_level = collect(range(-0.3, 0.3; length = 32)),
        flat_pxg_rapm = collect(range(0.4, -0.4; length = 32)),
    )
    long_tape = ReverseDiff.GradientTape(
        let d = DynamicPPL.LogDensityFunction(
                PXG_PG.build_turing_model(model, long_fs))
            x -> LogDensityProblems.logdensity(d, x)
        end,
        theta)
    @test length(long_tape.tape) == length(raw_tape.tape)
end

@testset "The level role reaches the tape through its own column" begin
    builder = CountModelBuilder(:pxg_level_ad)
    add!(
        builder,
        PXG_PG.GlobalInterception(μ = Normal(0.2, 0.1)),
        PXG_PG.TimeDecayDynamics(days_half_life = 180.0),
        PXG_PG.GlobalHomeAdvantage(),
        PxGCovariate(role = LevelRole()),
        NegativeBinomialObservation(GlobalDispersion()),
        # The NegBin arm requires a clamped rate; `build_count_model` enforces it.
        ClampGuard(),
    )
    model = build_count_model(builder)
    @test model isa PXG_PG.NegBinCountModel

    n = 8
    # Only the level column is supplied: a level-role covariate that secretly read the supremacy
    # key would fail here rather than silently pricing the wrong quantity.
    fs = pxg_model_feature_set(n = n, flat_pxg_level = collect(range(-0.3, 0.3; length = n)))
    turing_model = PXG_PG.build_turing_model(model, fs)
    varinfo = DynamicPPL.VarInfo(turing_model)
    turing_model(varinfo)
    theta = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model)
    objective = x -> LogDensityProblems.logdensity(density, x)

    gradient = ReverseDiff.gradient(objective, theta)
    @test isfinite(objective(theta))
    @test all(isfinite, gradient)
end

@testset "The builder rejects a non-finite pxG design vector" begin
    builder = CountModelBuilder(:pxg_guard)
    add!(
        builder,
        PXG_PG.GlobalInterception(),
        PXG_PG.TimeDecayDynamics(days_half_life = 180.0),
        PXG_PG.GlobalHomeAdvantage(),
        PxGCovariate(),
        NoGuard(),
    )
    model = build_count_model(builder)
    bad = pxg_model_feature_set(n = 8, flat_pxg_supremacy = [0.0, NaN, zeros(6)...])
    @test_throws ErrorException PXG_PG.build_turing_model(model, bad)
end
