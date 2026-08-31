# test/test_joint_gamma_poisson.jl
#
# The two-arm (Gamma proxy xG + Poisson goals) joint observation, from the feature extractor up to a
# compiled ReverseDiff tape and back out through latent extraction.
#
# The four properties worth defending, in order of how expensive they are to get wrong:
#
#   1. THE ALGEBRA. `engine.jl` inlines the Gamma log-density by hand so the tape stays narrow. An
#      inlining slip yields a smooth, differentiable, perfectly wrong posterior that no convergence
#      diagnostic would flag. Checked against `logpdf(Gamma(ν, μ/ν), x)` via the independent
#      reference in `builder/equations.jl`, which is written from the distributions and never calls
#      the engine.
#   2. THE MASK IS EXACT. A match without a pxG measurement must contribute EXACTLY zero to the
#      proxy arm — not "approximately zero", not "a small penalty". Tested destructively: the dummy
#      value under a zero mask is rewritten and the log-density must come back bit-identical.
#   3. THE TWO ARMS ARE SEPARABLE. κ scales goals off the same latent the Gamma arm measures. With
#      κ pinned to 1 the goals arm must equal a plain Poisson model's; with the mask all-zero the
#      whole thing must equal a plain Poisson model shifted by log κ.
#   4. AD SAFETY. One compiled tape, finite gradients, and a tape length that does not grow with
#      the number of observations.

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
using SpecialFunctions

const JGP_PG = BayesianFootball.Models.PreGame
const JGP_API = JGP_PG.Builder
const JGP_FEATURES = BayesianFootball.Features

# ==============================================================================
# FIXTURES
# ==============================================================================

"A DataStore carrying only the domains the proxy-xG ladder reads."
function jgp_store(; matches = DataFrame(), bbc = DataFrame(), bbc_events = DataFrame())
    empty = DataFrame()
    return BayesianFootball.Data.DataStore(
        BayesianFootball.Data.ScottishLower(), matches,
        empty, empty, empty, empty, empty, bbc, bbc_events)
end

"""
Three fixtures with hand-computable pxG:

    m1  live text, 4 shots, all `the centre of the box` right-footed, 1 goal -> base rate 0.25 each
    m2  no live text but a match page: 10 v 6 attempts
    m3  no BBC coverage at all
"""
function jgp_ladder_store()
    matches = DataFrame(
        match_id = Int[1, 2, 3],
        tournament_id = fill(56, 3),
        season = fill("24/25", 3),
        match_date = Date[Date(2024, 1, 1), Date(2024, 1, 8), Date(2024, 1, 15)],
        start_timestamp = DateTime[DateTime(2024, 1, 1, 15), DateTime(2024, 1, 8, 15),
                                   DateTime(2024, 1, 15, 15)],
        home_team = ["alpha", "gamma", "alpha"],
        away_team = ["beta", "delta", "gamma"],
        home_score = Union{Missing,Float64}[2.0, 1.0, 3.0],
        away_score = Union{Missing,Float64}[0.0, 1.0, 1.0],
    )
    bbc_events = DataFrame(
        match_id = fill(1, 4),
        time = fill(10, 4),
        added_time = fill(0, 4),
        event_type = ["goal", "attempt_saved", "attempt_missed", "attempt_missed"],
        is_home_event = Union{Missing,Bool}[true, true, false, false],
        text = fill("Right footed shot from the centre of the box.", 4),
    )
    bbc = DataFrame(
        match_id = Int[2],
        shots_h = Union{Missing,Float64}[10.0],
        shots_a = Union{Missing,Float64}[6.0],
    )
    return jgp_store(matches = matches, bbc = bbc, bbc_events = bbc_events)
end

function jgp_extract(config, ds; ordered_ids = Int.(ds.matches.match_id))
    F = Dict{Symbol,Any}()
    JGP_FEATURES.add_feature!(F, config, ordered_ids, Dict{String,Int}(), ds)
    return F
end

"""
A synthetic FeatureSet the engine can be built on directly. `mask` selects which matches carry a
proxy measurement, so a fold with partial BBC coverage — the real 22/23-into-23/24 case — is the
default shape under test rather than a special case.
"""
function jgp_feature_set(n::Int = 6; mask = Float64[i > n ÷ 2 ? 1.0 : 0.0 for i in 1:n],
                         pxg_h = Float64[0.6 + 0.2i for i in 1:n],
                         pxg_a = Float64[1.4 - 0.1i for i in 1:n])
    return BayesianFootball.FeatureSet(Dict{Symbol,Any}(
        :flat_home_ids => Int[isodd(i) ? 1 : 2 for i in 1:n],
        :flat_away_ids => Int[isodd(i) ? 2 : 1 for i in 1:n],
        :season_indices => ones(Int, n),
        :time_indices => ones(Int, n),
        :flat_months => Int[mod1(i, 12) for i in 1:n],
        :flat_home_goals => Int[mod(i, 3) for i in 1:n],
        :flat_away_goals => Int[mod(i + 1, 3) for i in 1:n],
        :dates => collect(0:(n - 1)),
        :flat_pxg_home => pxg_h,
        :flat_pxg_away => pxg_a,
        :flat_pxg_obs_available => mask,
        :n_teams => 2,
        :n_seasons => 1,
        :n_rounds => 1,
        :team_map => Dict("home" => 1, "away" => 2),
    ))
end

jgp_base_components() = (
    JGP_PG.GlobalInterception(μ = Normal(0.2, 0.1)),
    JGP_PG.TimeDecayDynamics(days_half_life = 180.0,
                             σ_att = Gamma(2.0, 0.15),
                             σ_def = Gamma(2.0, 0.15)),
    JGP_PG.GlobalHomeAdvantage(γ_global = Normal(0.2, 0.2)),
)

jgp_joint_model(; covariates = (), observation = JointGammaPoissonObservation()) =
    build_count_model(:jgp_test, jgp_base_components()..., observation, covariates...)

function jgp_density(model, feature_set; seed::Int = 20260831)
    turing_model = JGP_PG.build_turing_model(model, feature_set)
    Random.seed!(seed)
    varinfo = DynamicPPL.VarInfo(turing_model)
    turing_model(varinfo)
    θ = copy(varinfo[:])
    density = DynamicPPL.LogDensityFunction(turing_model)
    f = x -> LogDensityProblems.logdensity(density, x)
    return (; turing_model, varinfo, θ, f)
end

# ==============================================================================
# 1. THE FEATURE: MatchProxyXGFeature
# ==============================================================================

@testset "MatchProxyXGFeature configuration" begin
    c = MatchProxyXGFeature()
    @test c.fallback === :none
    @test c.k == 25.0
    @test c.floor > 0.0
    @test c.dummy > 0.0

    ds = jgp_ladder_store()

    # The one rung that would double-count is refused BY NAME, with the reason.
    err = try
        jgp_extract(MatchProxyXGFeature(fallback = :goals), ds)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("double-count", sprint(showerror, err))

    for bad in (MatchProxyXGFeature(fallback = :xg),
                MatchProxyXGFeature(floor = 0.0),
                MatchProxyXGFeature(floor = -1.0),
                MatchProxyXGFeature(dummy = 0.0))
        @test_throws ErrorException jgp_extract(bad, ds)
    end
end

@testset "MatchProxyXGFeature measurement ladder and mask" begin
    ds = jgp_ladder_store()

    # --- commentary only: match 1 is the only covered fixture ---------------------------
    F = jgp_extract(MatchProxyXGFeature(), ds)
    @test F[:flat_pxg_obs_available] == [1.0, 0.0, 0.0]
    @test F[:flat_pxg_home][1] ≈ 0.5      # 2 shots at the 0.25 base rate
    @test F[:flat_pxg_away][1] ≈ 0.5
    @test F[:pxg_obs_source_counts][:commentary] == 1
    @test F[:pxg_obs_source_counts][:none] == 2

    # The uncovered fixtures carry the dummy, never a 0 and never a NaN: `log` is taken of every
    # entry regardless of mask, and the builder rejects a non-finite design vector outright.
    @test F[:flat_pxg_home][2:3] == [1.0, 1.0]
    @test all(isfinite, F[:flat_pxg_home])
    @test all(>(0.0), F[:flat_pxg_away])

    # --- the shots rung brings match 2 in ------------------------------------------------
    G = jgp_extract(MatchProxyXGFeature(fallback = :shots), ds)
    @test G[:flat_pxg_obs_available] == [1.0, 1.0, 0.0]
    @test G[:flat_pxg_home][2] ≈ 10.0 * 0.25
    @test G[:flat_pxg_away][2] ≈ 6.0 * 0.25
    @test G[:pxg_obs_source_counts][:shot_counts] == 1

    # Match 3 has goals (3-1) and is STILL uncovered — the goals rung is not reachable from here,
    # which is the property that keeps the two arms from reading the same numbers.
    @test G[:flat_pxg_obs_available][3] == 0.0

    # --- no BBC coverage at all degrades to an all-zero mask, never an error --------------
    bare = jgp_store(matches = ds.matches)
    H = jgp_extract(MatchProxyXGFeature(fallback = :shots), bare)
    @test H[:flat_pxg_obs_available] == zeros(3)
    @test all(isfinite, H[:flat_pxg_home])

    # --- key hygiene: the form covariate's own availability key is NOT touched -------------
    # `PxGFeature` owns `:flat_pxg_available` for a different quantity. Running both features into
    # one F_data must leave two independent answers.
    both = Dict{Symbol,Any}()
    JGP_FEATURES.add_feature!(both, PxGFeature(), Int[1, 2, 3], Dict{String,Int}(), ds)
    form_available = copy(both[:flat_pxg_available])
    JGP_FEATURES.add_feature!(both, MatchProxyXGFeature(), Int[1, 2, 3], Dict{String,Int}(), ds)
    @test both[:flat_pxg_available] == form_available
    @test both[:flat_pxg_obs_available] == [1.0, 0.0, 0.0]
    @test both[:flat_pxg_available] != both[:flat_pxg_obs_available]
end

# ==============================================================================
# 2. THE COMPONENT CONTRACT
# ==============================================================================

@testset "JointGammaPoissonObservation traits and assembly" begin
    o = JointGammaPoissonObservation()
    @test JGP_API.observation_family(o) === :poisson
    @test JGP_API.observation_wired(o)
    @test JGP_API.observation_prefixes(o) == [:obs]
    @test JGP_API.observation_features(o) == [o.feature]
    @test JGP_API.observation_features(PoissonObservation()) == []

    # Priors are the work package's, and ν is bounded away from a shape of zero.
    @test mean(o.log_kappa_prior) == 0.0
    @test std(o.log_kappa_prior) == 0.2
    @test minimum(o.shape_prior) == 0.5
    @test o.feature isa MatchProxyXGFeature

    model = jgp_joint_model()
    # The goals arm is Poisson, so the SCORE GRID is the double-Poisson grid. The Gamma arm is a
    # fit-time likelihood; it must not change which grid prices the markets.
    @test model isa PoissonCountModel
    @test BayesianFootball.latent_family(model) isa
          BayesianFootball.Models.Latents.PoissonCountFamily

    # The chain schema is derived, in declaration order.
    sites = cb_varinfo_sites(model)
    @test sites[end - 1] === Symbol("obs.ν")
    @test sites[end] === Symbol("obs.log_κ")
    @test cb_parameter_count(model, 2) == cb_parameter_count(jgp_joint_model(
        observation = PoissonObservation()), 2) + 2

    # The observation pulls its own feature into `required_features`, exactly as a covariate does.
    feats = BayesianFootball.Features.required_features(model)
    @test any(f -> f isa MatchProxyXGFeature, feats)
    @test !any(f -> f isa MatchProxyXGFeature,
               BayesianFootball.Features.required_features(
                   jgp_joint_model(observation = PoissonObservation())))
end

@testset "the referee refuses an unsafe joint configuration" begin
    base = jgp_base_components()

    # `exp(-η)` in the Gamma arm is unbounded below. NoGuard leaves it uncontrolled on a compiled
    # tape, where there is no branch left to catch the overflow.
    err = try
        build_count_model(:jgp_noguard, base..., JointGammaPoissonObservation(), NoGuard())
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("finite η floor", sprint(showerror, err))

    # A Gamma shape that can reach 0 is not a density.
    @test_throws ErrorException build_count_model(
        :jgp_bad_shape, base...,
        JointGammaPoissonObservation(shape_prior = Normal(4.0, 1.5)))

    # The proxy arm must be fed by the feature that emits its keys.
    @test_throws ErrorException build_count_model(
        :jgp_bad_feed, base...,
        JointGammaPoissonObservation(feature = PxGFeature()))

    # The valid model reports every rule as passing.
    b = CountModelBuilder(:jgp_ok)
    add!(b, base...)
    add!(b, JointGammaPoissonObservation())
    @test all(r -> r.pass, validate(b))
end

# ==============================================================================
# 3. THE DESIGN LAYER
# ==============================================================================

@testset "observation_design folds the mask into the weights once" begin
    model = jgp_joint_model()
    fs = jgp_feature_set(6)
    z = JGP_API.cb_design(model, fs)
    od = z.observation_data

    @test od isa JointGammaPoissonDesign
    @test od.n_observed == Int(sum(fs.data[:flat_pxg_obs_available]))
    @test od.log_pxg_h ≈ log.(fs.data[:flat_pxg_home])
    @test od.log_pxg_a ≈ log.(fs.data[:flat_pxg_away])
    @test od.mask_weights ≈ fs.data[:flat_pxg_obs_available] .* z.match_weights
    # A masked-out match has an EXACT zero weight, not a small one.
    @test all(iszero, od.mask_weights[fs.data[:flat_pxg_obs_available] .== 0.0])

    # Observations without a data channel dispatch the argument away entirely.
    @test JGP_API.cb_design(jgp_joint_model(observation = PoissonObservation()),
                            jgp_feature_set(6)).observation_data === nothing

    # --- the refusals ---------------------------------------------------------------------
    n = 6
    @test_throws ErrorException JGP_API.cb_design(
        model, jgp_feature_set(n; mask = fill(0.5, n)))             # partial mask
    @test_throws ErrorException JGP_API.cb_design(
        model, jgp_feature_set(n; pxg_h = vcat(0.0, fill(1.0, n - 1))))  # outside Gamma support
    @test_throws ErrorException JGP_API.cb_design(
        model, jgp_feature_set(n; pxg_a = vcat(NaN, fill(1.0, n - 1))))  # non-finite

    short = BayesianFootball.FeatureSet(
        merge(copy(jgp_feature_set(n).data), Dict{Symbol,Any}(:flat_pxg_home => ones(n - 1))))
    @test_throws ErrorException JGP_API.cb_design(model, short)

    missing_key = Dict{Symbol,Any}(jgp_feature_set(n).data)
    delete!(missing_key, :flat_pxg_obs_available)
    @test_throws ErrorException JGP_API.cb_design(
        model, BayesianFootball.FeatureSet(missing_key))
end

# ==============================================================================
# 4. THE ALGEBRA
# ==============================================================================

@testset "the inlined two-arm likelihood matches logpdf(Gamma) and logpdf(Poisson)" begin
    model = jgp_joint_model()
    fs = jgp_feature_set(8)
    small = jgp_density(model, fs)

    # The reference in equations.jl is written from the DISTRIBUTIONS and never calls the engine.
    data = JGP_API.cb_equation_data(model, fs)
    params = JGP_API.cb_params_from_varinfo(model, small.varinfo)
    @test isfinite(small.f(small.θ))
    @test small.f(small.θ) ≈ JGP_API.cb_logjoint(model, params, data) atol = 1e-9

    # Same at several other points, so the agreement is not a fluke of one draw.
    for δ in (0.01, -0.02, 0.05)
        point = small.θ .+ δ .* cos.(collect(eachindex(small.θ)))
        vi = DynamicPPL.unflatten(small.varinfo, point)
        p = JGP_API.cb_params_from_varinfo(model, vi)
        @test small.f(point) ≈ JGP_API.cb_logjoint(model, p, data) atol = 1e-8
    end

    # And with covariates attached, so the two-arm term composes with the covariate walk.
    with_cov = jgp_joint_model(covariates = (WealthCovariate(), DistanceCovariate()))
    cov_fs_data = Dict{Symbol,Any}(jgp_feature_set(8).data)
    cov_fs_data[:flat_delta_wealth_logsum] = collect(range(-0.4, 0.4; length = 8))
    cov_fs_data[:flat_distance] = collect(range(-1.0, 1.0; length = 8))
    cov_fs = BayesianFootball.FeatureSet(cov_fs_data)
    cov = jgp_density(with_cov, cov_fs; seed = 7)
    cov_data = JGP_API.cb_equation_data(with_cov, cov_fs)
    cov_params = JGP_API.cb_params_from_varinfo(with_cov, cov.varinfo)
    @test cov.f(cov.θ) ≈ JGP_API.cb_logjoint(with_cov, cov_params, cov_data) atol = 1e-9
end

@testset "the mask is exact, and the arms are separable" begin
    model = jgp_joint_model()
    n = 8
    mask = Float64[i > 4 ? 1.0 : 0.0 for i in 1:n]

    base_fs = jgp_feature_set(n; mask = mask)
    base = jgp_density(model, base_fs)

    # --- destructive: rewriting a MASKED-OUT observation cannot move the log-density ------
    rewritten_h = copy(base_fs.data[:flat_pxg_home])
    rewritten_h[1:4] .= 97.0
    rewritten = jgp_density(model, jgp_feature_set(n; mask = mask, pxg_h = rewritten_h))
    @test rewritten.f(base.θ) == base.f(base.θ)

    # --- rewriting a COVERED observation must move it ---------------------------------------
    moved_h = copy(base_fs.data[:flat_pxg_home])
    moved_h[5] += 0.7
    moved = jgp_density(model, jgp_feature_set(n; mask = mask, pxg_h = moved_h))
    @test moved.f(base.θ) != base.f(base.θ)

    # --- an all-zero mask collapses the proxy arm, leaving goals at λ = κ·μ ---------------
    # Compared at the LIKELIHOOD, not the log-joint: the joint model carries two extra priors,
    # so its log-joint is legitimately different even when the two likelihoods agree exactly.
    dark_fs = jgp_feature_set(n; mask = zeros(n))
    dark = jgp_density(model, dark_fs)
    joint_params = JGP_API.cb_params_from_varinfo(model, dark.varinfo)
    dark_data = JGP_API.cb_equation_data(model, dark_fs)

    plain_model = jgp_joint_model(observation = PoissonObservation())
    plain_data = JGP_API.cb_equation_data(plain_model, dark_fs)

    η_h = fill(0.3, n)
    η_a = fill(-0.1, n)
    joint_ll = JGP_API.cb_loglik(model.observation, joint_params, dark_data, η_h, η_a)
    # The same intensities reached through a plain Poisson: log λ = η + log κ.
    plain_ll = JGP_API.cb_loglik(PoissonObservation(), joint_params, plain_data,
                                 η_h .+ joint_params.log_κ, η_a .+ joint_params.log_κ)
    @test joint_ll ≈ plain_ll atol = 1e-9

    # And with the mask ON, the proxy arm must add something — otherwise the two-arm model
    # is a one-arm model wearing an extra parameter.
    lit_data = JGP_API.cb_equation_data(model, jgp_feature_set(n; mask = ones(n)))
    lit_ll = JGP_API.cb_loglik(model.observation, joint_params, lit_data, η_h, η_a)
    @test lit_ll != joint_ll
end

# ==============================================================================
# 5. AD SAFETY
# ==============================================================================

@testset "the joint arm compiles to one stable ReverseDiff tape" begin
    model = jgp_joint_model(covariates = (WealthCovariate(),))

    function cov_fs(n)
        d = Dict{Symbol,Any}(jgp_feature_set(n).data)
        d[:flat_delta_wealth_logsum] = collect(range(-0.4, 0.4; length = n))
        return BayesianFootball.FeatureSet(d)
    end

    small = jgp_density(model, cov_fs(6))
    large = jgp_density(model, cov_fs(60))

    raw_small = ReverseDiff.GradientTape(small.f, small.θ)
    raw_large = ReverseDiff.GradientTape(large.f, large.θ)

    # Fully vectorised: the tape shape is independent of the number of observations.
    @test length(raw_small.tape) == length(raw_large.tape)

    tape = ReverseDiff.compile(raw_small)
    compiled = similar(small.θ)
    ReverseDiff.gradient!(compiled, tape, small.θ)
    forward = ForwardDiff.gradient(small.f, small.θ)
    relerr(a, b) = norm(a - b) / max(norm(a), norm(b), 1.0)

    @test all(isfinite, compiled)
    @test relerr(compiled, ReverseDiff.gradient(small.f, small.θ)) <= 1e-8
    @test relerr(compiled, forward) <= 1e-6

    # The COMPILED tape must stay correct as the trajectory moves — this is where a value branch
    # inside `@model` would have baked one side of itself into the tape.
    for δ in (0.002, -0.004, 0.01)
        point = small.θ .+ δ .* sin.(collect(eachindex(small.θ)))
        replayed = similar(point)
        ReverseDiff.gradient!(replayed, tape, point)
        @test all(isfinite, replayed)
        @test relerr(replayed, ReverseDiff.gradient(small.f, point)) <= 1e-8
    end
end

# ==============================================================================
# 6. EXTRACTION
# ==============================================================================

@testset "extraction separates μ from λ = κ·μ" begin
    model = jgp_joint_model()
    n_draws = 4

    columns = cb_chain_columns(model, 2; n_seasons = 1)
    values = zeros(Float64, n_draws, length(columns), 1)
    for (j, name) in enumerate(columns)
        values[:, j, 1] .= name == "obs.ν" ? 4.0 :
                           name == "obs.log_κ" ? log(0.8) :
                           name == "dyn.σ_a" || name == "dyn.σ_d" ? 0.2 :
                           name == "inter.μ" ? 0.1 :
                           name == "ha.γ_global" ? 0.25 : 0.0
    end
    chain = MCMCChains.Chains(values, Symbol.(columns))

    df = DataFrame(
        match_id = Int[101],
        home_team = ["home"],
        away_team = ["away"],
        match_date = [Date(2025, 3, 1)],
        season_idx = Int[1],
    )
    fs = jgp_feature_set(6)
    out = JGP_PG.extract_parameters(model, df, fs, chain)
    rates = out[101]

    @test haskey(rates, :μ_h) && haskey(rates, :κ) && haskey(rates, :ν)
    @test rates.λ_h ≈ 0.8 .* rates.μ_h
    @test rates.λ_a ≈ 0.8 .* rates.μ_a
    @test all(rates.ν .== 4.0)
    # `true_xg_*` carries μ, NOT λ: μ is exactly what the Gamma arm measured, and κ is the league
    # finishing factor separating the two. This is the one observation where they differ.
    @test rates.true_xg_h ≈ rates.μ_h
    @test rates.true_xg_h != rates.λ_h

    # The Poisson score grid reads λ, and only λ.
    latents = BayesianFootball.extract_latents(model, chain, df, fs)
    @test latents isa CountLatents
    @test BayesianFootball.observation_family(latents) === :poisson
    @test vec(latents.λ_home) ≈ rates.λ_h
end
