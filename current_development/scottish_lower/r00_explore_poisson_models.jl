# current_development/scottish_lower/r00_explore_poisson_models.jl
#
# RUNNER: Single-Fold Visual Explorer for the Scottish Lower Poisson family.
#
# Fast REPL iteration on ONE walk-forward fold across four pure-Poisson engines:
#   Model 00  Baseline           η = μ + γ·home + α + β
#   Model 02  + Squad Wealth     η ± w_wealth · Δz_wealth
#   Model 03  + Travel Distance  η ± w_dist   · z_dist
#   Model 04  + Joint Wealth & Distance
#
# What it shows
#   1. The fold: what is fitted, what is held out, and the kickoff cut between them.
#   2. Posterior parameter tables (μ, γ, σ_a, σ_d, w_wealth, w_dist) with 90% CrI and R-hat.
#   3. Genuinely out-of-sample t+1 fixture predictions (λ_h, λ_a, 1X2, O2.5, BTTS) vs results.
#   4. A cross-model leaderboard on the same fold.
#
# FOLD CONTRACT (this is easy to get wrong — see _protocol/config.jl and _protocol/folds.jl):
#   `boundary.history_match_ids` + `boundary.target_match_ids` are ALL observations
#   through step t, and ALL of them are FITTED. The held-out fixtures are step t+1,
#   fetched separately with `Data.get_next_matches`. Kickoff filtration then drops any
#   nominally-prior match whose kickoff is not strictly before the earliest OOS kickoff
#   (56/57 biweeks are misaligned by one, and postponements cross the cut).

using BayesianFootball
using ThreadPinning; pinthreads(:cores)
using Turing
using DynamicPPL: to_submodel
using ReverseDiff          # AutoReverseDiff itself is re-exported by Turing
using Distributions
using DataFrames
using Dates
using Random
using Statistics
using SpecialFunctions
using Printf
using MCMCChains

const PG       = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Data     = BayesianFootball.Data


# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================

FOLD_INDEX     = 1        # which walk-forward fold of 24/25 to explore
CHAINS         = 4        # parallel MCMC chains
SAMPLES        = 400      # post-warmup draws per chain
WARMUP         = 400      # warmup draws per chain
ACCEPT_RATE    = 0.65
MAX_DEPTH      = 10
HALF_LIFE_DAYS = 180.0    # time-decay half-life for the match weights
MAX_GOALS      = 12       # score-matrix truncation
SEED           = 20260825


# ==============================================================================
# 2. DATASTORE AND FOLD CONSTRUCTION
# ==============================================================================

println("\n" * "="^96)
println(" 1. LOADING SCOTTISH LOWER DATASTORE AND CUTTING FOLD $(FOLD_INDEX)")
println("="^96)

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

splitter = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["24/25"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true,
)

boundaries = Data.create_id_boundaries(ds, splitter)
println("  Walk-forward folds available: $(length(boundaries))")
1 <= FOLD_INDEX <= length(boundaries) ||
    error("FOLD_INDEX must be in 1:$(length(boundaries)), got $(FOLD_INDEX)")

boundary, meta = boundaries[FOLD_INDEX]

# --- The genuinely held-out slate: step t+1 -----------------------------------
oos_df = DataFrame(Data.get_next_matches(ds, (boundary, meta), splitter))
nrow(oos_df) > 0 || error("fold $(FOLD_INDEX) has no t+1 fixtures")

# --- Kickoff filtration -------------------------------------------------------
kickoff = Dict{Int, DateTime}(
    Int(r.match_id) => DateTime(r.match_date) + Hour(r.match_hour)
    for r in eachrow(ds.matches) if !ismissing(r.match_date) && !ismissing(r.match_hour)
)

oos_kickoffs = [get(kickoff, Int(id), nothing) for id in oos_df.match_id]
any(isnothing, oos_kickoffs) && error("fold $(FOLD_INDEX) has OOS fixtures without a full kickoff")
cutoff = minimum(something.(oos_kickoffs))

fitted_before_cut(id) = haskey(kickoff, Int(id)) && kickoff[Int(id)] < cutoff
history_keep = Int[id for id in boundary.history_match_ids if fitted_before_cut(id)]
target_keep  = Int[id for id in boundary.target_match_ids  if fitted_before_cut(id)]
fitted_ids   = vcat(history_keep, target_keep)
dropped_ids  = setdiff(Int.(vcat(boundary.history_match_ids, boundary.target_match_ids)), fitted_ids)

# The boundary handed to the feature builder contains ONLY fitted matches, so every
# flat_* vector it produces is training data. Nothing from the t+1 slate enters it.
fold_boundary = Data.SplitBoundary(boundary.fold_id, boundary.target_step, history_keep, target_keep)

last_fitted = maximum(kickoff[id] for id in fitted_ids)
@printf("  Fold %d  |  season %s  step %d\n", FOLD_INDEX, meta.target_season, meta.time_step)
@printf("  Fitted matches      : %4d  (history %d + target %d)\n",
        length(fitted_ids), length(history_keep), length(target_keep))
@printf("  Dropped by kickoff  : %4d\n", length(dropped_ids))
@printf("  Held-out t+1 slate  : %4d\n", nrow(oos_df))
@printf("  Last fitted kickoff : %s\n", Dates.format(last_fitted, "yyyy-mm-dd HH:MM"))
@printf("  First OOS kickoff   : %s\n", Dates.format(cutoff, "yyyy-mm-dd HH:MM"))

# Season index of the OOS slate inside this fold's own season vocabulary.
fold_matches = subset(ds.matches, :match_id => ByRow(id -> Int(id) in Set(fitted_ids)))
fold_seasons = sort(unique(fold_matches.season))
fold_season_map = Dict(s => i for (i, s) in enumerate(fold_seasons))


# ==============================================================================
# 3. THE FOUR MODEL CONFIGURATIONS
# ==============================================================================

abstract type AbstractExplorePoissonModel <: BayesianFootball.TypesInterfaces.AbstractPoissonModel end

_base_interception() = PG.GlobalInterception(μ = Normal(0.2, 0.1))
_base_home_adv()     = PG.GlobalHomeAdvantage(γ_global = Normal(0.2, 0.2))
_base_dynamics()     = PG.TimeDecayDynamics(days_half_life = HALF_LIFE_DAYS,
                                            σ_att = Gamma(2.0, 0.15),
                                            σ_def = Gamma(2.0, 0.15))

Base.@kwdef struct ExploreModel00{I,T,H} <: AbstractExplorePoissonModel
    interception_config::I  = _base_interception()
    dynamics_config::T      = _base_dynamics()
    homeadvantage_config::H = _base_home_adv()
end

Base.@kwdef struct ExploreModel02{I,T,H,W} <: AbstractExplorePoissonModel
    interception_config::I  = _base_interception()
    dynamics_config::T      = _base_dynamics()
    homeadvantage_config::H = _base_home_adv()
    wealth_feature::W       = Features.SquadWealthFeature(log_scale = 0.50)
    w_wealth_prior::Distribution = truncated(Normal(0.10, 0.05), lower = 0.0)
end

Base.@kwdef struct ExploreModel03{I,T,H,D} <: AbstractExplorePoissonModel
    interception_config::I  = _base_interception()
    dynamics_config::T      = _base_dynamics()
    homeadvantage_config::H = _base_home_adv()
    distance_feature::D     = Features.DistanceFeature(metric = :log_dist_z)
    w_dist_prior::Distribution = truncated(Normal(0.04, 0.03), lower = 0.0)
end

Base.@kwdef struct ExploreModel04{I,T,H,W,D} <: AbstractExplorePoissonModel
    interception_config::I  = _base_interception()
    dynamics_config::T      = _base_dynamics()
    homeadvantage_config::H = _base_home_adv()
    wealth_feature::W       = Features.SquadWealthFeature(log_scale = 0.50)
    distance_feature::D     = Features.DistanceFeature(metric = :log_dist_z)
    w_wealth_prior::Distribution = truncated(Normal(0.10, 0.05), lower = 0.0)
    w_dist_prior::Distribution   = truncated(Normal(0.04, 0.03), lower = 0.0)
end

uses_wealth(::AbstractExplorePoissonModel)              = false
uses_wealth(::Union{ExploreModel02, ExploreModel04})    = true
uses_distance(::AbstractExplorePoissonModel)            = false
uses_distance(::Union{ExploreModel03, ExploreModel04})  = true

function Features.required_features(m::AbstractExplorePoissonModel)
    req = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.TimeIndicesFeature(),
    ]
    uses_wealth(m)   && push!(req, m.wealth_feature)
    uses_distance(m) && push!(req, m.distance_feature)
    return req
end


# ==============================================================================
# 4. TURING ENGINES
# ==============================================================================
#
# One engine per active-covariate combination. A single engine with a runtime
# branch would either register prior-only sites the model does not use or place
# control flow inside the @model block; both are avoided here.
#
# The likelihood is evaluated directly in log-intensity space,
#   log p(y | η) = y·η − exp(η) − log Γ(y+1),
# with the log-factorial precomputed outside the model.

@model function engine_base(h, a, s, mo, yh, ya, wt, lfh, lfa, nt, ns, m)
    inter ~ to_submodel(PG.build_interception(m.interception_config, ns, 12))
    ha    ~ to_submodel(PG.build_home_advantage(m.homeadvantage_config, nt))
    dyn   ~ to_submodel(PG.build_dynamics(m.dynamics_config, nt))

    b  = view(inter.μ_base, s) .+ view(inter.δ_month, mo)
    eh = clamp.(b .+ view(ha, h) .+ view(dyn.α, h) .+ view(dyn.β, a), -10.0, 10.0)
    ea = clamp.(b             .+ view(dyn.α, a) .+ view(dyn.β, h), -10.0, 10.0)

    Turing.@addlogprob! sum(wt .* (yh .* eh .- exp.(eh) .- lfh)) +
                        sum(wt .* (ya .* ea .- exp.(ea) .- lfa))
end

@model function engine_wealth(h, a, s, mo, yh, ya, wt, xw, lfh, lfa, nt, ns, m)
    inter ~ to_submodel(PG.build_interception(m.interception_config, ns, 12))
    ha    ~ to_submodel(PG.build_home_advantage(m.homeadvantage_config, nt))
    dyn   ~ to_submodel(PG.build_dynamics(m.dynamics_config, nt))
    w_wealth ~ m.w_wealth_prior

    b  = view(inter.μ_base, s) .+ view(inter.δ_month, mo)
    q  = w_wealth .* xw
    eh = clamp.(b .+ view(ha, h) .+ view(dyn.α, h) .+ view(dyn.β, a) .+ q, -10.0, 10.0)
    ea = clamp.(b             .+ view(dyn.α, a) .+ view(dyn.β, h) .- q, -10.0, 10.0)

    Turing.@addlogprob! sum(wt .* (yh .* eh .- exp.(eh) .- lfh)) +
                        sum(wt .* (ya .* ea .- exp.(ea) .- lfa))
end

@model function engine_distance(h, a, s, mo, yh, ya, wt, xd, lfh, lfa, nt, ns, m)
    inter ~ to_submodel(PG.build_interception(m.interception_config, ns, 12))
    ha    ~ to_submodel(PG.build_home_advantage(m.homeadvantage_config, nt))
    dyn   ~ to_submodel(PG.build_dynamics(m.dynamics_config, nt))
    w_dist ~ m.w_dist_prior

    b  = view(inter.μ_base, s) .+ view(inter.δ_month, mo)
    q  = w_dist .* xd
    eh = clamp.(b .+ view(ha, h) .+ view(dyn.α, h) .+ view(dyn.β, a) .+ q, -10.0, 10.0)
    ea = clamp.(b             .+ view(dyn.α, a) .+ view(dyn.β, h) .- q, -10.0, 10.0)

    Turing.@addlogprob! sum(wt .* (yh .* eh .- exp.(eh) .- lfh)) +
                        sum(wt .* (ya .* ea .- exp.(ea) .- lfa))
end

@model function engine_joint(h, a, s, mo, yh, ya, wt, xw, xd, lfh, lfa, nt, ns, m)
    inter ~ to_submodel(PG.build_interception(m.interception_config, ns, 12))
    ha    ~ to_submodel(PG.build_home_advantage(m.homeadvantage_config, nt))
    dyn   ~ to_submodel(PG.build_dynamics(m.dynamics_config, nt))
    w_wealth ~ m.w_wealth_prior
    w_dist   ~ m.w_dist_prior

    b  = view(inter.μ_base, s) .+ view(inter.δ_month, mo)
    q  = w_wealth .* xw .+ w_dist .* xd
    eh = clamp.(b .+ view(ha, h) .+ view(dyn.α, h) .+ view(dyn.β, a) .+ q, -10.0, 10.0)
    ea = clamp.(b             .+ view(dyn.α, a) .+ view(dyn.β, h) .- q, -10.0, 10.0)

    Turing.@addlogprob! sum(wt .* (yh .* eh .- exp.(eh) .- lfh)) +
                        sum(wt .* (ya .* ea .- exp.(ea) .- lfa))
end

"Pull the fitted-side vectors out of a FeatureSet and assert they are model-safe."
function explore_training_data(m::AbstractExplorePoissonModel, fs::Features.FeatureSet)
    d  = fs.data
    n  = length(d[:flat_home_ids])
    h  = Vector{Int}(d[:flat_home_ids])
    a  = Vector{Int}(d[:flat_away_ids])
    s  = Vector{Int}(d[:season_indices])
    mo = Vector{Int}(d[:flat_months])
    yh = Vector{Int}(d[:flat_home_goals])
    ya = Vector{Int}(d[:flat_away_goals])
    wt = 0.5 .^ (Vector{Float64}(d[:dates]) ./ m.dynamics_config.days_half_life)
    xw = uses_wealth(m)   ? Vector{Float64}(d[:flat_delta_wealth]) : Float64[]
    xd = uses_distance(m) ? Vector{Float64}(d[:flat_distance])     : Float64[]

    all(v -> length(v) == n, (h, a, s, mo, yh, ya, wt)) || error("ragged feature vectors")
    all(v -> isempty(v) || length(v) == n, (xw, xd))    || error("ragged covariate vectors")
    all(isfinite, wt) && all(isfinite, xw) && all(isfinite, xd) || error("non-finite covariate")

    return (; n, h, a, s, mo, yh, ya, wt, xw, xd,
            lfh = loggamma.(Float64.(yh) .+ 1.0),
            lfa = loggamma.(Float64.(ya) .+ 1.0),
            nt = Int(d[:n_teams]), ns = Int(d[:n_seasons]))
end

function build_explore_turing(m::AbstractExplorePoissonModel, fs::Features.FeatureSet)
    z = explore_training_data(m, fs)
    if uses_wealth(m) && uses_distance(m)
        return engine_joint(z.h, z.a, z.s, z.mo, z.yh, z.ya, z.wt, z.xw, z.xd, z.lfh, z.lfa, z.nt, z.ns, m)
    elseif uses_wealth(m)
        return engine_wealth(z.h, z.a, z.s, z.mo, z.yh, z.ya, z.wt, z.xw, z.lfh, z.lfa, z.nt, z.ns, m)
    elseif uses_distance(m)
        return engine_distance(z.h, z.a, z.s, z.mo, z.yh, z.ya, z.wt, z.xd, z.lfh, z.lfa, z.nt, z.ns, m)
    else
        return engine_base(z.h, z.a, z.s, z.mo, z.yh, z.ya, z.wt, z.lfh, z.lfa, z.nt, z.ns, m)
    end
end


# ==============================================================================
# 5. OUT-OF-SAMPLE COVARIATES, INTENSITIES AND PRICES
# ==============================================================================

"""
Point-in-time wealth differential for the held-out slate.

`_build_match_wealth_records` carries each team's last observed starting-XI value
forward with a decay towards a population baseline, so the OOS rows must be built in
the SAME chronological pass as the fitted rows. The baseline is still anchored on the
fitted matches only, and `log_scale` is fixed by config, so fitted values are
bit-identical to the ones the likelihood saw.
"""
function oos_wealth_covariate(m, fitted_ids::Vector{Int}, target_df::DataFrame)
    ordered = vcat(fitted_ids, Int.(target_df.match_id))
    records = Features._build_match_wealth_records(
        ds.lineups, ds.matches, ordered, fitted_ids, m.wealth_feature)
    delta = Float64[get(records, Int32(id), (delta = 0.0,)).delta for id in target_df.match_id]
    avail = Float64[get(records, Int32(id), (available = 0.0,)).available for id in target_df.match_id]
    return delta, avail
end

"Travel distance for the held-out slate, from the same fixed stadium catalog."
function oos_distance_covariate(m, target_df::DataFrame)
    catalog = Features.load_stadium_catalog(m.distance_feature.geocodes_csv)
    table = Features.build_match_distance_table(target_df; geocodes_df = catalog)
    metric = m.distance_feature.metric
    return Vector{Float64}(table[!, metric]), Float64.(table.distance_fallback)
end

"Posterior-averaged 1X2 / totals / BTTS prices from a pair of intensity draws."
function poisson_market_prices(λ_h::Vector{Float64}, λ_a::Vector{Float64}; max_goals::Int = MAX_GOALS)
    grid = 0:max_goals
    p_hw = p_dr = p_aw = p_o25 = p_btts = 0.0
    mass = 0.0
    for k in eachindex(λ_h)
        ph = pdf.(Poisson(λ_h[k]), grid)
        pa = pdf.(Poisson(λ_a[k]), grid)
        for gh in grid, ga in grid
            p = ph[gh + 1] * pa[ga + 1]
            mass += p
            gh > ga  ? (p_hw += p) : gh == ga ? (p_dr += p) : (p_aw += p)
            gh + ga > 2      && (p_o25  += p)
            gh > 0 && ga > 0 && (p_btts += p)
        end
    end
    # Renormalise by the retained grid mass so the truncation cannot leak probability.
    return (home = p_hw / mass, draw = p_dr / mass, away = p_aw / mass,
            over25 = p_o25 / mass, btts = p_btts / mass)
end

"""
OOS intensities and prices for every t+1 fixture.

Team effects come from the component extractors, which reapply the zero-sum
centering that `build_dynamics` performs inside the model — reconstructing them by
hand as `raw .* σ` would leave the intercept and every intensity biased.
"""
function explore_oos_predictions(m::AbstractExplorePoissonModel, fs::Features.FeatureSet,
                                 chain::Chains, target_df::DataFrame, fitted_ids::Vector{Int})
    d         = fs.data
    team_map  = d[:team_map]
    n_teams   = Int(d[:n_teams])
    n_seasons = Int(d[:n_seasons])
    n_samp    = size(chain, 1) * size(chain, 3)

    inter = PG.extract_interception(chain, m.interception_config, n_seasons)
    ha    = PG.extract_home_advantage(chain, m.homeadvantage_config, n_teams)
    dyn   = PG.extract_dynamics(chain, m.dynamics_config, "dyn", n_teams)

    ww = uses_wealth(m)   ? vec(Array(chain[:w_wealth])) : zeros(n_samp)
    wd = uses_distance(m) ? vec(Array(chain[:w_dist]))   : zeros(n_samp)

    xw, wealth_avail = uses_wealth(m) ?
        oos_wealth_covariate(m, fitted_ids, target_df) :
        (zeros(nrow(target_df)), zeros(nrow(target_df)))
    xd, dist_fallback = uses_distance(m) ?
        oos_distance_covariate(m, target_df) :
        (zeros(nrow(target_df)), zeros(nrow(target_df)))
    all(isfinite, xw) && all(isfinite, xd) || error("non-finite OOS covariate")

    rows = NamedTuple[]
    for (i, r) in enumerate(eachrow(target_df))
        h_idx = get(team_map, String(r.home_team), 0)
        a_idx = get(team_map, String(r.away_team), 0)

        # An unseen club falls back to the zero-sum population effect.
        α_h = h_idx > 0 ? dyn.α[:, h_idx] : zeros(n_samp)
        β_h = h_idx > 0 ? dyn.β[:, h_idx] : zeros(n_samp)
        α_a = a_idx > 0 ? dyn.α[:, a_idx] : zeros(n_samp)
        β_a = a_idx > 0 ? dyn.β[:, a_idx] : zeros(n_samp)
        γ   = m.homeadvantage_config isa PG.GlobalHomeAdvantage ? ha[:, 1] :
              (h_idx > 0 ? ha[:, h_idx] : zeros(n_samp))

        s_idx = get(fold_season_map, r.season, n_seasons)
        base  = inter.μ_base[:, s_idx] .+ inter.δ_month[:, Dates.month(r.match_date)]

        q  = ww .* xw[i] .+ wd .* xd[i]
        λh = exp.(clamp.(base .+ γ .+ α_h .+ β_a .+ q, -10.0, 10.0))
        λa = exp.(clamp.(base     .+ α_a .+ β_h .- q, -10.0, 10.0))

        px = poisson_market_prices(λh, λa)

        push!(rows, (
            match_id       = Int(r.match_id),
            date           = r.match_date,
            home_team      = String(r.home_team),
            away_team      = String(r.away_team),
            actual_hg      = ismissing(r.home_score) ? -1 : Int(r.home_score),
            actual_ag      = ismissing(r.away_score) ? -1 : Int(r.away_score),
            wealth_dz      = xw[i],
            wealth_avail   = wealth_avail[i],
            dist_z         = xd[i],
            dist_fallback  = dist_fallback[i],
            mean_λ_h       = mean(λh),
            mean_λ_a       = mean(λa),
            prob_home      = px.home,
            prob_draw      = px.draw,
            prob_away      = px.away,
            prob_over25    = px.over25,
            prob_btts      = px.btts,
        ))
    end
    return DataFrame(rows)
end

"Mean negative log score of the three-way market on graded fixtures."
function log_loss_1x2(preds::DataFrame)
    graded = filter(r -> r.actual_hg >= 0 && r.actual_ag >= 0, preds)
    nrow(graded) == 0 && return NaN
    return mean(
        -log(max(1e-12, r.actual_hg > r.actual_ag ? r.prob_home :
                        r.actual_hg == r.actual_ag ? r.prob_draw : r.prob_away))
        for r in eachrow(graded))
end

function log_loss_binary(preds::DataFrame, prob_col::Symbol, hit)
    graded = filter(r -> r.actual_hg >= 0 && r.actual_ag >= 0, preds)
    nrow(graded) == 0 && return NaN
    return mean(
        -log(max(1e-12, hit(r) ? r[prob_col] : 1.0 - r[prob_col]))
        for r in eachrow(graded))
end


# ==============================================================================
# 6. EXECUTION
# ==============================================================================

models_to_test = [
    ("00  Pure Poisson control",       ExploreModel00()),
    ("02  + Squad Wealth (Δz)",        ExploreModel02()),
    ("03  + Travel Distance (z)",      ExploreModel03()),
    ("04  + Joint Wealth & Distance",  ExploreModel04()),
]

println("\n" * "="^96)
println(" 2. SINGLE-FOLD EXPLORATION  (fold $(FOLD_INDEX), $(CHAINS) chains x $(SAMPLES) draws, $(WARMUP) warmup)")
println("="^96)

leaderboard = NamedTuple[]
predictions_by_model = Dict{String, DataFrame}()

for (name, m) in models_to_test
    println("\n" * "-"^96)
    println(" >>> $name")
    println("-"^96)

    # --- 6.1 Features -------------------------------------------------------
    fs = Features.create_features(fold_boundary, ds, m, splitter.dynamics_col)
    z  = explore_training_data(m, fs)
    @printf("  [features] %d fitted matches, %d teams, %d seasons | %s\n",
            z.n, z.nt, z.ns,
            join([string(typeof(f).name.name) for f in Features.required_features(m)], ", "))
    if uses_wealth(m)
        @printf("             wealth Δz: mean %+.3f  sd %.3f  |  observed both sides on %.0f%% of fitted matches\n",
                mean(z.xw), std(z.xw), 100 * mean(fs.data[:flat_wealth_available] .>= 1.0))
    end
    if uses_distance(m)
        @printf("             distance z: mean %+.3f  sd %.3f  |  catalog fallback on %.0f%% of fitted matches\n",
                mean(z.xd), std(z.xd), 100 * mean(fs.data[:flat_distance_fallback] .== 1))
    end

    # --- 6.2 Sampling -------------------------------------------------------
    turing_model = build_explore_turing(m, fs)
    Random.seed!(SEED)
    t_start = time()
    chain = sample(
        turing_model,
        NUTS(WARMUP, ACCEPT_RATE; max_depth = MAX_DEPTH),
        MCMCThreads(), SAMPLES, CHAINS;
        progress = false,
        adtype   = AutoReverseDiff(compile = true),
    )
    elapsed = time() - t_start

    stats    = DataFrame(summarystats(chain))
    finite_col(col) = Float64[Float64(x) for x in col if !ismissing(x) && isfinite(x)]
    rhats    = finite_col(stats.rhat)
    # MCMCChains renamed :ess to :ess_bulk; accept either.
    esss     = finite_col(hasproperty(stats, :ess_bulk) ? stats.ess_bulk : stats.ess)
    rhat_max = isempty(rhats) ? NaN : maximum(rhats)
    ess_min  = isempty(esss)  ? NaN : minimum(esss)
    # `numerical_error` is a NUTS internal; guard so a sampler change cannot break the runner.
    divergent = :numerical_error in names(chain, :internals) ?
                Int(sum(Array(chain[:numerical_error]))) : -1

    @printf("  [sampled in %.1fs] max R-hat %.3f (target < 1.05) | min ESS %.0f | divergences %d\n",
            elapsed, rhat_max, ess_min, divergent)

    # --- 6.3 Posterior parameter table --------------------------------------
    function param_row(label, site, marker = "")
        v = vec(Array(chain[site]))
        @printf("  %-27s | %+7.3f | [%+7.3f, %+7.3f] | %6.3f %s\n",
                label, mean(v), quantile(v, 0.05), quantile(v, 0.95), std(v), marker)
    end

    println("  " * "-"^80)
    @printf("  %-27s | %7s | %-20s | %6s\n", "Latent parameter", "Mean", "  90% credible int.", "SD")
    println("  " * "-"^80)
    param_row("Baseline intercept (μ)",   Symbol("inter.μ"))
    param_row("Home advantage (γ)",       Symbol("ha.γ_global"))
    param_row("Attack volatility (σ_a)",  Symbol("dyn.σ_a"))
    param_row("Defence volatility (σ_d)", Symbol("dyn.σ_d"))
    uses_wealth(m)   && param_row("Squad wealth (w_wealth)", :w_wealth, "<-- feature")
    uses_distance(m) && param_row("Travel distance (w_dist)", :w_dist,  "<-- feature")
    println("  " * "-"^80)

    # --- 6.4 Held-out slate --------------------------------------------------
    preds = explore_oos_predictions(m, fs, chain, oos_df, fitted_ids)
    predictions_by_model[name] = preds

    println("\n  [out-of-sample t+1 slate]")
    println("  " * "-"^104)
    @printf("  %-20s %-20s | %6s %6s | %5s %5s | %6s %6s %6s | %6s %6s | %s\n",
            "Home", "Away", "Δz_w", "z_dst", "λ_h", "λ_a", "Home%", "Draw%", "Away%", "O2.5%", "BTTS%", "Score")
    println("  " * "-"^104)
    for r in eachrow(preds)
        score = r.actual_hg >= 0 ? @sprintf("%d-%d", r.actual_hg, r.actual_ag) : " - "
        @printf("  %-20s %-20s | %+6.2f %+6.2f | %5.2f %5.2f | %5.1f%% %5.1f%% %5.1f%% | %5.1f%% %5.1f%% | %s\n",
                first(r.home_team, 20), first(r.away_team, 20), r.wealth_dz, r.dist_z,
                r.mean_λ_h, r.mean_λ_a,
                100r.prob_home, 100r.prob_draw, 100r.prob_away,
                100r.prob_over25, 100r.prob_btts, score)
    end
    println("  " * "-"^104)

    ll_1x2  = log_loss_1x2(preds)
    ll_o25  = log_loss_binary(preds, :prob_over25, r -> r.actual_hg + r.actual_ag > 2)
    ll_btts = log_loss_binary(preds, :prob_btts,   r -> r.actual_hg > 0 && r.actual_ag > 0)
    @printf("  log-loss  1X2 %.4f | O2.5 %.4f | BTTS %.4f   (%d graded fixtures)\n",
            ll_1x2, ll_o25, ll_btts,
            nrow(filter(r -> r.actual_hg >= 0, preds)))

    push!(leaderboard, (
        name     = name,
        time_s   = elapsed,
        rhat     = rhat_max,
        ess      = ess_min,
        div      = divergent,
        μ        = mean(chain[Symbol("inter.μ")]),
        γ        = mean(chain[Symbol("ha.γ_global")]),
        w_wealth = uses_wealth(m)   ? mean(chain[:w_wealth]) : NaN,
        w_dist   = uses_distance(m) ? mean(chain[:w_dist])   : NaN,
        ll_1x2   = ll_1x2,
        ll_o25   = ll_o25,
        ll_btts  = ll_btts,
    ))
end


# ==============================================================================
# 7. LEADERBOARD
# ==============================================================================

fmt(x) = isnan(x) ? "     — " : @sprintf("%+7.3f", x)

println("\n" * "="^96)
println(" 3. FOLD $(FOLD_INDEX) LEADERBOARD   ($(nrow(oos_df)) held-out fixtures)")
println("="^96)
@printf("  %-32s | %6s | %5s | %6s | %4s | %7s | %7s | %7s | %7s | %7s\n",
        "Model", "Time", "R-hat", "ESS", "div", "γ", "w_wealth", "w_dist", "LL 1X2", "LL O2.5")
println("  " * "-"^116)
for r in leaderboard
    @printf("  %-32s | %5.1fs | %5.3f | %6.0f | %4d | %s | %s | %s | %7.4f | %7.4f\n",
            r.name, r.time_s, r.rhat, r.ess, r.div,
            fmt(r.γ), fmt(r.w_wealth), fmt(r.w_dist), r.ll_1x2, r.ll_o25)
end
println("  " * "-"^116)
println("  Lower log-loss is better. One fold of $(nrow(oos_df)) fixtures is a smoke test, not evidence:")
println("  a per-fold log-loss difference of this size is well inside sampling noise.")
println("="^96)
