# current_development/bbc_xg_proxy/l03_funnel_cascade.jl
#
# LOADER (temporary module). THINNED-POISSON FUNNEL CASCADE for ScottishLower (56/57).
#
#     Shots ~ Poisson(λ_s)
#     SoT   | Shots ~ Binomial(Shots, p₁)
#     Goals | SoT   ~ Binomial(SoT,   p₂)
#
# By Poisson thinning the goals marginal stays Poisson(λ_s·p₁·p₂), so EVERY downstream pricing
# object is unchanged (Poisson score grid, market probs, eval-vs-close, Kelly) — the funnel only
# enriches the OBSERVATION model with ~7× the count volume of goals (shots ≈ 10 vs goals ≈ 1.4).
#
# Stage 1 = GLOBAL p₁/p₂ (this file). Stage 2 (l04) makes them hierarchical per team.
#
# THE STAGE-1 CAVEAT (drives the diagnostics in r03): with global p₁/p₂,
# λ_goals = λ_s(α_i, β_j)·p₁·p₂ — all team-strength variation in goals must come from SHOT
# VOLUME, so goal-rate spread is forced to equal shot-rate spread (which is proportionally
# tighter). Plus the shots term outweighs goals ~7:1 in the log-likelihood. `funnel_weight`
# tempers the shots+SoT terms only so a later grid can separate "shots swamped goals" from
# "conversion is genuinely heterogeneous" (the latter is what l04 fixes).
#
# Skeleton: current_development/scottish_lower_smile/l01_team_dp_league.jl §1 (TeamDPGoalsModel)
# — same components, same zero-sum league offset, same AD-safety idioms. HA stays on log λ_s
# ONLY: the 2026-07-17 EDA found home advantage is entirely shot VOLUME (p₁ 0.437 vs 0.443,
# p₂ 0.323 vs 0.324 h/a) — conversion is home-invariant.
#
# HOOKS INTO THE PROJECT API (no src/ changes — loader-local dispatch only):
#   Features.AbstractFeatureConfig + Features.add_feature!   -> BBCFunnelFeature
#   Features.required_features / PreGame.build_turing_model  -> engine plumbing
#   PreGame.extract_parameters                               -> posterior -> per-match λ draws
#   Pred.extract_params / Pred.compute_score_matrix          -> Poisson grid pricing
#   ([[dixoncoles-prediction-dispatch-union]]: the last two are MANDATORY — the struct subtypes
#    AbstractTimeDecayTeamModel <: AbstractNegBinModel but has no r column.)

using Turing
using Distributions
using DataFrames
using Dates
using LibPQ
using Serialization
using StatsFuns: logit
using LogExpFunctions: log1pexp

const ROOT_L03 = pkgdir(BayesianFootball)

# TeamDPGoalsModel (comparator) + _unpack_core + _poisson_score + the const aliases
# (PreGame / Features / Pred / Data) all come from the smile-stream loader.
include(joinpath(ROOT_L03, "current_development/scottish_lower_smile/l01_team_dp_league.jl"))

# ==========================================
# 1. BBC COUNTS — fetch + cache
# ==========================================
# NOTE: we do NOT reuse bbc_xg_proxy/l01_xg_proxy.jl::fetch_matches_wide — its PROXY_STATS
# deliberately excludes `shotsTotal` (collinear with its components for the xG GLM), and
# rebuilding shots as SoT+off+blocked is unsafe here (shotsBlocked is 33–65% `filled` in the
# lower tiers, WP0 §3). We pull shotsTotal directly. Same de-dup as WP0: `filled=false`
# (genuine) wins, but filled rows ARE used — they are backfilled-from-secondary, not imputed.

const FUNNEL_STATS = ["shotsTotal", "shotsOnTarget"]

"""One row per match: shots/SoT per side + goals + meta. Missing stats stay `missing`."""
function fetch_funnel_counts(conn; tournaments::Vector{Int} = [56, 57])
    t_in = join(tournaments, ",")
    stats_in = join(("'" * s * "'" for s in FUNNEL_STATS), ",")
    sql = """
    WITH st AS (
        SELECT DISTINCT ON (match_id, stat_type)
               match_id, stat_type, home_value, away_value
        FROM bbc.match_stats
        WHERE stat_type IN ($stats_in)
        ORDER BY match_id, stat_type, filled   -- genuine (filled=false) wins de-dup
    ),
    piv AS (
        SELECT match_id,
               max(home_value) FILTER (WHERE stat_type = 'shotsTotal')    AS shots_h,
               max(away_value) FILTER (WHERE stat_type = 'shotsTotal')    AS shots_a,
               max(home_value) FILTER (WHERE stat_type = 'shotsOnTarget') AS sot_h,
               max(away_value) FILTER (WHERE stat_type = 'shotsOnTarget') AS sot_a
        FROM st GROUP BY match_id
    )
    SELECT m.match_id, m.tournament_id, s.name AS season, m.start_timestamp,
           m.home_score, m.away_score,
           piv.shots_h, piv.shots_a, piv.sot_h, piv.sot_a
    FROM bbc.match_meta b
    JOIN sofascore.matches m ON m.match_id = b.match_id
    JOIN sofascore.seasons s ON s.season_id = m.season_id AND s.tournament_id = m.tournament_id
    JOIN piv ON piv.match_id = b.match_id
    WHERE m.tournament_id IN ($t_in) AND b.scores_match
    ORDER BY m.start_timestamp
    """
    df = DataFrame(execute(conn, sql))
    for c in ["shots_h", "shots_a", "sot_h", "sot_a"]
        df[!, c] = passmissing(Float64).(df[!, c])
    end
    return df
end

const FUNNEL_CACHE_PATH = joinpath(@__DIR__, "bbc_funnel_counts.jls")
const BBC_FUNNEL_DF = Ref{DataFrame}()                 # raw table (r03 Stage-0a QA reads this)
const BBC_FUNNEL    = Ref{Dict{Int, NamedTuple}}()     # match_id -> counts (the extractor reads this)

_cnt(x) = ismissing(x) || !isfinite(x) || x < 0 ? -1 : round(Int, x)   # -1 = absent sentinel

function _build_funnel_lookup(df::DataFrame)
    d = Dict{Int, NamedTuple}()
    for r in eachrow(df)
        d[Int(r.match_id)] = (
            shots_h = _cnt(r.shots_h), shots_a = _cnt(r.shots_a),
            sot_h   = _cnt(r.sot_h),   sot_a   = _cnt(r.sot_a),
        )
    end
    return d
end

"""
Load the bbc shot counts ONCE (Features.create_features runs per fold — src/features/builder.jl
— so the DB must never be hit inside the extractor). Called eagerly at include time.
"""
function ensure_bbc_funnel!(; tournaments::Vector{Int} = [56, 57], refresh::Bool = false)
    if !refresh && isassigned(BBC_FUNNEL)
        return BBC_FUNNEL[]
    end
    df = if !refresh && isfile(FUNNEL_CACHE_PATH)
        println("[l03] loading bbc funnel counts from cache: $FUNNEL_CACHE_PATH")
        deserialize(FUNNEL_CACHE_PATH)
    else
        println("[l03] querying bbc funnel counts (tournaments=$tournaments)...")
        conn = LibPQ.Connection(ENV["BF_DB_URL"])
        local d
        try
            d = fetch_funnel_counts(conn; tournaments)
        finally
            close(conn)
        end
        serialize(FUNNEL_CACHE_PATH, d)
        println("[l03] cached $(nrow(d)) matches -> $FUNNEL_CACHE_PATH")
        d
    end
    BBC_FUNNEL_DF[] = df
    BBC_FUNNEL[]    = _build_funnel_lookup(df)
    return BBC_FUNNEL[]
end

# ==========================================
# 2. FEATURE — loader-local (mirrors src/features/extractors/stats_extractors.jl)
# ==========================================
struct BBCFunnelFeature <: Features.AbstractFeatureConfig end

# AD-safety: emits Int counts with a 0 DUMMY where absent (never missing/NaN) plus a Float64
# usability mask. `sot ≤ shots` is folded into the mask here; `goals ≤ sot` is folded in at
# unpack time (goals live in the feature dict, not in the bbc table).
function Features.add_feature!(F_data::Dict, ::BBCFunnelFeature, ordered_ids, team_map::Dict,
                               ds::Data.DataStore)
    lut = ensure_bbc_funnel!()
    absent = (shots_h = -1, shots_a = -1, sot_h = -1, sot_a = -1)

    n = length(ordered_ids)
    shots_h = zeros(Int, n); shots_a = zeros(Int, n)
    sot_h   = zeros(Int, n); sot_a   = zeros(Int, n)
    mask_h  = zeros(Float64, n); mask_a = zeros(Float64, n)

    for (i, id) in enumerate(ordered_ids)
        c = get(lut, Int(id), absent)
        ok_h = c.shots_h >= 0 && c.sot_h >= 0 && c.sot_h <= c.shots_h
        ok_a = c.shots_a >= 0 && c.sot_a >= 0 && c.sot_a <= c.shots_a
        if ok_h
            shots_h[i] = c.shots_h; sot_h[i] = c.sot_h; mask_h[i] = 1.0
        end
        if ok_a
            shots_a[i] = c.shots_a; sot_a[i] = c.sot_a; mask_a[i] = 1.0
        end
    end

    F_data[:flat_home_shots_n]  = shots_h
    F_data[:flat_away_shots_n]  = shots_a
    F_data[:flat_home_sot_n]    = sot_h
    F_data[:flat_away_sot_n]    = sot_a
    F_data[:flat_funnel_mask_h] = mask_h
    F_data[:flat_funnel_mask_a] = mask_a
end

# ==========================================
# 3. ENGINE — TeamFunnelDPGoalsModel (funnel_pois)
# ==========================================
Base.@kwdef struct TeamFunnelDPGoalsModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig
    } <: PreGame.AbstractTimeDecayTeamModel
      # μ_base now lives on the SHOT scale: log(10.2) ≈ 2.32 (kwarg, no src change needed)
      interception_config::I  = PreGame.HierarchicalMonthlyInterception(
                                    prior_μ_base = Normal(2.3, 0.3))
      dynamics_config::T      = PreGame.TimeDecayDynamics()
      homeadvantage_config::H = PreGame.HierarchicalTeamHomeAdvantage()
      p1_prior::Distribution  = Normal(logit(0.44), 0.5)   # SoT | shot   (EDA: 0.44)
      p2_prior::Distribution  = Normal(logit(0.32), 0.5)   # goal | SoT   (EDA: 0.32)
      funnel_weight::Float64  = 1.0    # tempers the shots+SoT terms ONLY (0 ⇒ ≈ none_pois)
      league_offset_sd::Float64 = 0.1
      league_ha_sd::Float64     = 0.1
      league_ha_on::Bool        = false
end

"""
_unpack_core + the funnel counts, with ALL masking resolved to SAFE DUMMIES here.

Why dummies and not post-hoc masking: `-Inf * 0.0 == NaN` in Julia, which would poison the
whole gradient. Every masked-out slot is therefore evaluated on inputs that are *valid*
(0 successes out of 0 trials, Poisson at 0) and contributes exactly 0 after the mask.
"""
function _unpack_funnel(data, config)
    d = _unpack_core(data, config)

    shots_h = Vector{Int}(data[:flat_home_shots_n]); shots_a = Vector{Int}(data[:flat_away_shots_n])
    sot_h   = Vector{Int}(data[:flat_home_sot_n]);   sot_a   = Vector{Int}(data[:flat_away_sot_n])
    sm_h    = Vector{Float64}(data[:flat_funnel_mask_h])
    sm_a    = Vector{Float64}(data[:flat_funnel_mask_a])

    # cascade usable = stats usable AND goals ≤ SoT (≈1% of 56/57 matches fail this — own goals)
    cm_h = sm_h .* Float64.(d.home_goals .<= sot_h)
    cm_a = sm_a .* Float64.(d.away_goals .<= sot_a)

    zh, za = Int.(sm_h .> 0.5), Int.(sm_a .> 0.5)     # stats gate as 0/1 Int
    ch, ca = Int.(cm_h .> 0.5), Int.(cm_a .> 0.5)     # cascade gate as 0/1 Int

    return (; d...,
        shots_h_s = zh .* shots_h, shots_a_s = za .* shots_a,
        sot_h_s   = zh .* sot_h,   sot_a_s   = za .* sot_a,
        stats_mask_h = sm_h,       stats_mask_a = sm_a,
        sot_h_c   = ch .* sot_h,   sot_a_c   = ca .* sot_a,
        goals_h_c = ch .* d.home_goals, goals_a_c = ca .* d.away_goals,
        casc_mask_h = cm_h,        casc_mask_a = cm_a,
    )
end

@model function build_team_funnel_dp_goals_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    home_goals::Vector{Int}, away_goals::Vector{Int},
    match_weights::Vector{Float64},
    shots_h_s::Vector{Int}, shots_a_s::Vector{Int},
    sot_h_s::Vector{Int}, sot_a_s::Vector{Int},
    stats_mask_h::Vector{Float64}, stats_mask_a::Vector{Float64},
    sot_h_c::Vector{Int}, sot_a_c::Vector{Int},
    goals_h_c::Vector{Int}, goals_a_c::Vector{Int},
    casc_mask_h::Vector{Float64}, casc_mask_a::Vector{Float64},
    funnel_weight::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64,
    config
)
    # --- latent block: IDENTICAL to build_team_dp_goals_league_engine, but the rates are
    #     SHOT rates. HA on the home side only (conversion is home-invariant — see header). ---
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(PreGame.build_dynamics(config.dynamics_config, n_teams))

    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    int_m = view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    log_λ_s_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                       view(dyn.α, home_ids) .+ view(dyn.β, away_ids), -10.0, 10.0)
    log_λ_s_a = clamp.(int_m .+ lg .+
                       view(dyn.α, away_ids) .+ view(dyn.β, home_ids), -10.0, 10.0)
    λ_s_h = exp.(log_λ_s_h) .+ 1e-6
    λ_s_a = exp.(log_λ_s_a) .+ 1e-6

    # --- conversion (global; logit scale) ---
    p1_raw ~ config.p1_prior
    p2_raw ~ config.p2_prior
    # numerically-stable log p / log(1-p) straight off the logit (log1pexp, no logistic round-trip)
    log_p1, log_q1 = -log1pexp(-p1_raw), -log1pexp(p1_raw)
    log_p2, log_q2 = -log1pexp(-p2_raw), -log1pexp(p2_raw)
    p1, p2 = exp(log_p1), exp(log_p2)

    # AD-safe rejection (same idiom as l01)
    is_bad = any(isnan, λ_s_h) || any(isnan, λ_s_a) || any(isinf, λ_s_h) || any(isinf, λ_s_a) ||
             isnan(p1) || isnan(p2)
    λ_s_h = ifelse.(isnan.(λ_s_h) .| isinf.(λ_s_h), one.(λ_s_h), λ_s_h)
    λ_s_a = ifelse.(isnan.(λ_s_a) .| isinf.(λ_s_a), one.(λ_s_a), λ_s_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # --- LIKELIHOOD (branch-free, decay-weighted, per side) ---
    # Binomial terms are written as the parameter-dependent part of the log-pmf only; the
    # binomial coefficient does not involve p₁/p₂ and the cascade/marginal routing is fixed by
    # DATA, so dropping it shifts the log-posterior by a constant. This also keeps the compiled
    # ReverseDiff tape clean (AutoReverseDiff(compile=true), src/samplers/types.jl).
    ll_shots_h = logpdf.(Poisson.(λ_s_h), shots_h_s) .* stats_mask_h
    ll_shots_a = logpdf.(Poisson.(λ_s_a), shots_a_s) .* stats_mask_a

    ll_sot_h = (sot_h_s .* log_p1 .+ (shots_h_s .- sot_h_s) .* log_q1) .* stats_mask_h
    ll_sot_a = (sot_a_s .* log_p1 .+ (shots_a_s .- sot_a_s) .* log_q1) .* stats_mask_a

    # goals: cascade route where the counts are consistent, marginal Poisson(λ_s·p₁·p₂)
    # otherwise — so own-goal violations AND missing-stats matches still inform the goal rate.
    λ_g_h = λ_s_h .* (p1 * p2) .+ 1e-6
    λ_g_a = λ_s_a .* (p1 * p2) .+ 1e-6
    ll_goals_h = casc_mask_h .* (goals_h_c .* log_p2 .+ (sot_h_c .- goals_h_c) .* log_q2) .+
                 (1 .- casc_mask_h) .* logpdf.(Poisson.(λ_g_h), home_goals)
    ll_goals_a = casc_mask_a .* (goals_a_c .* log_p2 .+ (sot_a_c .- goals_a_c) .* log_q2) .+
                 (1 .- casc_mask_a) .* logpdf.(Poisson.(λ_g_a), away_goals)

    Turing.@addlogprob! sum(
        (funnel_weight .* (ll_shots_h .+ ll_shots_a .+ ll_sot_h .+ ll_sot_a) .+
         ll_goals_h .+ ll_goals_a) .* match_weights)
end

function Features.required_features(model::TeamFunnelDPGoalsModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), LeagueFeature(), Features.TimeIndicesFeature(),
        BBCFunnelFeature(),
    ]
end

function PreGame.build_turing_model(config::TeamFunnelDPGoalsModel, feature_set)
    d = _unpack_funnel(feature_set.data, config)
    return build_team_funnel_dp_goals_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        d.home_goals, d.away_goals, d.match_weights,
        d.shots_h_s, d.shots_a_s, d.sot_h_s, d.sot_a_s, d.stats_mask_h, d.stats_mask_a,
        d.sot_h_c, d.sot_a_c, d.goals_h_c, d.goals_a_c, d.casc_mask_h, d.casc_mask_a,
        config.funnel_weight,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        config.league_ha_on ? 1.0 : 0.0, config)
end

# ==========================================
# 4. EXTRACTION
# ==========================================
# Mirrors l01 §4 _extract_team_core, but UNTYPED in `model` so l04's hierarchical engine can
# reuse it (l01's version is pinned to the TeamLeagueModels Union). Returns SHOT rates.
function _extract_funnel_core(model, df, feature_set, chain)
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_leagues = Int(data[:n_leagues])
    team_map  = data[:team_map]
    league_lookup = data[:league_lookup]

    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt   = PreGame.extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)
    δ_mat    = _extract_league_offsets(chain, n_leagues, "δ_league_raw")
    γ_mat    = model.league_ha_on ? _extract_league_offsets(chain, n_leagues, "γ_league_raw") :
                                    zeros(size(δ_mat))

    n_samples = size(chain, 1) * size(chain, 3)
    core = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        mid   = Int(row.match_id)
        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)
        l_idx = get(league_lookup, mid, 0)

        α_h = h_idx > 0 ? dyn_nt.α[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn_nt.β[:, h_idx] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn_nt.α[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn_nt.β[:, a_idx] : zeros(n_samples)
        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)
        lg  = l_idx > 0 ? δ_mat[:, l_idx] : zeros(n_samples)
        γlg = l_idx > 0 ? γ_mat[:, l_idx] : zeros(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        m_idx = month(row.match_date)
        int_v = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        log_λ_s_h = clamp.(int_v .+ lg .+ γ_h .+ γlg .+ α_h .+ β_a, -10.0, 10.0)
        log_λ_s_a = clamp.(int_v .+ lg .+                α_a .+ β_h, -10.0, 10.0)

        core[mid] = (; λ_s_h = exp.(log_λ_s_h) .+ 1e-6,
                       λ_s_a = exp.(log_λ_s_a) .+ 1e-6,
                       h_idx, a_idx)
    end
    return core, n_samples
end

_logistic_draws(chain, sym) = 1 ./ (1 .+ exp.(-vec(Array(chain[Symbol(sym)]))))

function PreGame.extract_parameters(model::TeamFunnelDPGoalsModel, df, feature_set, chain)
    core, _ = _extract_funnel_core(model, df, feature_set, chain)
    p1 = _logistic_draws(chain, "p1_raw")
    p2 = _logistic_draws(chain, "p2_raw")
    conv = p1 .* p2

    results = Dict{Int, NamedTuple}()
    for (mid, c) in core
        λ_h = c.λ_s_h .* conv
        λ_a = c.λ_s_a .* conv
        results[mid] = (; λ_h, λ_a, λ_s_h = c.λ_s_h, λ_s_a = c.λ_s_a, p1, p2,
                          true_xg_h = λ_h, true_xg_a = λ_a)
    end
    return results
end

# ==========================================
# 5. PREDICTION OVERRIDES (loader-local; plain Poisson grid — thinning keeps it exact)
# ==========================================
Pred.extract_params(::TeamFunnelDPGoalsModel, row) = (λ_h = row.λ_h, λ_a = row.λ_a)
Pred.compute_score_matrix(::TeamFunnelDPGoalsModel, params; max_goals::Int = 12) =
    _poisson_score(params.λ_h, params.λ_a; max_goals)   # _poisson_score from l01 §5

# ==========================================
# 6. EAGER CACHE LOAD (never inside the per-fold extractor)
# ==========================================
ensure_bbc_funnel!()

println("[l03] funnel loader ready: TeamFunnelDPGoalsModel (funnel_pois) — " *
        "Shots~Poisson(λ_s), SoT|Shots~Bin(p₁), Goals|SoT~Bin(p₂); goals marginal stays " *
        "Poisson(λ_s·p₁·p₂). Counts cached for $(length(BBC_FUNNEL[])) matches. " *
        "Comparator TeamDPGoalsModel (none_pois) available from l01.")
