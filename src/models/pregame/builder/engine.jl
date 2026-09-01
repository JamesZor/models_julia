# ==============================================================================
# 05 — Composable Count Model Builder : THE ONE ENGINE
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# ONE `@model`. It replaces `_engw`, `_engd`, `_engj`
# (02_poisson_wealth/l00_feature_poisson.jl), `build_weighted_poisson_goals_engine`
# (00_team_poisson/l01_model.jl), and `build_weighted_goals_engine`
# (src/.../team_level/time_decay/goals.jl) — five engines whose only differences
# are which covariates appear in η and which density reads it.
#
# HOW IT STAYS FAST  (docs/turing_ad_performance_guide.md)
#
#   The predictor set is a TYPED TUPLE on the model object, so
#   `_predictor_block(::Tuple{A,B}, ...)` recurses through `Base.tail` and Julia
#   unrolls it at compile time into straight-line code. There is no runtime loop
#   over predictors, no runtime branch, and no `Vector{AbstractPredictorTerm}`
#   dispatch inside `@model` — the tape sees exactly the same instruction sequence
#   a hand-written engine would emit.
#
#   The zero-predictor case returns `nothing`, and
#   `_predictor_shift(η, ::Nothing) = η` dispatches the addition away entirely.
#   A model with no predictors therefore
#   records a tape IDENTICAL to the baseline engine's — composition costs nothing
#   when you compose nothing. (Returning `zeros(n)` instead would have added two
#   broadcast nodes per gradient, forever, to every model that uses no predictors.)
#
#   Everything conditional lives in `build_turing_model`: log-factorials, decay
#   weights, design-vector validation, type casting. Inside `@model` there is only
#   broadcast arithmetic, indexed parameter selection, `clamp`, `sum`, and
#   `@addlogprob!`.
#
#   One departure from the guide, made on measurement: parameter selection uses
#   `A[idx]`, not `view(A, idx)`. See §4 and docs/tickets/T002.
#
# ==============================================================================

# Dependencies and builder definitions are loaded by builder-module.jl.


# ==============================================================================
# 1. FEATURE DERIVATION
# ==============================================================================
#
# Not written per model. The structural block is fixed by the linear predictor;
# the rest is whatever the assembled predictors ask for. Add a term and this
# function's answer changes without this function changing.

"The features every composable count model needs, whatever else is bolted on."
const CB_STRUCTURAL_FEATURES = (
    CB_Features.TeamIDsFeature,
    CB_Features.GoalsFeature,
    CB_Features.DatesFeature,
    CB_Features.MonthFeature,
    CB_Features.TimeIndicesFeature,
)

function CB_Features.required_features(model::ComposableCountModel)
    out = CB_Features.AbstractFeatureConfig[T() for T in CB_STRUCTURAL_FEATURES]
    append!(out, dynamics_features(model.dynamics))
    for term in model.covariates
        append!(out, predictor_features(term))
    end
    # An observation with its own data channel — the joint model's proxy-xG arm is the first —
    # declares it the same way a covariate does. Poisson and NegBin contribute nothing here.
    append!(out, observation_features(model.observation))
    return out
end


# ==============================================================================
# 2. THE PREDICTOR BLOCK  (compile-time unrolled)
# ==============================================================================

"""
One covariate: sample its scalar weight, return its signed contribution to each
side. The site is named `w`; the caller prefixes it with the covariate's own name,
so the chain carries `wealth.w`, `distance.w`, … derived from the components.
"""
@model function _predictor_term(c::AbstractCovariateConfig, x::Vector{Float64})
    w ~ covariate_prior(c)
    q = w .* x
    h, a = covariate_sides(covariate_role(c), q)
    return (; h, a)
end

@model function _predictor_term(c::PlayerLineupPillar,
                                design::AbstractPlayerLineupDesign)
    effects ~ to_submodel(_player_lineup_term(c, design), false)
    return effects
end

"""
The unrolled walk over the predictor tuple.

`to_submodel(..., false)` suppresses DynamicPPL's automatic left-hand-side
prefixing so that the explicit `prefix(..., Val(name))` is the ONLY prefix a
predictor site carries; without it the recursion would nest names
(`head.head.head.w`) and the chain schema would depend on how many predictors
happened to precede this one.
"""
@model function _predictor_block(terms::Tuple, designs::Tuple, n::Int)
    head ~ to_submodel(
        DynamicPPL.prefix(
            _predictor_term(first(terms), first(designs)),
            Val(predictor_name(first(terms))),
        ),
        false)
    rest ~ to_submodel(
        _predictor_block(Base.tail(terms), Base.tail(designs), n), false)
    return (; h = _predictor_acc(head.h, rest.h),
              a = _predictor_acc(head.a, rest.a))
end

"Base case: no predictors, no site, no tape node. `nothing` is a structural zero."
@model function _predictor_block(::Tuple{}, ::Tuple{}, n::Int)
    return (; h = nothing, a = nothing)
end

# Dispatch, not a branch: resolved from the type at compile time.
_predictor_acc(x, ::Nothing) = x
_predictor_acc(x, y)         = x .+ y
_predictor_shift(η, ::Nothing) = η
_predictor_shift(η, q)         = η .+ q


# ==============================================================================
# 3. THE OBSERVATION BLOCK
# ==============================================================================
#
# Takes the two log-intensities and returns a scalar log-likelihood. The engine
# adds it. Splitting it out this way is what lets a new count density be one method
# rather than one engine file.

"""
Poisson, evaluated directly in log-intensity space: `y·η − exp(η) − log Γ(y+1)`.
The log-factorial is data, precomputed in `build_turing_model`.
"""
@model function _observe(::PoissonObservation,
                         η_h, η_a,
                         yh::Vector{Int}, ya::Vector{Int}, wts::Vector{Float64},
                         lfh::Vector{Float64}, lfa::Vector{Float64},
                         n_teams::Int, n_months::Int, ::Nothing)
    # The decay weight is applied in its OWN broadcast, not fused into the tracked
    # likelihood expression. Same value to the last bit; 1.6x the gradient
    # throughput on fold 1: fusing a constant into the tracked expression widens the
    # elementwise kernel ReverseDiff differentiates. Measured, not assumed — see the
    # corroboration section of docs/tickets/T002.
    ll_h = yh .* η_h .- exp.(η_h) .- lfh
    ll_a = ya .* η_a .- exp.(η_a) .- lfa
    return sum(ll_h .* wts) + sum(ll_a .* wts)
end

# Smooth, branch-free saturation of log-dispersion to (-10, 10). The high even
# power is effectively identity over the prior's typical region (the difference
# at log_r=3.1 is below machine-relevant density tolerance) while remaining
# compiled-tape safe when a trajectory crosses ±10.
_cb_bound_dispersion_log(x) = x / (1 + (x / 10)^24)^(1 / 24)

@model function _build_count_dispersion(config::CB_PG.GlobalDispersion,
                                        n_teams::Int, n_months::Int)
    log_r ~ config.log_r
    r = exp(_cb_bound_dispersion_log(log_r))
    return (; h = r, a = r)
end

@model function _build_count_dispersion(config::CB_PG.HomeAwayDispersion,
                                        n_teams::Int, n_months::Int)
    log_r ~ config.log_r
    δ_r_home ~ config.δ_r_home
    r_a = exp(_cb_bound_dispersion_log(log_r))
    r_h = exp(_cb_bound_dispersion_log(log_r + δ_r_home))
    return (; h = r_h, a = r_a)
end

"""
Negative binomial, retaining the legacy `disp.*` sites while using a branch-free,
compiled-tape-safe dispersion bound.
"""
@model function _observe(o::NegativeBinomialObservation,
                         η_h, η_a,
                         yh::Vector{Int}, ya::Vector{Int}, wts::Vector{Float64},
                         lfh::Vector{Float64}, lfa::Vector{Float64},
                         n_teams::Int, n_months::Int, ::Nothing)
    disp ~ to_submodel(_build_count_dispersion(o.dispersion, n_teams, n_months))
    λ_h = exp.(η_h)
    λ_a = exp.(η_a)

    # Evaluate the density directly. Constructing a DiscreteDistribution inside
    # this broadcast makes ForwardDiff dual values flow through the generic
    # integer-support path when ReverseDiff compiles the tape. The direct formula
    # is mathematically identical and keeps every tracked operation continuous.
    total_h = log.(disp.h .+ λ_h)
    total_a = log.(disp.a .+ λ_a)
    ll_h = SpecialFunctions.loggamma.(yh .+ disp.h) .-
           SpecialFunctions.loggamma.(disp.h) .- lfh .+
           disp.h .* (log.(disp.h) .- total_h) .+
           yh .* (η_h .- total_h)
    ll_a = SpecialFunctions.loggamma.(ya .+ disp.a) .-
           SpecialFunctions.loggamma.(disp.a) .- lfa .+
           disp.a .* (log.(disp.a) .- total_a) .+
           ya .* (η_a .- total_a)
    return sum(ll_h .* wts) + sum(ll_a .* wts)
end


"""
The two scalars the joint observation owns. Declared in one submodel so the chain carries
`obs.ν` and `obs.log_κ`, in this order — which is the θ layout `cb_varinfo_sites` reports.
"""
@model function _joint_gamma_poisson_params(o::JointGammaPoissonObservation)
    ν ~ o.shape_prior
    log_κ ~ o.log_kappa_prior
    return (; ν, log_κ)
end

"""
Two arms on one latent `μ = exp(η)`.

    ARM 1   pxg ~ Gamma(ν, μ/ν)        masked to the matches that have a measurement
    ARM 2   y   ~ Poisson(κ · μ)       every match in the fold

Both are written out in log-intensity space with every data-only quantity precomputed, so the tape
sees only broadcasts of tracked scalars against constant vectors:

    log Gamma(x; ν, μ/ν) = (ν−1)·log x − ν·x·e^(−η) − ν·η + ν·log ν − log Γ(ν)
    log Poisson(y; κμ)   = y·(η + log κ) − e^(η + log κ) − log Γ(y+1)

`log x` and `log Γ(y+1)` are data. The availability mask is data, and it was multiplied into the
time-decay weights ONCE in `observation_design` — so masking here costs the same single broadcast
that weighting already cost, and a masked-out match contributes a finite term times an exact zero
rather than a branch.

WHY THE ARMS ARE NOT FUSED. Writing `sum((ll_goals .+ ll_proxy) .* wts)` would be the same number,
but it widens the elementwise kernel ReverseDiff differentiates across two densities with different
masks. Kept separate for the same measured reason `_observe(::PoissonObservation, …)` keeps the
decay weight in its own broadcast — see docs/tickets/T002.
"""
@model function _observe(o::JointGammaPoissonObservation,
                         η_h, η_a,
                         yh::Vector{Int}, ya::Vector{Int}, wts::Vector{Float64},
                         lfh::Vector{Float64}, lfa::Vector{Float64},
                         n_teams::Int, n_months::Int, od::JointGammaPoissonDesign)
    obs ~ to_submodel(_joint_gamma_poisson_params(o))
    ν = obs.ν

    # --- ARM 2: Poisson goals on λ = κ·μ, over the whole fold ------------------
    ζ_h = η_h .+ obs.log_κ
    ζ_a = η_a .+ obs.log_κ
    ll_h = yh .* ζ_h .- exp.(ζ_h) .- lfh
    ll_a = ya .* ζ_a .- exp.(ζ_a) .- lfa
    goals_ll = sum(ll_h .* wts) + sum(ll_a .* wts)

    # --- ARM 1: Gamma proxy xG on μ, over the covered matches only -------------
    # `log_norm` collects the two terms that depend on ν alone. Broadcasting it in as a tracked
    # scalar is one tape node; recomputing `loggamma(ν)` per match would be n.
    log_norm = ν * log(ν) - SpecialFunctions.loggamma(ν)
    inv_μ_h = exp.(.-η_h)
    inv_μ_a = exp.(.-η_a)
    g_h = (ν - 1.0) .* od.log_pxg_h .- (ν .* od.pxg_h) .* inv_μ_h .- ν .* η_h .+ log_norm
    g_a = (ν - 1.0) .* od.log_pxg_a .- (ν .* od.pxg_a) .* inv_μ_a .- ν .* η_a .+ log_norm
    proxy_ll = sum(g_h .* od.mask_weights) + sum(g_a .* od.mask_weights)

    return goals_ll + proxy_ll
end


# ==============================================================================
# 4. THE ENGINE
# ==============================================================================
#
#     η_h = μ_{s(i)} + δ_{m(i)} + γ_{h(i)} + α_{h(i)} + β_{a(i)} + Σ_k q_k,h(i)
#     η_a = μ_{s(i)} + δ_{m(i)}            + α_{a(i)} + β_{h(i)} + Σ_k q_k,a(i)
#
# The numerical guard on η is a COMPONENT (`ClampGuard` / `NoGuard`), because the
# arms disagree about it: arms 02/03/04 clamp to [-10, 10] and arm 00 does not.
# `apply_guard` dispatches on its type, so `NoGuard` emits no instruction at all.
# r01_demo.jl §7 checks that the clamp never binds at the draws it compares, which
# is what makes the two settings the same function there.
#
# Declaration order is inter → ha → dyn → predictors → observation, matching
# `_engw`/`_engd`/`_engj` exactly, so the θ vector this model produces is
# element-for-element the θ vector the hand-written arms produce.

@model function _cb_dynamics_effects(
    config::Union{CB_PG.TimeDecayDynamics,CB_PG.StaticZeroDynamics},
    home_ids::Vector{Int}, away_ids::Vector{Int}, ::Nothing, n_teams::Int,
)
    state ~ to_submodel(CB_PG.build_dynamics(config, n_teams), false)
    return (;
        att_h = state.α[home_ids], def_a = state.β[away_ids],
        att_a = state.α[away_ids], def_h = state.β[home_ids],
    )
end

@model function composable_count_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_ids::Vector{Int}, month_ids::Vector{Int},
    home_goals::Vector{Int}, away_goals::Vector{Int},
    match_weights::Vector{Float64},
    dynamics_data,
    predictor_designs::Tuple,
    log_fact_h::Vector{Float64}, log_fact_a::Vector{Float64},
    observation_data,
    n_matches::Int, n_teams::Int, n_seasons::Int, n_months::Int,
    config::ComposableCountModel,
)
    # --- 1. STRUCTURAL COMPONENTS (reused from src, unmodified) ---------------
    inter ~ to_submodel(CB_PG.build_interception(config.interception, n_seasons, n_months))
    ha    ~ to_submodel(CB_PG.build_home_advantage(config.home_advantage, n_teams))
    dyn   ~ to_submodel(_cb_dynamics_effects(
        config.dynamics, home_ids, away_ids, dynamics_data, n_teams))

    # --- 2. PREDICTORS (unrolled over the typed tuple) ------------------------
    pred ~ to_submodel(
        _predictor_block(config.covariates, predictor_designs, n_matches), false)

    # --- 3. LOG-INTENSITIES ---------------------------------------------------
    # `A[idx]`, NOT `view(A, idx)`. The AD guide's Rule 4 says the opposite and is
    # wrong for this ReverseDiff version: `view` on a `TrackedArray` yields a
    # `SubArray` of `TrackedReal` that the tape walks element by element, while
    # `getindex` stays a single vectorised node. Same log-density to the last bit,
    # 5x the gradient throughput on fold 1 — see docs/tickets/T002, which raised
    # this against the `src` engines before this prototype existed.
    base = inter.μ_base[season_ids] .+ inter.δ_month[month_ids]

    η_h = apply_guard(config.guard,
                      _predictor_shift(base .+ ha[home_ids] .+
                                       dyn.att_h .+ dyn.def_a, pred.h))
    η_a = apply_guard(config.guard,
                      _predictor_shift(base .+
                                       dyn.att_a .+ dyn.def_h, pred.a))

    # --- 4. OBSERVATION -------------------------------------------------------
    ll ~ to_submodel(_observe(config.observation, η_h, η_a, home_goals, away_goals,
                              match_weights, log_fact_h, log_fact_a, n_teams, n_months,
                              observation_data),
                     false)
    Turing.@addlogprob! ll
end


# ==============================================================================
# 5. THE BUILDER LAYER  (all conditional logic lives here)
# ==============================================================================

"""
    cb_design(model, feature_set) -> NamedTuple

Every vector the engine consumes, cast to a concrete type, checked, and — for the
predictors — assembled into a Tuple whose TYPE encodes how many there are. This is
the only place that touches the `FeatureSet`.

The checks are not defensive padding. A covariate column of the wrong length is a
fold-alignment bug, and a non-finite entry becomes a `-Inf` log-density three hours
into a grid run; both are cheap to catch here and expensive to catch there.
"""
function cb_design(model::ComposableCountModel, feature_set)
    d = feature_set.data

    home_ids   = Vector{Int}(d[:flat_home_ids])
    away_ids   = Vector{Int}(d[:flat_away_ids])
    season_ids = Vector{Int}(d[:season_indices])
    month_ids  = Vector{Int}(d[:flat_months])
    home_goals = Vector{Int}(d[:flat_home_goals])
    away_goals = Vector{Int}(d[:flat_away_goals])
    n_matches  = length(home_ids)

    # The structural team dynamics exclusively own likelihood recency. A lineup
    # pillar is a point-in-time predictor and cannot alter this clock.
    match_weights = dynamics_match_weights(
        model.dynamics, Vector{Float64}(d[:dates]))

    # Precomputed once, outside `@model`: log Γ(y+1) is data, not a parameter.
    log_fact_h = SpecialFunctions.loggamma.(Float64.(home_goals) .+ 1.0)
    log_fact_a = SpecialFunctions.loggamma.(Float64.(away_goals) .+ 1.0)

    dynamics_data = dynamics_design(model.dynamics, feature_set, n_matches)

    # `map` over a Tuple returns a Tuple, so predictor count and design types are
    # baked into the engine specialization and the recursion is compile-time unrolled.
    predictor_designs = map(
        term -> predictor_design(term, feature_set, n_matches), model.covariates)

    for (name, v) in (("home_ids", home_ids), ("away_ids", away_ids),
                      ("season_ids", season_ids), ("month_ids", month_ids),
                      ("home_goals", home_goals), ("away_goals", away_goals),
                      ("match_weights", match_weights))
        length(v) == n_matches ||
            error("design vector $name has length $(length(v)); expected $n_matches")
    end
    all(isfinite, match_weights) || error("non-finite match weights (check days_half_life)")

    # The observation layer's own design data, if it has any. Built AFTER the length checks so a
    # mis-sized goals vector is reported as such rather than as a mask-length mismatch.
    observation_data = observation_design(model.observation, feature_set, n_matches, match_weights)

    return (; home_ids, away_ids, season_ids, month_ids, home_goals, away_goals,
              match_weights, dynamics_data, predictor_designs, log_fact_h, log_fact_a, observation_data,
              n_matches, n_teams = Int(d[:n_teams]), n_seasons = Int(d[:n_seasons]),
              n_months = 12)
end

function CB_PG.build_turing_model(model::ComposableCountModel, feature_set)
    z = cb_design(model, feature_set)
    return composable_count_engine(
        z.home_ids, z.away_ids, z.season_ids, z.month_ids,
        z.home_goals, z.away_goals, z.match_weights,
        z.dynamics_data, z.predictor_designs, z.log_fact_h, z.log_fact_a, z.observation_data,
        z.n_matches, z.n_teams, z.n_seasons, z.n_months,
        model,
    )
end


# ==============================================================================
# 6. EXTRACTION  (also derived, never written per model)
# ==============================================================================
#
# The structural half delegates to the same `src` component extractors the
# hand-written engines call. The covariate half walks the same tuple in the same
# order and reads `<name>.w` out of the chain — the site name the engine created
# from the same `covariate_name`. Nothing here mentions wealth or distance.

function _cb_extract_dynamics(chain::Chains,
                              config::Union{CB_PG.TimeDecayDynamics,CB_PG.StaticZeroDynamics},
                              prefix::String, n_teams::Int)
    return CB_PG.extract_dynamics(chain, config, prefix, n_teams)
end

function _cb_oos_dynamics(
    config::Union{CB_PG.TimeDecayDynamics,CB_PG.StaticZeroDynamics}, draw,
    lineup_map, match_id::Int, home_index::Int, away_index::Int, n_samples::Int,
)
    return (;
        att_h = home_index > 0 ? draw.α[:, home_index] : zeros(n_samples),
        def_a = away_index > 0 ? draw.β[:, away_index] : zeros(n_samples),
        att_a = away_index > 0 ? draw.α[:, away_index] : zeros(n_samples),
        def_h = home_index > 0 ? draw.β[:, home_index] : zeros(n_samples),
    )
end

function CB_PG.extract_parameters(model::ComposableCountModel,
                                  df::AbstractDataFrame,
                                  feature_set,
                                  chain::Chains)
    d         = feature_set.data
    n_teams   = Int(d[:n_teams])
    n_seasons = Int(d[:n_seasons])
    team_map  = d[:team_map]
    n_samples = size(chain, 1) * size(chain, 3)

    inter_nt = CB_PG.extract_interception(chain, model.interception, n_seasons)
    ha_mat   = CB_PG.extract_home_advantage(chain, model.home_advantage, n_teams)
    dyn_nt   = _cb_extract_dynamics(chain, model.dynamics, "dyn", n_teams)
    lineup_map = get(d, :player_lineup_ratings_map,
                     Dict{Int,CB_Features.PMLineupAggregate}())

    # Team and predictor posterior blocks are reconstructed independently. In
    # particular, a missing lineup bridge entry cannot erase dyn.α/dyn.β.
    predictor_draws = [
        predictor_extract(chain, term, String(predictor_name(term)))
        for term in model.covariates
    ]
    predictor_sources = [
        _cb_predictor_oos_source(term, feature_set, df, lineup_map)
        for term in model.covariates
    ]

    disp_nt = _cb_extract_observation(model.observation, chain, n_teams)

    global_ha = model.home_advantage isa CB_PG.GlobalHomeAdvantage
    results = Dict{Int, NamedTuple}()

    for row in eachrow(df)
        mid   = Int(row.match_id)
        h_idx = get(team_map, row.home_team, 0)
        a_idx = get(team_map, row.away_team, 0)

        team_effects = _cb_oos_dynamics(model.dynamics, dyn_nt, lineup_map, mid,
                                         h_idx, a_idx, n_samples)
        γ_h = global_ha ? ha_mat[:, 1] : (h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples))

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        base  = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, Dates.month(row.match_date)]

        q_h = zeros(n_samples)
        q_a = zeros(n_samples)
        for k in eachindex(model.covariates)
            q = predictor_oos(
                model.covariates[k], predictor_draws[k], predictor_sources[k], row)
            q_h .+= q.h
            q_a .+= q.a
        end

        η_h = apply_guard(model.guard,
                          base .+ γ_h .+ team_effects.att_h .+ team_effects.def_a .+ q_h)
        η_a = apply_guard(model.guard,
                          base .+       team_effects.att_a .+ team_effects.def_h .+ q_a)
        λ_h = exp.(η_h)
        λ_a = exp.(η_a)

        results[mid] = _cb_rates(model.observation, λ_h, λ_a, disp_nt, h_idx, a_idx,
                                 Dates.month(row.match_date))
    end
    return results
end

_cb_predictor_oos_source(::PlayerLineupPillar, feature_set, df, lineup_map) = lineup_map
function _cb_predictor_oos_source(c::AbstractCovariateConfig, feature_set, df, lineup_map)
    values = _cb_checked_oos(c, feature_set, df)
    return Dict(Int(row.match_id) => values[i] for (i, row) in enumerate(eachrow(df)))
end

function _cb_checked_oos(c::AbstractCovariateConfig, feature_set, df)
    x = covariate_oos(c, feature_set, df)
    length(x) == nrow(df) ||
        error("covariate $(covariate_name(c)) OOS column has length $(length(x)); expected $(nrow(df))")
    all(isfinite, x) || error("covariate $(covariate_name(c)) OOS column has non-finite entries")
    return Float64.(x)
end

_cb_extract_observation(::PoissonObservation, chain, n_teams) = nothing

function _cb_extract_observation(::JointGammaPoissonObservation, chain, n_teams)
    ν = vec(Array(chain[Symbol("obs.ν")]))
    κ = exp.(vec(Array(chain[Symbol("obs.log_κ")])))
    return (; ν, κ)
end

function _cb_extract_observation(o::NegativeBinomialObservation{<:CB_PG.GlobalDispersion},
                                 chain, n_teams)
    log_r = vec(Array(chain[Symbol("disp.log_r")]))
    r = exp.(_cb_bound_dispersion_log.(log_r))
    return (; h = r, a = r)
end

function _cb_extract_observation(o::NegativeBinomialObservation{<:CB_PG.HomeAwayDispersion},
                                 chain, n_teams)
    log_r = vec(Array(chain[Symbol("disp.log_r")]))
    home_offset = vec(Array(chain[Symbol("disp.δ_r_home")]))
    r_a = exp.(_cb_bound_dispersion_log.(log_r))
    r_h = exp.(_cb_bound_dispersion_log.(log_r .+ home_offset))
    return (; h = r_h, a = r_a)
end

# The prediction NamedTuple must carry exactly what the score grid for this family
# reads. `true_xg_h/a` mirror λ so the downstream evaluation path is unchanged.
_cb_rates(::PoissonObservation, λ_h, λ_a, _, h_idx, a_idx, m_idx) =
    (; λ_h, λ_a, true_xg_h = λ_h, true_xg_a = λ_a)

# The joint model is the one case where λ and the expected xG genuinely differ, so `true_xg_*`
# carries μ rather than mirroring λ: μ IS the quantity the Gamma arm measured, and κ is exactly the
# league finishing factor separating the two. `extract_latents(::PoissonCountFamily, …)` reads
# `λ_h/λ_a` and ignores the rest, so μ, κ and ν ride along as diagnostics without changing the grid.
function _cb_rates(::JointGammaPoissonObservation, μ_h, μ_a, obs_nt, h_idx, a_idx, m_idx)
    λ_h = obs_nt.κ .* μ_h
    λ_a = obs_nt.κ .* μ_a
    return (; λ_h, λ_a, μ_h, μ_a, κ = obs_nt.κ, ν = obs_nt.ν,
              true_xg_h = μ_h, true_xg_a = μ_a)
end

function _cb_rates(::NegativeBinomialObservation, λ_h, λ_a, disp_nt, h_idx, a_idx, m_idx)
    r = CB_PG.reconstruct_dispersion(disp_nt, h_idx, a_idx, m_idx)
    return (; λ_h, λ_a, r_h = r.h, r_a = r.a, true_xg_h = λ_h, true_xg_a = λ_a)
end
