# current_development/ab_test_dixon_coles/l10_market_conjugate.jl
#
# Loader for r10: inference-time market conditioning of L1 goal-rate posteriors.
#
# Idea (conjugate, post-hoc, NO MCMC resampling):
#   extract_parameters() returns per-match posterior DRAWS of the log goal-rate
#   θ_1 = log λ_home, θ_2 = log λ_away (the DC score matrix reads θ_1/θ_2 directly).
#   The market pillar is Gaussian in log-λ, so we condition the posterior on the
#   target match's line m_s = log(flat_market_λ_s) via a per-match shrinkage weight k:
#
#     Transform A (location shift, variance-preserving):
#         ℓ'_j = ℓ_j + k·(m_s − μ_mod),     μ_mod = mean_j ℓ_j
#
#     Transform B (full conjugate, variance-shrinking):
#         k*       = (1/σ²)/(1/τ²_mod + 1/σ²)        σ² = mean(chain[:σ_market])²
#         var_post = 1/(1/τ²_mod + 1/σ²)             τ²_mod = var_j ℓ_j
#         μ_post   = μ_mod + k*·(m_s − μ_mod)
#         ℓ'_j     = μ_post + sqrt(var_post/τ²_mod)·(ℓ_j − μ_mod)
#
#   We then re-run the EXISTING pipeline (model_inference → process_signals →
#   pnl → tearsheet) on already-fitted chains, so one model is re-scored under many k.
#
# GUARDRAIL: condition on m built from `ds` (SofaScore-derived flat_market_λ); evaluate
# G/ROI on a DIFFERENT line by passing a Betfair `ds_eval` (1X2/DC only) to backtest_at_k.

using BayesianFootball
using DataFrames
using Statistics

const _Data        = BayesianFootball.Data
const _Features    = BayesianFootball.Features
const _Models      = BayesianFootball.Models
const _Experiments = BayesianFootball.Experiments
const _Predictions = BayesianFootball.Predictions
const _Signals     = BayesianFootball.Signals
const _BackTesting = BayesianFootball.BackTesting

# ---------------------------------------------------------------------------
# 1. Market anchor m per target match (SofaScore-derived flat_market_λ)
#    Reuses the SAME machinery the in-model market pillar uses
#    (src/features/extractors/market_extractors.jl).
# ---------------------------------------------------------------------------
"""
    build_market_info(ds; target_ids=nothing, config=DixonColesMarketFeature())

Returns Dict{match_id => (m_h, m_a)} where m_s = log(flat_market_λ_s).
Built from `ds.odds` (the SofaScore line), independent of any model's features so
it works for any base model. Pass `ds` (NOT the Betfair `ds_eval`).
"""
function build_market_info(ds; target_ids=nothing, config=_Features.DixonColesMarketFeature())
    odds = ds.odds
    if target_ids !== nothing
        id_set = Set(Int.(target_ids))
        odds = subset(odds, :match_id => ByRow(in(id_set)))
    end
    info = Dict{Int, NamedTuple{(:m_h, :m_a), Tuple{Float64, Float64}}}()
    n_fail = 0
    for g in groupby(odds, :match_id)
        mid = Int(first(g.match_id))
        try
            res    = _Features.fit_market_implied_parameters(g, config)
            params = _Features.extract_parameters(config, res.minimizer)  # (λ_home, λ_away, ρ)
            info[mid] = (m_h = log(params.λ_home), m_a = log(params.λ_away))
        catch err
            n_fail += 1
        end
    end
    @info "build_market_info: $(length(info)) matches fitted, $(n_fail) failed."
    return info
end

# ---------------------------------------------------------------------------
# 2. σ² plug-in for the conjugate transform (from the base model's own chains)
# ---------------------------------------------------------------------------
"""
    extract_sigma2_market(exp) -> Float64 | nothing

Mean σ_market across all folds' chains, squared. Used only for Transform B.
Returns nothing if the model has no :σ_market variable (pure no-market model).
"""
function extract_sigma2_market(exp)
    σs = Float64[]
    for item in exp.training_results.items
        chain = item[1]
        try
            if :σ_market in names(chain)
                append!(σs, vec(Array(chain[:σ_market])))
            end
        catch err
            # ignore folds without the variable
        end
    end
    isempty(σs) && return nothing
    return mean(σs)^2
end

# ---------------------------------------------------------------------------
# 3. The per-match k-update (post-hoc transform on the latents DataFrame)
# ---------------------------------------------------------------------------
function _update_side(ℓ::AbstractVector{<:Real}, m::Float64, k::Float64, mode::Symbol, sigma2)
    μ = mean(ℓ)
    if mode === :shift
        ℓ2 = ℓ .+ k .* (m - μ)
    elseif mode === :conjugate
        sigma2 === nothing && error("Transform B (:conjugate) needs sigma2 (no σ_market in chain).")
        τ2     = max(var(ℓ), 1e-8)
        kstar  = (1.0 / sigma2) / (1.0 / τ2 + 1.0 / sigma2)
        varpst = 1.0 / (1.0 / τ2 + 1.0 / sigma2)
        μ_post = μ + kstar * (m - μ)
        ℓ2     = μ_post .+ sqrt(varpst / τ2) .* (ℓ .- μ)
    else
        error("unknown mode $mode (use :shift or :conjugate)")
    end
    return collect(Float64, ℓ2)
end

"""
    apply_k_update(latents, market_info, k; mode=:shift, sigma2=nothing) -> LatentStates

Returns a NEW LatentStates (deepcopy) with θ_1/θ_2 (and λ_h/λ_a kept consistent)
shifted toward the market anchor. Matches with no market line are left unchanged.
The DC/DP score matrices read θ_1/θ_2, so those are the operative columns.
"""
function apply_k_update(latents, market_info, k::Float64; mode::Symbol=:shift, sigma2=nothing)
    df = deepcopy(latents.df)
    for i in 1:nrow(df)
        mid = Int(df.match_id[i])
        haskey(market_info, mid) || continue
        mi = market_info[mid]
        θ1 = _update_side(df.θ_1[i], mi.m_h, k, mode, sigma2)
        θ2 = _update_side(df.θ_2[i], mi.m_a, k, mode, sigma2)
        df.θ_1[i] = θ1
        df.θ_2[i] = θ2
        df.λ_h[i] = exp.(θ1)
        df.λ_a[i] = exp.(θ2)
    end
    return _Experiments.LatentStates(df, latents.model)
end

# ---------------------------------------------------------------------------
# 4. Backtest one experiment at a fixed k (mirrors src/backtesting/processor.jl)
#    base_latents is computed ONCE per experiment and reused across k.
# ---------------------------------------------------------------------------
"""
    backtest_at_k(ds_eval, exp, base_latents, market_info, k, signals;
                  market_config, mode=:shift, sigma2=nothing, label=nothing) -> DataFrame

Applies the k-update to `base_latents`, runs model_inference → process_signals on
`ds_eval.odds` (the held-out line), computes pnl, tags model_name with k. Returns the
per-bet DataFrame (concatenate over k and wrap in BacktestLedger for a tearsheet).
"""
function backtest_at_k(ds_eval, exp, base_latents, market_info, k::Float64,
                       signals; market_config, mode::Symbol=:shift, sigma2=nothing,
                       label=nothing, odds_column::Symbol=:odds_close)
    lat_k = apply_k_update(base_latents, market_info, k; mode=mode, sigma2=sigma2)
    ppd   = _Predictions.model_inference(lat_k; market_config=market_config)
    sig   = _Signals.process_signals(ppd, ds_eval.odds, signals; odds_column=odds_column)
    df    = sig.df

    df.pnl = map(eachrow(df)) do r
        if ismissing(r.is_winner) || r.stake == 0.0
            0.0
        elseif r.is_winner
            r.stake * (r.odds - 1.0)
        else
            -r.stake
        end
    end

    name = label === nothing ? "$(exp.config.name)_$(mode)_k$(round(k, digits=3))" : label
    df.model_name       = fill(name, nrow(df))
    df.model_parameters = fill(_Models.model_parameters(exp.config.model), nrow(df))
    return df
end

# ---------------------------------------------------------------------------
# 5. Sweep helper: one experiment, many k -> tearsheet
# ---------------------------------------------------------------------------
"""
    sweep_experiment(ds_cond, ds_eval, exp, ks, signals; market_config, kwargs...)

ds_cond : SofaScore DataStore used to BUILD the market anchor m (conditioning line).
ds_eval : Betfair DataStore used to EVALUATE (held-out line).
Returns (ledger_df, base_latents, market_info). base_latents is returned so callers can
also run the :conjugate point or a SofaScore secondary eval without recomputing.
"""
function sweep_experiment(ds_cond, ds_eval, exp, ks::AbstractVector{<:Real}, signals;
                          market_config, odds_column::Symbol=:odds_close,
                          add_conjugate::Bool=true)
    base_latents = _Experiments.extract_oos_predictions(ds_cond, exp)
    target_ids   = unique(Int.(base_latents.df.match_id))
    market_info  = build_market_info(ds_cond; target_ids=target_ids)
    sigma2       = extract_sigma2_market(exp)

    blocks = DataFrame[]
    for k in ks
        push!(blocks, backtest_at_k(ds_eval, exp, base_latents, market_info, Float64(k),
                                    signals; market_config=market_config,
                                    mode=:shift, odds_column=odds_column))
    end
    if add_conjugate && sigma2 !== nothing
        push!(blocks, backtest_at_k(ds_eval, exp, base_latents, market_info, 0.0,
                                    signals; market_config=market_config, mode=:conjugate,
                                    sigma2=sigma2, odds_column=odds_column,
                                    label="$(exp.config.name)_conjugate"))
    end

    ledger_df = reduce(vcat, blocks)
    return (ledger = ledger_df, base_latents = base_latents,
            market_info = market_info, sigma2 = sigma2)
end

println("[l10] loaded: build_market_info, extract_sigma2_market, apply_k_update, backtest_at_k, sweep_experiment")
