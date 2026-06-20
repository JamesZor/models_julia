#=
l03_inplay_turing.jl  —  Bayesian (Turing.jl) in-play scoring-intensity model.

Bayesian twin of the l02 Poisson GLM, with TWO configurable hierarchical axes:

  GAME STATE (config.game_state):
    :none          — no game-state effect (pregame + time + home only)
    :linear        — global trailing/leading dummies (the original l03 behaviour)
    :hier_replace  — drop the dummies, use a partial-pooled per-state intercept δ_state[gs_idx]
                     (global + game_state[index]; gs_idx = clamp(goal_diff,-3,3)+4 ∈ 1..7)
    :hier_addon    — keep the dummies AND add δ_state deviations on top

  TEAM EFFECTS (orthogonal, independently toggled, non-centered):
    use_team_attack / use_team_defense / use_team_trailing / use_team_leading

Target = realized remaining goals; Poisson with time-exposure offset log(rem_frac).
log(pregame λ) stays a covariate; deltas capture only RESIDUAL in-play behaviour.

Follows docs/turing_ad_performance_guide.md: broadcast + view, no scalar loops, clamp+1e-6,
config branches only on CONSTANT fields (static compile=true tape). The per-config design-matrix
column selection happens in `make_model` (outside @model), so one `InPlayInputs` serves every variant.

Reuses: build_intensity_dataset (l02), Samplers.run_sampler / NUTSConfig.
=#

using Turing
using Distributions
using DataFrames
using Statistics
using LinearAlgebra
using MCMCChains

const Samplers = BayesianFootball.Samplers

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

Base.@kwdef struct InPlayIntensityConfig
    game_state::Symbol      = :linear            # :none | :linear | :hier_replace | :hier_addon
    use_team_attack::Bool   = false
    use_team_defense::Bool  = false
    use_team_trailing::Bool = false
    use_team_leading::Bool  = false
    use_momentum::Bool      = false          # causal SofaScore-momentum covariate (l06)
    σ_α::Float64 = 2.0
    σ_β::Float64 = 1.5
    prior_σ_team::ContinuousUnivariateDistribution = truncated(Normal(0, 0.5); lower = 0.0)
    prior_σ_gs::ContinuousUnivariateDistribution   = truncated(Normal(0, 0.5); lower = 0.0)
end

# Full design columns (always built; make_model selects the active subset per mode).
const ALL_X_COLS    = (:t_m, :t_m2, :is_home, :trailing, :leading, :man_adv, :log_pregame)
const X_STANDARDISE = (:t_m, :t_m2, :log_pregame)
const N_GAME_STATES = 7                          # clamp(goal_diff, -3, 3) -> 1..7

has_gs_hier(c::InPlayIntensityConfig) = c.game_state === :hier_replace || c.game_state === :hier_addon

"Indices into ALL_X_COLS that are active for this config (drop trailing/leading when not :linear/:addon)."
function active_cols(c::InPlayIntensityConfig)
    keep_dummies = c.game_state === :linear || c.game_state === :hier_addon
    return keep_dummies ? collect(1:7) : [1, 2, 3, 6, 7]   # drop trailing(4), leading(5)
end

# ---------------------------------------------------------------------------
# 2. Inputs builder (long format + team identity + game-state index + design matrix)
# ---------------------------------------------------------------------------

struct InPlayInputs
    y::Vector{Int}
    offset::Vector{Float64}
    X::Matrix{Float64}              # full 7-column standardised design
    trailing::Vector{Float64}
    leading::Vector{Float64}
    team_idx::Vector{Int}
    opp_idx::Vector{Int}
    gs_idx::Vector{Int}             # game-state index 1..N_GAME_STATES
    mom::Vector{Float64}            # standardised causal net momentum per row (0 where uncovered)
    n_teams::Int
    n_states::Int
    team_names::Vector{String}
    match_id::Vector{Int}
    x_cols::NTuple{7,Symbol}
    x_center::Vector{Float64}
    x_scale::Vector{Float64}
    mom_center::Float64
    mom_scale::Float64
end

function build_intensity_inputs(panel, ds; tmax = 80.0, mom_lookup = nothing, mom_decay = 0.03)
    D = build_intensity_dataset(panel, ds; tmax = tmax)
    teams = select(ds.matches, :match_id => ByRow(Int) => :match_id, :home_team, :away_team)
    D = leftjoin(D, teams, on = :match_id)
    dropmissing!(D, [:home_team, :away_team])
    D.team = [r.is_home == 1.0 ? String(r.home_team) : String(r.away_team) for r in eachrow(D)]
    D.opp  = [r.is_home == 1.0 ? String(r.away_team) : String(r.home_team) for r in eachrow(D)]

    names = sort(unique(vcat(D.team, D.opp)))
    tmap  = Dict(n => i for (i, n) in enumerate(names))

    X = Matrix{Float64}(undef, nrow(D), length(ALL_X_COLS))
    centers = zeros(length(ALL_X_COLS)); scales = ones(length(ALL_X_COLS))
    for (j, c) in enumerate(ALL_X_COLS)
        col = Float64.(D[!, c])
        if c in X_STANDARDISE
            centers[j] = mean(col); scales[j] = std(col) + 1e-9
        end
        X[:, j] = (col .- centers[j]) ./ scales[j]
    end
    gs_idx = clamp.(Int.(D.goal_diff), -3, 3) .+ 4    # -> 1..7

    # causal net momentum per row (l06 `row_net_momentum`); standardise. 0 / centre 0 when no lookup.
    if mom_lookup === nothing
        mom = zeros(nrow(D)); mc = 0.0; msd = 1.0
    else
        mom_raw = [row_net_momentum(mom_lookup, r.match_id, r.t_m, r.is_home; decay = mom_decay) for r in eachrow(D)]
        mc = mean(mom_raw); msd = std(mom_raw) + 1e-9
        mom = (mom_raw .- mc) ./ msd
    end

    return InPlayInputs(
        Vector{Int}(D.rem_goals), Vector{Float64}(D.logrem), X,
        Float64.(D.trailing), Float64.(D.leading),
        [tmap[t] for t in D.team], [tmap[o] for o in D.opp], gs_idx, mom,
        length(names), N_GAME_STATES, names, Vector{Int}(D.match_id),
        ALL_X_COLS, centers, scales, mc, msd,
    )
end

function subset_inputs(inp::InPlayInputs, ids)
    m = [mid in ids for mid in inp.match_id]
    InPlayInputs(inp.y[m], inp.offset[m], inp.X[m, :], inp.trailing[m], inp.leading[m],
                 inp.team_idx[m], inp.opp_idx[m], inp.gs_idx[m], inp.mom[m], inp.n_teams, inp.n_states,
                 inp.team_names, inp.match_id[m], inp.x_cols, inp.x_center, inp.x_scale,
                 inp.mom_center, inp.mom_scale)
end

# ---------------------------------------------------------------------------
# 3. The Turing model (AD-guide compliant)
# ---------------------------------------------------------------------------

@model function inplay_intensity(y, offset, X, team_idx, opp_idx, trailing, leading,
                                 gs_idx, mom, n_teams, n_states, config::InPlayIntensityConfig)
    α ~ Normal(0, config.σ_α)
    β ~ filldist(Normal(0, config.σ_β), size(X, 2))
    logλ = α .+ X * β                                    # global effects (active columns)

    if config.use_momentum
        β_mom ~ Normal(0, config.σ_β)
        logλ = logλ .+ β_mom .* mom                      # causal momentum covariate
    end
    if config.game_state === :hier_replace || config.game_state === :hier_addon
        σ_gs ~ config.prior_σ_gs
        z_gs ~ filldist(Normal(0, 1), n_states)
        logλ = logλ .+ view(z_gs .* σ_gs, gs_idx)        # global + δ_state[gs_idx]
    end
    if config.use_team_attack
        σ_att ~ config.prior_σ_team
        z_att ~ filldist(Normal(0, 1), n_teams)
        logλ = logλ .+ view(z_att .* σ_att, team_idx)
    end
    if config.use_team_defense
        σ_def ~ config.prior_σ_team
        z_def ~ filldist(Normal(0, 1), n_teams)
        logλ = logλ .- view(z_def .* σ_def, opp_idx)
    end
    if config.use_team_trailing
        σ_tr ~ config.prior_σ_team
        z_tr ~ filldist(Normal(0, 1), n_teams)
        logλ = logλ .+ (view(z_tr .* σ_tr, team_idx) .* trailing)
    end
    if config.use_team_leading
        σ_ld ~ config.prior_σ_team
        z_ld ~ filldist(Normal(0, 1), n_teams)
        logλ = logλ .+ (view(z_ld .* σ_ld, team_idx) .* leading)
    end

    logμ = clamp.(logλ .+ offset, -20.0, 20.0)
    μ    = exp.(logμ) .+ 1e-6
    @addlogprob! sum(logpdf.(Poisson.(μ), y))
end

"Instantiate the @model for a given inputs + config (selects the active design columns)."
function make_model(inp::InPlayInputs, config::InPlayIntensityConfig)
    Xa = inp.X[:, active_cols(config)]
    inplay_intensity(inp.y, inp.offset, Xa, inp.team_idx, inp.opp_idx,
                     inp.trailing, inp.leading, inp.gs_idx, inp.mom, inp.n_teams, inp.n_states, config)
end

# ---------------------------------------------------------------------------
# 4. Posterior prediction + held-out evaluation
# ---------------------------------------------------------------------------

_chainvec(chain, name) = vec(Array(chain[name]))
_chainmat(chain, base, k) = reduce(hcat, _chainvec(chain, Symbol("$base[$i]")) for i in 1:k)

"""
    posterior_logmu(chain, inp, config) -> Matrix (n_draws × n_rows) of log μ (mean remaining goals).
Rebuilds every effect from its draws honouring the config (game-state + team blocks).
"""
function posterior_logmu(chain, inp::InPlayInputs, config::InPlayIntensityConfig)
    Xa = inp.X[:, active_cols(config)]
    αv = _chainvec(chain, :α)
    B  = _chainmat(chain, :β, size(Xa, 2))
    base = αv .+ B * permutedims(Xa)                     # [S × N]
    if config.use_momentum
        base = base .+ _chainvec(chain, :β_mom) .* permutedims(inp.mom)
    end
    if has_gs_hier(config)
        eff = (_chainmat(chain, :z_gs, inp.n_states) .* _chainvec(chain, :σ_gs))   # [S × n_states]
        base = base .+ eff[:, inp.gs_idx]
    end
    if config.use_team_attack
        base = base .+ (_chainmat(chain, :z_att, inp.n_teams) .* _chainvec(chain, :σ_att))[:, inp.team_idx]
    end
    if config.use_team_defense
        base = base .- (_chainmat(chain, :z_def, inp.n_teams) .* _chainvec(chain, :σ_def))[:, inp.opp_idx]
    end
    if config.use_team_trailing
        base = base .+ (_chainmat(chain, :z_tr, inp.n_teams) .* _chainvec(chain, :σ_tr))[:, inp.team_idx] .* permutedims(inp.trailing)
    end
    if config.use_team_leading
        base = base .+ (_chainmat(chain, :z_ld, inp.n_teams) .* _chainvec(chain, :σ_ld))[:, inp.team_idx] .* permutedims(inp.leading)
    end
    return clamp.(base .+ permutedims(inp.offset), -20.0, 20.0)
end

"""
    held_out_elpd(chain, inp, config) -> (; elpd, per_obs)
Bayesian held-out score on the count target (log mean Poisson pmf over draws).
"""
function held_out_elpd(chain, inp::InPlayInputs, config::InPlayIntensityConfig)
    μ = exp.(posterior_logmu(chain, inp, config)) .+ 1e-6
    S, N = size(μ)
    elpd = 0.0
    for n in 1:N
        lp = logpdf.(Poisson.(@view μ[:, n]), inp.y[n])
        m = maximum(lp)
        elpd += m + log(sum(exp.(lp .- m))) - log(S)
    end
    return (; elpd, per_obs = elpd / N)
end

baseline_config() = InPlayIntensityConfig(game_state = :none)
