#=
l08_nhpp_turing.jl  —  Bayesian Non-Homogeneous Poisson Process (NHPP) in-play scoring model.

WHY: the l02/l03 count model assumes a CONSTANT rate over the remaining window (μ = rate × (90−t)/90),
which under-predicts late-game goals (validated bias +0.108 ± 0.032, 3.4σ, in the 75–88' bin — see the
MLE diagnostic). The NHPP models a genuinely time-varying intensity λ(t) and integrates it properly
(incl. stoppage), which removes that bias.

KEY TRICK — discretised NHPP = Poisson regression on fine time-bins:
  Split each match into Δt-minute slices. For each (match × side × slice):
      y      = goals that side scored in the slice
      offset = log(Δt)                       # exposure
      log λ  = α + log(pregame_λ_side) + β·z(t) + [hier time] + [hier team] + [hier state]
      y ~ Poisson(λ · Δt) = Poisson(exp(log λ + offset))
  Σ over slices of the Poisson log-lik = Σ_goals log λ(t_g) − ∫λ dt (the NHPP likelihood) as Δt → 0.
  This is AD-safe (pure Poisson, broadcast) and handles the piecewise game state automatically (each
  slice carries its own running score).

HIERARCHIES (all toggleable, non-centered `z~filldist(Normal(0,1),n); θ=z.*σ`, AD-guide compliant):
  - hier_time  : global β·z(t) + δ_time[time_bin]      → flexible intensity SHAPE (the "different times")
  - hier_team  : δ_team[team]                          → team-specific in-play scoring level
  - hier_state : δ_state[goal_diff bucket]             → partial-pooled game-state effect

Follows docs/turing_ad_performance_guide.md: broadcast + view, no scalar loops in @model, config branches
on CONSTANT flags only, clamp + 1e-6. Reuses goal times from `ds.incidents` and pregame λ from the panel.
=#

using Turing, Distributions, DataFrames, Statistics, LinearAlgebra, MCMCChains
const Samplers = BayesianFootball.Samplers

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
Base.@kwdef struct NHPPConfig
    hier_time::Bool  = true
    hier_team::Bool  = false
    hier_state::Bool = true
    Δt::Float64      = 5.0      # slice width (minutes)
    Tend::Float64    = 95.0     # integrate through stoppage
    σ_α::Float64     = 2.0
    σ_β::Float64     = 1.0
    σ_state_prior::ContinuousUnivariateDistribution = truncated(Normal(0, 0.5); lower = 0.0)
    σ_time_prior::ContinuousUnivariateDistribution  = truncated(Normal(0, 0.5); lower = 0.0)
    σ_team_prior::ContinuousUnivariateDistribution  = truncated(Normal(0, 0.4); lower = 0.0)
end

# ---------------------------------------------------------------------------
# 2. Dataset builder — (match × side × time-slice) long format
# ---------------------------------------------------------------------------
struct NHPPInputs
    y::Vector{Int}
    offset::Vector{Float64}
    z::Vector{Float64}              # centred match-time (t_mid − 45)/45
    log_pg::Vector{Float64}         # log pregame λ of the scoring side
    trailing::Vector{Float64}
    leading::Vector{Float64}
    team_idx::Vector{Int}
    gs_idx::Vector{Int}             # clamp(goal_diff,-3,3)+4 ∈ 1..7
    time_idx::Vector{Int}           # slice index 1..n_timebins
    n_teams::Int
    n_states::Int
    n_timebins::Int
    team_names::Vector{String}
    match_id::Vector{Int}
end

"""
    build_nhpp_inputs(matchseq, pgmap, teammap; Δt, Tend) -> NHPPInputs

`matchseq`  :: Vector of (pgh, pga, goals=Vector{(t, home::Bool)} sorted) per match (own goals already
               credited to the scoring side).  `pgmap` :: match_id => (pgλ_h, pgλ_a).
`teammap`   :: match_id => (home_team::String, away_team::String).
One row per (match, side, slice). Running game state is taken at the slice MID-point.
"""
function build_nhpp_inputs(matchseq, pgmap, teammap, mids; Δt = 5.0, Tend = 95.0)
    edges = collect(0.0:Δt:Tend); nb = length(edges) - 1
    mids_used = [m for m in mids if haskey(teammap, m)]
    names = sort(unique(vcat([teammap[m][1] for m in mids_used], [teammap[m][2] for m in mids_used])))
    tmap = Dict(n => i for (i, n) in enumerate(names))

    y = Int[]; off = Float64[]; zc = Float64[]; lpg = Float64[]
    tr = Float64[]; ld = Float64[]; tid = Int[]; gsi = Int[]; tidx = Int[]; mvec = Int[]

    for (mseq, mid) in zip(matchseq, mids)
        haskey(teammap, mid) || continue
        ht, at = teammap[mid]; pgh, pga = mseq.pgh, mseq.pga
        gtimes = mseq.goals
        for b in 1:nb
            lo, hi = edges[b], edges[b+1]; tmid = (lo + hi) / 2
            # running score at slice start (goals strictly before lo)
            gh = count(g -> g.home && g.t < lo, gtimes); ga = count(g -> !g.home && g.t < lo, gtimes)
            yh = count(g ->  g.home && lo <= g.t < hi, gtimes)
            ya = count(g -> !g.home && lo <= g.t < hi, gtimes)
            gd_h = gh - ga
            for (ih, yc, pg, team, gdside) in ((true, yh, pgh, ht, gd_h), (false, ya, pga, at, -gd_h))
                push!(y, yc); push!(off, log(Δt)); push!(zc, (tmid - 45) / 45)
                push!(lpg, log(pg)); push!(tr, Float64(gdside < 0)); push!(ld, Float64(gdside > 0))
                push!(tid, tmap[team]); push!(gsi, clamp(gdside, -3, 3) + 4); push!(tidx, b); push!(mvec, mid)
            end
        end
    end
    NHPPInputs(y, off, zc, lpg, tr, ld, tid, gsi, tidx, length(names), 7, nb, names, mvec)
end

# ---------------------------------------------------------------------------
# 3. The Turing model (AD-guide compliant)
# ---------------------------------------------------------------------------
@model function nhpp_intensity(y, offset, z, log_pg, trailing, leading, team_idx, gs_idx, time_idx,
                               n_teams, n_states, n_timebins, config::NHPPConfig)
    α    ~ Normal(0, config.σ_α)
    β    ~ Normal(0, config.σ_β)              # global linear time drift
    γ_tr ~ Normal(0, 0.5)
    γ_ld ~ Normal(0, 0.5)
    logλ = α .+ log_pg .+ β .* z .+ γ_tr .* trailing .+ γ_ld .* leading

    if config.hier_time                        # global + δ_time[bin]: flexible intensity shape
        σ_time ~ config.σ_time_prior
        z_time ~ filldist(Normal(0, 1), n_timebins)
        logλ = logλ .+ view(z_time .* σ_time, time_idx)
    end
    if config.hier_team                        # team-specific in-play level
        σ_team ~ config.σ_team_prior
        z_team ~ filldist(Normal(0, 1), n_teams)
        logλ = logλ .+ view(z_team .* σ_team, team_idx)
    end
    if config.hier_state                       # partial-pooled game-state
        σ_gs ~ config.σ_state_prior
        z_gs ~ filldist(Normal(0, 1), n_states)
        logλ = logλ .+ view(z_gs .* σ_gs, gs_idx)
    end

    μ = exp.(clamp.(logλ .+ offset, -20.0, 20.0)) .+ 1e-6
    @addlogprob! sum(logpdf.(Poisson.(μ), y))
end

make_nhpp_model(inp::NHPPInputs, c::NHPPConfig) =
    nhpp_intensity(inp.y, inp.offset, inp.z, inp.log_pg, inp.trailing, inp.leading,
                   inp.team_idx, inp.gs_idx, inp.time_idx, inp.n_teams, inp.n_states, inp.n_timebins, c)

# ---------------------------------------------------------------------------
# 4. Posterior intensity + expected remaining goals (for the residual test / hedge)
# ---------------------------------------------------------------------------
_cv(ch, s) = vec(Array(ch[s]))
_cm(ch, base, k) = reduce(hcat, _cv(ch, Symbol("$base[$i]")) for i in 1:k)

"""
    expected_remaining(chain, c; pg_h, pg_a, gh, ga, t_now, Tend) -> posterior vector of E[remaining goals]

Sums the posterior intensity over slices from `t_now` to `Tend`, holding the CURRENT score fixed (same
convention as the count-model residual test). Returns one value per posterior draw.
"""
function expected_remaining(chain, c::NHPPConfig; pg_h, pg_a, gh, ga, t_now, Tend = c.Tend)
    αv = _cv(chain, :α); βv = _cv(chain, :β); gtr = _cv(chain, :γ_tr); gld = _cv(chain, :γ_ld)
    nb = Int(cld(Tend, c.Δt)); edges = collect(0.0:c.Δt:Tend)
    gd_h = gh - ga
    zt = has_hier_time(chain) ? (_cm(chain, :z_time, length(edges)-1) .* _cv(chain, :σ_time)) : nothing
    zg = has_hier_state(chain) ? (_cm(chain, :z_gs, 7) .* _cv(chain, :σ_gs)) : nothing
    Λh = zeros(length(αv)); Λa = zeros(length(αv))
    for b in 1:(length(edges)-1)
        lo, hi = edges[b], edges[b+1]; hi <= t_now && continue
        tmid = (lo + hi)/2; z = (tmid - 45)/45; dt = hi - max(lo, t_now)
        baseh = αv .+ log(pg_h) .+ βv .* z .+ gtr .* (gd_h < 0) .+ gld .* (gd_h > 0)
        basea = αv .+ log(pg_a) .+ βv .* z .+ gtr .* (-gd_h < 0) .+ gld .* (-gd_h > 0)
        if zt !== nothing; baseh = baseh .+ zt[:, b]; basea = basea .+ zt[:, b]; end
        if zg !== nothing
            baseh = baseh .+ zg[:, clamp(gd_h,-3,3)+4]; basea = basea .+ zg[:, clamp(-gd_h,-3,3)+4]
        end
        Λh = Λh .+ exp.(baseh) .* dt; Λa = Λa .+ exp.(basea) .* dt
    end
    Λh .+ Λa
end

has_hier_time(ch)  = any(occursin.("z_time", string.(MCMCChains.names(ch, :parameters))))
has_hier_state(ch) = any(occursin.("z_gs",   string.(MCMCChains.names(ch, :parameters))))
has_hier_team(ch)  = any(occursin.("z_team", string.(MCMCChains.names(ch, :parameters))))
