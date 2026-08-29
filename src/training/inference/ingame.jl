# src/training/inference/ingame.jl
#
# The zero-allocation live rate solver.
#
# An in-play goal-intensity model does not learn how good a team is. It learns a
# MULTIPLIER on a rate the pre-game model already estimated:
#
#     log λ_side(t) = log λ_side^pre + α + β·z(t)
#                                    + γ_state·(leading / trailing)
#                                    + γ_red·man_adv
#                                    + δ_time[bin]
#
# `λ^pre` is not a covariate. It is an OFFSET with a fixed coefficient of 1, so
# everything the in-play chain learnt is defined relative to whichever pre-game posterior
# produced it. `solve_ingame_rates!` therefore takes the pre-game container as an
# argument rather than reading one from anywhere: an in-game chain fitted against
# posterior A and priced six weeks later against posterior B is wrong by exactly the
# ratio of the two baselines — a few percent, invisible on a chart, surviving every
# convergence check, showing up only as a slow bleed in the P&L.
#
# WHY ZERO ALLOCATIONS HERE AND NOT IN THE PRE-GAME PATH
#
# The pre-game path prices a fixture once, hours before kickoff; a few kilobytes per call
# is beneath notice. The in-play path resolves the remaining intensity on EVERY posterior
# draw EVERY time the score, the man-count or the clock moves, for every live match at
# once. At 2,000 draws and a 20-match Saturday card, one `Vector` allocated per draw is
# hundreds of thousands of allocations per repricing tick.
#
# That is a LATENCY problem, not a throughput one: those allocations are what schedules
# the garbage collector, and a GC pause between seeing a price and acting on it is the
# whole cost. So the rules from `src/predictions/score_grids/kernels.jl` apply verbatim —
# every buffer allocated once by the caller and passed in; no `view` of a matrix, because
# a `SubArray` is a heap object unless escape analysis elides it and a zero-allocation
# claim resting on an optimisation that MIGHT fire is not a claim; scalar `A[i, k]`
# indexing, scalar accumulators, no broadcast, no closure, no comprehension, no `Ref`.
#
# SCOPE. This file graduates the rate solver only. Live MARKET pricing (`LiveBook`,
# `price_live_market!`) stays in `current_development/07_unified_inference_framework/`
# until `current_development/inplay_scottish` settles on a single integrator convention
# (`:flat` vs `:expo`); a market pricer belongs in `src/predictions/` in any case, and
# graduating it here would put a second pricing kernel outside that module.

# ==============================================================================
# 1. THE MATCH STATE
# ==============================================================================

"""
    MatchState(t, g_h, g_a, r_h, r_a)
    MatchState(; t = 0.0, g_h = 0, g_a = 0, r_h = 0, r_a = 0)

Everything the in-game intensity is conditioned on: the clock, the score, the red cards.
`isbits`, so it lives in registers and passing it costs nothing.

| field | is                               |
|-------|----------------------------------|
| `t`   | match minute now, in `[0, Tend]` |
| `g_h` | goals scored by home SO FAR      |
| `g_a` | goals scored by away SO FAR      |
| `r_h` | red cards shown to home          |
| `r_a` | red cards shown to away          |

`g_h`/`g_a` do two jobs and both matter: they set the game-state term (a trailing team
attacks harder) AND they are the base the remaining-goals distribution is added to when a
market is priced. Carrying them in one object is what stops the two from disagreeing.
"""
struct MatchState
    t::Float64
    g_h::Int
    g_a::Int
    r_h::Int
    r_a::Int
end

MatchState(; t::Real = 0.0, g_h::Integer = 0, g_a::Integer = 0,
             r_h::Integer = 0, r_a::Integer = 0) =
    MatchState(Float64(t), Int(g_h), Int(g_a), Int(r_h), Int(r_a))

"Kickoff: nothing has happened yet."
kickoff_state() = MatchState(0.0, 0, 0, 0, 0)

"Goal difference from the home side's point of view."
@inline goal_diff(s::MatchState) = s.g_h - s.g_a

"Man advantage for the home side: `+1` when the away team is a player down."
@inline man_advantage(s::MatchState) = Float64(s.r_a - s.r_h)

Base.show(io::IO, s::MatchState) = @printf(io, "MatchState(%.1f', %d-%d, reds %d-%d)",
                                           s.t, s.g_h, s.g_a, s.r_h, s.r_a)


# ==============================================================================
# 2. THE IN-GAME MODEL
# ==============================================================================

"""
    NHPPIntensityModel(; name = :nhpp_intensity, Δt = 5.0, Tend = 95.0, time_bins = true)

A non-homogeneous Poisson process for goal arrivals, conditioned on a pre-game rate.

    log λ_side(t) = log λ_side^pre + α + β·z(t) + γ_trail·1[trailing]
                                   + γ_lead·1[leading] + γ_red·man_adv + δ_time[bin]

with `z(t) = (t − 45)/45` the centred clock, and `δ_time` a per-bin offset over
`Δt`-minute bins of `[0, Tend]`.

`Tend = 95` rather than 90 on purpose, matching `inplay_scottish`'s `NHPPXConfig`: the
stoppage-time mass of a match whose minutes are clamped at 90 has to go somewhere, and
`[90, 95)` is where that convention puts it. A model fitted on one convention and
integrated on the other is off by the whole stoppage-time contribution.

The chain sites this reads are `α`, `β`, `γ_tr`, `γ_ld`, `γ_man`, and optionally
`z_time[b]` × `σ_time` — the same names `inplay_scottish/l01_nhpp_scottish.jl` samples,
so a chain fitted there replays here without translation.
"""
Base.@kwdef struct NHPPIntensityModel <: AbstractInGameModel
    name::Symbol = :nhpp_intensity
    Δt::Float64 = 5.0
    Tend::Float64 = 95.0
    time_bins::Bool = true
end

"Number of `Δt`-wide bins covering `[0, Tend]`."
n_time_bins(m::NHPPIntensityModel) = Int(cld(m.Tend, m.Δt))

Base.show(io::IO, m::NHPPIntensityModel) =
    print(io, "NHPPIntensityModel(", m.name, ", Δt=", m.Δt, ", Tend=", m.Tend,
          ", bins=", m.time_bins ? n_time_bins(m) : 0, ")")


# ==============================================================================
# 3. BUILDING THE WORKSPACE — posterior draws, flattened and paired
# ==============================================================================
#
# THE DRAW-PAIRING DECISION. The pre-game container has `n_draws(pre)` draws; the in-play
# chain has its own, usually a different number. Λ is a product of one draw from each, so
# they must be paired — and the pairing must be FIXED, not resampled per call, or two
# consecutive repricings of an unchanged match state return different prices, and a
# staking layer that sees a price move takes it as information.
#
# `build_ingame_workspace` therefore resolves the pairing ONCE, at construction, into
# dense vectors of length `n_target`, with a seeded RNG. The index vector is not kept:
# after that the hot path is a straight indexed read with no RNG, no indirection and no
# branch anywhere near it.

"""
    build_ingame_workspace(chain, model, n_target; seed = 20240601) -> IngameRatesWorkspace

Flatten an in-play `Chains` and pair it to `n_target` pre-game draws.

When the chain has exactly `n_target` draws they pair 1:1 in order. Otherwise one index
per target draw is sampled, once, from a seeded `Xoshiro` — see the section header for
why that happens here rather than at solve time.

A missing site becomes a ZERO vector, not an error: `γ_red` genuinely does not exist in a
chain fitted without a red-card term, and its correct contribution is `exp(0) = 1`. A
missing `α`, however, IS an error — a kernel with no level term is not a kernel.
"""
function build_ingame_workspace(chain::Chains, model::NHPPIntensityModel,
                                n_target::Integer; seed::Integer = 20240601)
    nd = Int(n_target)
    nd > 0 || error("build_ingame_workspace: n_target must be positive, got $nd.")

    site(s) = _inf_chain_vector(chain, s)
    α = site(:α)
    α === nothing && error(
        "build_ingame_workspace: the chain has no `α` site. An in-game kernel without a " *
        "level term has nothing to multiply the pre-game rate by.")

    src_n = length(α)
    idx = src_n == nd ? collect(1:nd) : rand(Xoshiro(Int(seed)), 1:src_n, nd)

    take(s) = (v = site(s); v === nothing ? zeros(nd) : v[idx])

    nb = n_time_bins(model)
    edges = collect(range(0.0, model.Tend; length = nb + 1))

    # Bin-major, because the hot loop walks bins for a fixed draw (see the container's
    # docstring in types.jl).
    δ_time = zeros(nb, nd)
    if model.time_bins
        z = _inf_chain_matrix(chain, :z_time, nb)
        σ = site(:σ_time)
        if z !== nothing
            sv = σ === nothing ? ones(src_n) : σ
            @inbounds for k in 1:nd, b in 1:nb
                δ_time[b, k] = z[idx[k], b] * sv[idx[k]]
            end
        end
    end

    return IngameRatesWorkspace(α[idx], take(:β), take(:γ_tr), take(:γ_ld), take(:γ_man),
                                δ_time, edges, model.Tend)
end

"""
    _inf_chain_vector(chain, site) -> Vector{Float64} or nothing

One scalar site, flattened over chains. `nothing` when the site is absent.

`MCMCChains` 7.7 removed `haskey(::Chains, ::Symbol)`, so membership is tested against
`names(chain)` rather than through the removed method.
"""
function _inf_chain_vector(chain::Chains, site::Symbol)
    try
        site in MCMCChains.names(chain) || return nothing
        return vec(Array(chain[site]))
    catch
        return nothing
    end
end

"An indexed vector site `base[1] … base[k]` as an `(n_draws × k)` matrix, or `nothing`."
function _inf_chain_matrix(chain::Chains, base::Symbol, k::Integer)
    cols = Vector{Vector{Float64}}()
    for i in 1:k
        v = _inf_chain_vector(chain, Symbol("$(base)[$i]"))
        v === nothing && return nothing
        push!(cols, v)
    end
    return reduce(hcat, cols)
end

"""
    alloc_live_rates(ws::IngameRatesWorkspace) -> LiveMatchRates

The destination `solve_ingame_rates!` writes into. One per worker, not one per call —
allocating it inside the tick loop is the thing this whole file exists to avoid.
"""
alloc_live_rates(ws::IngameRatesWorkspace) =
    LiveMatchRates(Vector{Float64}(undef, workspace_n_draws(ws)),
                   Vector{Float64}(undef, workspace_n_draws(ws)))


# ==============================================================================
# 4. THE SOLVER  —  Λ(t → Tend)
# ==============================================================================
#
# The integral of the instantaneous rate from now to full time, per posterior draw:
#
#     Λ_side(t) = λ_side^pre · Σ_bins exp(α + β·z_mid + γ_state + γ_red·man + δ[b]) · dt
#
# with `dt` the part of each bin still ahead of `t`. Bins entirely in the past contribute
# nothing; the bin containing `t` contributes its remainder.
#
# THE STATE IS HELD FIXED over the integration. This is a conditional forecast — "given
# it stays 1-0 with eleven a side" — not a simulation of the rest of the match. It is the
# right quantity for a market settling on the final score BECAUSE the alternative
# (integrating over future state paths) is what the market price already does, and
# double-counting it is how an in-play model talks itself into an edge that is not there.
#
# ALLOCATION BUDGET: 0 bytes.
#   * `α[k]`, `δ_time[b, k]` are scalar loads; no `view`, no slice.
#   * The destination is the caller's, written in place.
#   * The bin loop is the INNER loop and `δ_time` is bin-major, so it walks contiguous
#     memory.

"""
    solve_ingame_rates!(rates, ws, λ_pre_h, λ_pre_a, state) -> rates

Integrated remaining goal intensity per side, one entry per posterior draw, written into
the caller's `LiveMatchRates`. 0 bytes.

`λ_pre_h` / `λ_pre_a` are the pre-game per-draw rates for THIS fixture, in the same draw
order the workspace was paired to. The `CountLatents` method below is the one to call in
practice; this signature exists for a caller holding rates from somewhere else.
"""
function solve_ingame_rates!(rates::LiveMatchRates, ws::IngameRatesWorkspace,
                             λ_pre_h::AbstractVector{Float64},
                             λ_pre_a::AbstractVector{Float64},
                             s::MatchState)
    nd = workspace_n_draws(ws)
    length(rates) == nd || error(
        "solve_ingame_rates!: the destination holds $(length(rates)) draws but the " *
        "workspace has $nd. Use `alloc_live_rates(ws)`.")
    (length(λ_pre_h) >= nd && length(λ_pre_a) >= nd) || error(
        "solve_ingame_rates!: pre-game rates are shorter than the workspace's $nd draws.")

    Λ_h = rates.Λ_home
    Λ_a = rates.Λ_away
    nb = workspace_n_bins(ws)
    gd = goal_diff(s)
    man_h = man_advantage(s)
    man_a = -man_h
    trail_h = gd < 0
    lead_h = gd > 0
    t_now = s.t

    @inbounds for k in 1:nd
        αk = ws.α[k]
        βk = ws.β[k]
        st_h = (trail_h ? ws.γ_trail[k] : 0.0) + (lead_h ? ws.γ_lead[k] : 0.0)
        st_a = (lead_h ? ws.γ_trail[k] : 0.0) + (trail_h ? ws.γ_lead[k] : 0.0)
        rd_h = ws.γ_red[k] * man_h
        rd_a = ws.γ_red[k] * man_a

        acc_h = 0.0
        acc_a = 0.0
        for b in 1:nb
            lo = ws.edges[b]
            hi = ws.edges[b + 1]
            hi <= t_now && continue
            dt = hi - (lo > t_now ? lo : t_now)
            dt <= 0.0 && continue
            zc = (0.5 * (lo + hi) - 45.0) / 45.0
            base = αk + βk * zc + ws.δ_time[b, k]
            acc_h += exp(base + st_h + rd_h) * dt
            acc_a += exp(base + st_a + rd_a) * dt
        end
        Λ_h[k] = λ_pre_h[k] * acc_h
        Λ_a[k] = λ_pre_a[k] * acc_a
    end
    return rates
end

"""
    solve_ingame_rates!(rates, ws, pre::CountLatents, i, state) -> rates

The same, reading fixture row `i`'s pre-game rates straight out of the typed container.
0 bytes — and that is the reason this is a separate method rather than a `view` at the
call site: `view(pre.λ_home, i, :)` is a `SubArray`, which is a heap object the escape
analyser is not obliged to elide.
"""
function solve_ingame_rates!(rates::LiveMatchRates, ws::IngameRatesWorkspace,
                             pre::CountLatents{Float64}, i::Int, s::MatchState)
    nd = workspace_n_draws(ws)
    length(rates) == nd || error(
        "solve_ingame_rates!: the destination holds $(length(rates)) draws but the " *
        "workspace has $nd. Use `alloc_live_rates(ws)`.")
    nd == n_draws(pre) || error(
        "solve_ingame_rates!: the workspace has $nd draws and the pre-game container " *
        "has $(n_draws(pre)). Build it with `build_ingame_workspace(chain, model, " *
        "n_draws(pre))` so the two posteriors are paired.")
    1 <= i <= n_matches(pre) || error(
        "solve_ingame_rates!: fixture row $i is outside 1:$(n_matches(pre)).")

    Λ_h = rates.Λ_home
    Λ_a = rates.Λ_away
    λh = getfield(pre, :λ_home)
    λa = getfield(pre, :λ_away)
    nb = workspace_n_bins(ws)
    gd = goal_diff(s)
    man_h = man_advantage(s)
    man_a = -man_h
    trail_h = gd < 0
    lead_h = gd > 0
    t_now = s.t

    @inbounds for k in 1:nd
        αk = ws.α[k]
        βk = ws.β[k]
        st_h = (trail_h ? ws.γ_trail[k] : 0.0) + (lead_h ? ws.γ_lead[k] : 0.0)
        st_a = (lead_h ? ws.γ_trail[k] : 0.0) + (trail_h ? ws.γ_lead[k] : 0.0)
        rd_h = ws.γ_red[k] * man_h
        rd_a = ws.γ_red[k] * man_a

        acc_h = 0.0
        acc_a = 0.0
        for b in 1:nb
            lo = ws.edges[b]
            hi = ws.edges[b + 1]
            hi <= t_now && continue
            dt = hi - (lo > t_now ? lo : t_now)
            dt <= 0.0 && continue
            zc = (0.5 * (lo + hi) - 45.0) / 45.0
            base = αk + βk * zc + ws.δ_time[b, k]
            acc_h += exp(base + st_h + rd_h) * dt
            acc_a += exp(base + st_a + rd_a) * dt
        end
        Λ_h[k] = λh[i, k] * acc_h
        Λ_a[k] = λa[i, k] * acc_a
    end
    return rates
end

"""
    solve_ingame_rates(ws, pre::CountLatents, i, state) -> LiveMatchRates

Allocating form, for the REPL. The hot path is [`solve_ingame_rates!`](@ref).
"""
solve_ingame_rates(ws::IngameRatesWorkspace, pre::CountLatents{Float64}, i::Integer,
                   s::MatchState) =
    solve_ingame_rates!(alloc_live_rates(ws), ws, pre, Int(i), s)
