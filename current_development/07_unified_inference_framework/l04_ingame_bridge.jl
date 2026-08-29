# ==============================================================================
# 07 — UNIFIED INFERENCE FRAMEWORK : THE IN-GAME CONDITIONAL BRIDGE
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# ------------------------------------------------------------------------------
# THE PROBLEM THIS SOLVES
# ------------------------------------------------------------------------------
#
# An in-play goal-intensity model does not learn how good a team is. It learns a
# MULTIPLIER on a rate the pre-game model already estimated:
#
#     log λ_side(t) = log λ_side^pre + α + β·z(t)
#                                    + γ_state·(leading / trailing)
#                                    + γ_red·man_adv
#                                    + δ_time[bin]
#
# The `λ^pre` term is not a covariate. It is an OFFSET with a fixed coefficient of 1,
# and everything the in-play chain learnt is defined relative to whichever pre-game
# posterior produced it. `current_development/inplay_scottish/l09_ingame.jl` already
# lives with this: its `InGameModel` carries a `PregameDraws` field precisely so that
# the two travel together, and its docstring warns that mixing an `:expo`-fitted chain
# with `:flat` integration "silently misprices".
#
# There is currently no CONTRACT that makes the pairing structural. `InGameFitConfig`
# holds its `pregame` source as a field, and `InGameFit` stores the resolved container,
# so a fit that was never given a baseline cannot be constructed, and a fit that was
# can always say which one.
#
# The failure this prevents is the quiet kind. An in-game chain fitted against pre-game
# posterior A and priced six weeks later against posterior B is wrong by exactly the
# ratio of the two baselines — a few percent — which is invisible on a chart, survives
# every convergence check, and shows up only as a slow bleed in the P&L.
#
# ------------------------------------------------------------------------------
# ZERO ALLOCATIONS, AND WHY IT IS NOT AN OPTIMISATION HERE
# ------------------------------------------------------------------------------
#
# `remaining_intensity!` and `price_live_market!` allocate 0 bytes.
#
# The pre-game path prices a fixture once, hours before kickoff; a few kilobytes per
# call is beneath notice. The in-play path prices EVERY market on EVERY posterior draw
# EVERY time the score, the man-count, or the clock moves — for every live match at
# once. At 2,000 draws, six markets and a 20-match Saturday afternoon card, a single
# `Vector` allocated per draw is ~1.4 million allocations per repricing tick.
#
# That is not a throughput problem. It is a LATENCY problem: those allocations are what
# schedules the garbage collector, and a GC pause in the middle of a repricing tick is
# a pause between seeing a price and being able to act on it.
#
# So the rules from `06_typed_posterior_latents/l03_score_grids.jl` apply verbatim:
#
#   * every buffer is allocated once, by the caller, and passed in;
#   * no `view` of a matrix — a `SubArray` is a heap object unless escape analysis
#     elides it, and a zero-allocation claim resting on an optimisation that MIGHT fire
#     is not a claim;
#   * scalar `A[i, k]` indexing, scalar accumulators, no broadcast, no closure, no
#     comprehension, no `Ref`;
#   * `Poisson(λ)` is constructed inside the loop and costs nothing — it is `isbits`,
#     and `pdf` on it does not allocate.
#
# `r01_demo.jl` §7 measures this with `@allocated` against an empty-closure baseline
# rather than asserting it.
#
# ------------------------------------------------------------------------------
# THE DRAW-PAIRING DECISION
# ------------------------------------------------------------------------------
#
# The pre-game container has `n_draws(pre)` draws. The in-play chain has its own,
# usually a different number. Λ is a product of one draw from each, so they must be
# paired — and the pairing must be FIXED, not resampled per call, or two consecutive
# repricings of an unchanged match state return different prices.
#
# `build_live_kernel` therefore resolves the pairing ONCE, at construction, into dense
# vectors of length `n_draws(pre)`, with a seeded RNG. After that the hot path is a
# straight indexed read and there is no RNG anywhere near it. `LiveKernel`'s constructor
# refuses a length that does not match the container it will be used with, so the
# mismatch is a construction-time error rather than a silently recycled index.
#
# ==============================================================================

using Distributions
using LinearAlgebra
using MCMCChains
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "l03_engine.jl"))

using BayesianFootball.Data: Market1X2, MarketBTTS, MarketOverUnder, outcomes


# ==============================================================================
# 1. THE MATCH STATE
# ==============================================================================

"""
    MatchState(t, g_h, g_a, r_h, r_a)

Everything the in-game intensity is conditioned on: the clock, the score, the red
cards. `isbits`, so it lives in a register and passing it costs nothing.

| field | is                                        |
|-------|-------------------------------------------|
| `t`   | match minute now, in `[0, Tend]`          |
| `g_h` | goals scored by home SO FAR               |
| `g_a` | goals scored by away SO FAR               |
| `r_h` | red cards shown to home                   |
| `r_a` | red cards shown to away                   |

`g_h`/`g_a` do two jobs and both matter: they set the game-state term (a trailing team
attacks harder) AND they are the base the remaining-goals distribution is added to when
a market is priced. Carrying them in one object is what stops the two from disagreeing.
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

"Man advantage for the home side: +1 when the away team is a player down."
@inline man_advantage(s::MatchState) = Float64(s.r_a - s.r_h)

Base.show(io::IO, s::MatchState) = @printf(io, "MatchState(%.1f', %d-%d, reds %d-%d)",
                                           s.t, s.g_h, s.g_a, s.r_h, s.r_a)


# ==============================================================================
# 2. THE IN-GAME MODEL
# ==============================================================================

"""
    NHPPIntensityModel(; name, Δt = 5.0, Tend = 95.0, time_bins = true)

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
so a chain fitted there can be replayed here without translation.
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
# 3. THE LIVE KERNEL — posterior draws, flattened and paired
# ==============================================================================

"""
    LiveKernel

The in-play chain's posterior, materialised as dense vectors ALREADY PAIRED to a
pre-game container's draw count.

| field     | shape              | meaning                                     |
|-----------|--------------------|---------------------------------------------|
| `α`       | `n_draws`          | log multiplier at `t = 45`, level term       |
| `β`       | `n_draws`          | slope on the centred clock `z(t)`            |
| `γ_trail` | `n_draws`          | log multiplier while trailing                |
| `γ_lead`  | `n_draws`          | log multiplier while leading                 |
| `γ_red`   | `n_draws`          | log multiplier per man of advantage          |
| `δ_time`  | `n_bins × n_draws` | per-bin offset; a zero matrix disables it    |
| `edges`   | `n_bins + 1`       | bin boundaries over `[0, Tend]`              |

`δ_time` is stored `(bins × draws)` — bin-major — because the hot loop walks BINS for a
fixed draw. That is the contiguous direction in a column-major array, and it is the
opposite convention to `06`'s latent matrices for the opposite reason: there, the
sweep is over fixtures at a fixed draw.

Every field is a `Vector`/`Matrix` of exactly `n_draws` columns. The constructor
enforces it, so a mismatch between the in-play posterior and the pre-game one is caught
here rather than becoming a recycled index inside the kernel.
"""
struct LiveKernel
    α::Vector{Float64}
    β::Vector{Float64}
    γ_trail::Vector{Float64}
    γ_lead::Vector{Float64}
    γ_red::Vector{Float64}
    δ_time::Matrix{Float64}
    edges::Vector{Float64}
    Tend::Float64

    function LiveKernel(α, β, γ_trail, γ_lead, γ_red, δ_time, edges, Tend)
        nd = length(α)
        for (nm, v) in ((:β, β), (:γ_trail, γ_trail), (:γ_lead, γ_lead), (:γ_red, γ_red))
            length(v) == nd || error(
                "LiveKernel: $nm has $(length(v)) draws but α has $nd. Every site must " *
                "come from the same paired posterior sweep.")
        end
        size(δ_time, 2) == nd || error(
            "LiveKernel: δ_time has $(size(δ_time, 2)) draw columns but α has $nd.")
        size(δ_time, 1) == length(edges) - 1 || error(
            "LiveKernel: δ_time has $(size(δ_time, 1)) bin rows but `edges` describes " *
            "$(length(edges) - 1) bins.")
        all(isfinite, α) && all(isfinite, β) || error(
            "LiveKernel: non-finite α or β — exp() of it is Inf and every price NaN.")
        return new(Vector{Float64}(α), Vector{Float64}(β),
                   Vector{Float64}(γ_trail), Vector{Float64}(γ_lead),
                   Vector{Float64}(γ_red), Matrix{Float64}(δ_time),
                   Vector{Float64}(edges), Float64(Tend))
    end
end

kernel_n_draws(K::LiveKernel) = length(K.α)
kernel_n_bins(K::LiveKernel) = size(K.δ_time, 1)

Base.show(io::IO, K::LiveKernel) =
    print(io, "LiveKernel(", kernel_n_draws(K), " draws, ", kernel_n_bins(K),
          " bins to ", K.Tend, "')")

"""
    build_live_kernel(chain, model, n_target; seed = 20240601) -> LiveKernel

Flatten an in-play `Chains` and pair it to `n_target` pre-game draws.

PAIRING. When the chain has exactly `n_target` draws they pair 1:1 in order. Otherwise
one index per target draw is sampled, once, from a seeded `Xoshiro`. Sampling at
CONSTRUCTION rather than at pricing time is what makes two repricings of an unchanged
match state return the same number — a property that matters more than it sounds,
because a staking layer that sees a price move takes it as information.

The index vector is not kept. Once the sites are materialised the pairing is baked into
the vectors, and the hot path holds no RNG, no index indirection and no branch.

A missing site is a ZERO vector, not an error: `γ_red` genuinely does not exist in a
chain fitted without a red-card term, and its correct contribution is `exp(0) = 1`.
A missing `α`, however, IS an error — a kernel with no level term is not a kernel.
"""
function build_live_kernel(chain::Chains, model::NHPPIntensityModel, n_target::Integer;
                           seed::Integer = 20240601)
    nd = Int(n_target)
    nd > 0 || error("build_live_kernel: n_target must be positive, got $nd.")

    site(s) = _uif_chain_vector(chain, s)
    α = site(:α)
    α === nothing && error(
        "build_live_kernel: the chain has no `α` site. An in-game kernel without a " *
        "level term has nothing to multiply the pre-game rate by.")

    src_n = length(α)
    idx = src_n == nd ? collect(1:nd) :
          rand(Xoshiro(Int(seed)), 1:src_n, nd)

    take(s) = (v = site(s); v === nothing ? zeros(nd) : v[idx])

    nb = n_time_bins(model)
    edges = collect(range(0.0, model.Tend; length = nb + 1))

    # Bin-major, because the hot loop walks bins for a fixed draw (§3).
    δ_time = zeros(nb, nd)
    if model.time_bins
        z = _uif_chain_matrix(chain, :z_time, nb)
        σ = site(:σ_time)
        if z !== nothing
            sv = σ === nothing ? ones(src_n) : σ
            @inbounds for k in 1:nd, b in 1:nb
                δ_time[b, k] = z[idx[k], b] * sv[idx[k]]
            end
        end
    end

    return LiveKernel(α[idx], take(:β), take(:γ_tr), take(:γ_ld), take(:γ_man),
                      δ_time, edges, model.Tend)
end

"""
    _uif_chain_vector(chain, site) -> Vector{Float64} or nothing

One scalar site, flattened over chains. `nothing` when the site is absent.

`MCMCChains` 7.7 removed `haskey(::Chains, ::Symbol)` (see
`06_typed_posterior_latents/README.md`, defect 3), so membership is tested against
`names(chain)` rather than through the removed method.
"""
function _uif_chain_vector(chain::Chains, site::Symbol)
    try
        site in MCMCChains.names(chain) || return nothing
        return vec(Array(chain[site]))
    catch
        return nothing
    end
end

"An indexed vector site `base[1] … base[k]` as an `(n_draws × k)` matrix, or `nothing`."
function _uif_chain_matrix(chain::Chains, base::Symbol, k::Integer)
    cols = Vector{Vector{Float64}}()
    for i in 1:k
        v = _uif_chain_vector(chain, Symbol("$(base)[$i]"))
        v === nothing && return nothing
        push!(cols, v)
    end
    return reduce(hcat, cols)
end


# ==============================================================================
# 4. REMAINING INTENSITY  —  Λ(t → Tend)
# ==============================================================================
#
# The integral of the instantaneous rate from now to full time, per posterior draw:
#
#     Λ_side(t) = λ_side^pre · Σ_bins exp(α + β·z_mid + γ_state + γ_red·man + δ[b]) · dt
#
# with `dt` the part of each bin that is still ahead of `t`. Bins entirely in the past
# contribute nothing; the bin containing `t` contributes its remainder.
#
# THE STATE IS HELD FIXED over the integration. This is a conditional forecast — "given
# it stays 1-0 with eleven a side" — not a simulation of the rest of the match. It is
# the right quantity for a market that settles on the final score BECAUSE the alternative
# (integrating over future state paths) is what the market price already does, and
# double-counting it is how an in-play model talks itself into an edge that isn't there.
#
# ALLOCATION BUDGET: 0 bytes.
#   * `Poisson` is never constructed here.
#   * `α[k]`, `δ_time[b, k]` are scalar loads; no `view`, no slice.
#   * `Λ_h` and `Λ_a` are the caller's, written in place.
#   * The bin loop is the INNER loop and `δ_time` is bin-major, so it walks contiguous
#     memory.

"""
    remaining_intensity!(Λ_h, Λ_a, K, λ_pre_h, λ_pre_a, state) -> (Λ_h, Λ_a)

Integrated remaining goal intensity per side, one entry per posterior draw, written
into the caller's vectors. 0 bytes.

`λ_pre_h` / `λ_pre_a` are the pre-game per-draw rates for THIS fixture, in the same
draw order the kernel was paired to. The `CountLatents` overload below is the one to
call in practice; this signature exists for a caller holding rates from somewhere else.
"""
function remaining_intensity!(Λ_h::Vector{Float64}, Λ_a::Vector{Float64},
                              K::LiveKernel,
                              λ_pre_h::AbstractVector{Float64},
                              λ_pre_a::AbstractVector{Float64},
                              s::MatchState)
    nd = kernel_n_draws(K)
    (length(Λ_h) == nd && length(Λ_a) == nd) || error(
        "remaining_intensity!: destinations are $(length(Λ_h))/$(length(Λ_a)) long but " *
        "the kernel has $nd draws. Use `alloc_intensity(K)`.")
    (length(λ_pre_h) >= nd && length(λ_pre_a) >= nd) || error(
        "remaining_intensity!: pre-game rates are shorter than the kernel's $nd draws.")

    nb = kernel_n_bins(K)
    gd = goal_diff(s)
    man_h = man_advantage(s)
    man_a = -man_h
    trail_h = gd < 0
    lead_h = gd > 0
    t_now = s.t

    @inbounds for k in 1:nd
        αk = K.α[k]
        βk = K.β[k]
        st_h = (trail_h ? K.γ_trail[k] : 0.0) + (lead_h ? K.γ_lead[k] : 0.0)
        st_a = (lead_h ? K.γ_trail[k] : 0.0) + (trail_h ? K.γ_lead[k] : 0.0)
        rd_h = K.γ_red[k] * man_h
        rd_a = K.γ_red[k] * man_a

        acc_h = 0.0
        acc_a = 0.0
        for b in 1:nb
            lo = K.edges[b]
            hi = K.edges[b + 1]
            hi <= t_now && continue
            dt = hi - (lo > t_now ? lo : t_now)
            dt <= 0.0 && continue
            zc = (0.5 * (lo + hi) - 45.0) / 45.0
            base = αk + βk * zc + K.δ_time[b, k]
            acc_h += exp(base + st_h + rd_h) * dt
            acc_a += exp(base + st_a + rd_a) * dt
        end
        Λ_h[k] = λ_pre_h[k] * acc_h
        Λ_a[k] = λ_pre_a[k] * acc_a
    end
    return (Λ_h, Λ_a)
end

"""
    remaining_intensity!(Λ_h, Λ_a, K, pre::CountLatents, i, state) -> (Λ_h, Λ_a)

The same, reading fixture row `i`'s pre-game rates straight out of the typed container.
0 bytes — and this is the reason it is a separate method rather than a `view` at the
call site: `view(pre.λ_home, i, :)` is a `SubArray`, which is a heap object the escape
analyser is not obliged to elide.
"""
function remaining_intensity!(Λ_h::Vector{Float64}, Λ_a::Vector{Float64},
                              K::LiveKernel, pre::CountLatents{Float64}, i::Int,
                              s::MatchState)
    nd = kernel_n_draws(K)
    (length(Λ_h) == nd && length(Λ_a) == nd) || error(
        "remaining_intensity!: destinations are $(length(Λ_h))/$(length(Λ_a)) long but " *
        "the kernel has $nd draws. Use `alloc_intensity(K)`.")
    nd == n_draws(pre) || error(
        "remaining_intensity!: the kernel has $nd draws and the pre-game container has " *
        "$(n_draws(pre)). Build the kernel with `build_live_kernel(chain, model, " *
        "n_draws(pre))` so the two posteriors are paired.")
    1 <= i <= n_matches(pre) || error(
        "remaining_intensity!: fixture row $i is outside 1:$(n_matches(pre)).")

    λh = getfield(pre, :λ_home)
    λa = getfield(pre, :λ_away)
    nb = kernel_n_bins(K)
    gd = goal_diff(s)
    man_h = man_advantage(s)
    man_a = -man_h
    trail_h = gd < 0
    lead_h = gd > 0
    t_now = s.t

    @inbounds for k in 1:nd
        αk = K.α[k]
        βk = K.β[k]
        st_h = (trail_h ? K.γ_trail[k] : 0.0) + (lead_h ? K.γ_lead[k] : 0.0)
        st_a = (lead_h ? K.γ_trail[k] : 0.0) + (trail_h ? K.γ_lead[k] : 0.0)
        rd_h = K.γ_red[k] * man_h
        rd_a = K.γ_red[k] * man_a

        acc_h = 0.0
        acc_a = 0.0
        for b in 1:nb
            lo = K.edges[b]
            hi = K.edges[b + 1]
            hi <= t_now && continue
            dt = hi - (lo > t_now ? lo : t_now)
            dt <= 0.0 && continue
            zc = (0.5 * (lo + hi) - 45.0) / 45.0
            base = αk + βk * zc + K.δ_time[b, k]
            acc_h += exp(base + st_h + rd_h) * dt
            acc_a += exp(base + st_a + rd_a) * dt
        end
        Λ_h[k] = λh[i, k] * acc_h
        Λ_a[k] = λa[i, k] * acc_a
    end
    return (Λ_h, Λ_a)
end

"A pair of destination vectors for `remaining_intensity!`. One per worker, not per call."
alloc_intensity(K::LiveKernel) =
    (Vector{Float64}(undef, kernel_n_draws(K)), Vector{Float64}(undef, kernel_n_draws(K)))

"""
    remaining_intensity(K, pre, i, state) -> (Λ_h, Λ_a)

Allocating form, for the REPL. The hot path is `remaining_intensity!`.
"""
function remaining_intensity(K::LiveKernel, pre::CountLatents{Float64}, i::Integer,
                             s::MatchState)
    Λ_h, Λ_a = alloc_intensity(K)
    return remaining_intensity!(Λ_h, Λ_a, K, pre, Int(i), s)
end


# ==============================================================================
# 5. LIVE MARKET PRICING
# ==============================================================================
#
# A live price is NOT the pre-game price recomputed with a smaller λ. The goals already
# scored are certain, and they shift the whole distribution:
#
#     final_h = g_h + N_h,   N_h ~ Poisson(Λ_h)
#
# So 1X2 asks `P(g_h + N_h > g_a + N_a)`, BTTS asks `P(g_h + N_h > 0 ∧ g_a + N_a > 0)`
# — which is 1 when both sides have already scored, whatever Λ is — and Over/Under
# compares `g_h + g_a + N_h + N_a` against the line.
#
# The pre-game kernels in `06/l03_score_grids.jl` cannot express any of that: they have
# no `state`. Hence a separate pricer, and hence `MatchState` travelling with Λ rather
# than being applied afterwards.
#
# THE SUMMATION ORDER mirrors `06`'s pre-game pricers term for term — away column
# outermost, home row inner, three contiguous runs for 1X2 — so that a live price at
# `t = 0` with an empty state is comparable to the pre-game price bit for bit, up to
# the difference in the intensities themselves. `r01_demo.jl` §8 checks exactly that.

"""
    LiveBook{N}

Preallocated destination AND scratch for `price_live_market!`.

Carrying the two marginal PMF buffers inside the book rather than in a separate
workspace argument is what lets `price_live_market!` keep the briefing's five-argument
signature and still allocate nothing. The buffers and the outputs have the same
lifetime and the same owner — one per worker — so splitting them across two objects
would only create a way to pass a mismatched pair.

`max_goals` truncates the REMAINING-goals distribution, not the final score. 12 is the
repository-wide grid truncation (`TPL_MAX_GOALS`), and it is far past generous for a
remaining-goals count whose Λ is under 3.
"""
struct LiveBook{N}
    out::NTuple{N, Vector{Float64}}
    p_h::Vector{Float64}
    p_a::Vector{Float64}
    max_goals::Int
end

"""
    alloc_live_book(market, n_draws; max_goals = TPL_MAX_GOALS) -> LiveBook

One book per market per worker. The outputs come back in `market_keys(market)` order,
the same order `06`'s `alloc_market_book` uses, so a caller can join on the same symbols.
"""
function alloc_live_book(m, nd::Integer; max_goals::Integer = TPL_MAX_GOALS)
    n = market_arity(m)
    out = ntuple(_ -> Vector{Float64}(undef, Int(nd)), n)
    return LiveBook{n}(out, zeros(Float64, Int(max_goals)), zeros(Float64, Int(max_goals)),
                       Int(max_goals))
end

Base.length(::LiveBook{N}) where {N} = N
Base.getindex(b::LiveBook, i::Int) = b.out[i]
Base.iterate(b::LiveBook, s::Int = 1) = s > length(b) ? nothing : (b.out[s], s + 1)
Base.eltype(::Type{<:LiveBook}) = Vector{Float64}

Base.show(io::IO, b::LiveBook{N}) where {N} =
    print(io, "LiveBook{", N, "}(", length(b.out[1]), " draws, max_goals=", b.max_goals, ")")

"""
    price_live_market!(book, Λ_h, Λ_a, state, market) -> book

Market probabilities for a live match state, one per posterior draw. 0 bytes.

`Λ_h` / `Λ_a` are remaining intensities from `remaining_intensity!`; `state` supplies
the goals already on the board.
"""
function price_live_market!(book::LiveBook{3}, Λ_h::Vector{Float64}, Λ_a::Vector{Float64},
                            s::MatchState, ::Market1X2)
    n = book.max_goals
    home, draw, away = book.out
    gh = s.g_h
    ga = s.g_a
    @inbounds for k in eachindex(Λ_h)
        _uif_pois_pmf!(book.p_h, Λ_h[k], n)
        _uif_pois_pmf!(book.p_a, Λ_a[k], n)
        ph = 0.0
        pd = 0.0
        pa = 0.0
        for c in 1:n                       # away remaining goals = c - 1
            fa = ga + (c - 1)
            pc = book.p_a[c]
            for r in 1:n                   # home remaining goals = r - 1
                fh = gh + (r - 1)
                p = book.p_h[r] * pc
                if fh > fa
                    ph += p
                elseif fh == fa
                    pd += p
                else
                    pa += p
                end
            end
        end
        home[k] = ph
        draw[k] = pd
        away[k] = pa
    end
    return book
end

function price_live_market!(book::LiveBook{2}, Λ_h::Vector{Float64}, Λ_a::Vector{Float64},
                            s::MatchState, ::MarketBTTS)
    n = book.max_goals
    yes, no = book.out
    gh = s.g_h
    ga = s.g_a
    @inbounds for k in eachindex(Λ_h)
        _uif_pois_pmf!(book.p_h, Λ_h[k], n)
        _uif_pois_pmf!(book.p_a, Λ_a[k], n)
        y = 0.0
        nn = 0.0
        for c in 1:n
            fa = ga + (c - 1)
            pc = book.p_a[c]
            for r in 1:n
                fh = gh + (r - 1)
                p = book.p_h[r] * pc
                if fh > 0 && fa > 0
                    y += p
                else
                    nn += p
                end
            end
        end
        yes[k] = y
        no[k] = nn
    end
    return book
end

"""
    price_live_market!(book, Λ_h, Λ_a, state, MarketOverUnder(line)) -> book

Over/under on the FULL-MATCH total, `g_h + g_a + N_h + N_a`.

On an integer line, cells whose total equals the line count towards NEITHER side, so
`over + under < 1`. That is `06`'s behaviour and `src/predictions/market_inference/
over_under.jl`'s before it: the push is dropped rather than voided or split, and a
container change is not the place to alter how pushes settle.
"""
function price_live_market!(book::LiveBook{2}, Λ_h::Vector{Float64}, Λ_a::Vector{Float64},
                            s::MatchState, m::MarketOverUnder)
    n = book.max_goals
    over, under = book.out
    base = s.g_h + s.g_a
    line = m.line
    @inbounds for k in eachindex(Λ_h)
        _uif_pois_pmf!(book.p_h, Λ_h[k], n)
        _uif_pois_pmf!(book.p_a, Λ_a[k], n)
        o = 0.0
        u = 0.0
        for c in 1:n
            pc = book.p_a[c]
            for r in 1:n
                total = base + (r - 1) + (c - 1)
                p = book.p_h[r] * pc
                if total > line
                    o += p
                elseif total < line
                    u += p
                end
            end
        end
        over[k] = o
        under[k] = u
    end
    return book
end

"Fill `p[1:n]` with `pdf(Poisson(λ), 0:n-1)`. Same loop, same order as `06`'s `_tpl_poisson_pmf!`."
@inline function _uif_pois_pmf!(p::Vector{Float64}, λ::Float64, n::Int)
    d = Poisson(λ)
    @inbounds for g in 1:n
        p[g] = pdf(d, g - 1)
    end
    return nothing
end

"""
    price_live_market(Λ_h, Λ_a, state, market; max_goals) -> Dict{Symbol, Vector{Float64}}

Allocating form, keyed exactly as `06`'s `price_market` and
`Predictions.compute_market_probs` key theirs. For the REPL and for a caller that is
pricing one state, once.
"""
function price_live_market(Λ_h::Vector{Float64}, Λ_a::Vector{Float64},
                           s::MatchState, m; max_goals::Integer = TPL_MAX_GOALS)
    book = alloc_live_book(m, length(Λ_h); max_goals = max_goals)
    price_live_market!(book, Λ_h, Λ_a, s, m)
    return Dict{Symbol, Vector{Float64}}(k => v for (k, v) in zip(market_keys(m), book.out))
end


# ==============================================================================
# 6. THE IN-GAME FIT
# ==============================================================================

"""
    pregame_latents(source) -> AbstractPosteriorLatents

Resolve an `InGameFitConfig.pregame` field to the container the pricer needs.

Accepts a completed `Fit` (whose `latents` must have been extracted) or a container
directly. A `Fit` with `latents === nothing` is refused with the reason, because the
alternative — silently pricing against no baseline — is the failure mode this whole
file exists to prevent.
"""
pregame_latents(l::AbstractPosteriorLatents) = l

function pregame_latents(f::Fit)
    l = getfield(f, :latents)
    l === nothing && error(
        "pregame_latents: the pre-game Fit `$(fit_name(f))` carries no latents. " *
        "It was run with `with_latents = false`, or its model's family is not " *
        "registered with `latent_family`. An in-game model has nothing to condition on " *
        "without them.")
    return l
end

pregame_latents(x) = error(
    "InGameFitConfig.pregame must be a `Fit` or an `AbstractPosteriorLatents`; " *
    "got a $(typeof(x)).")

"""
    fit_model(ds::DataStore, config::InGameFitConfig; kwargs...) -> InGameFit

The in-game lifecycle. Structurally identical to the pre-game one — split, features,
sample, audit — with the baseline resolved from `config.pregame` and stored on the
result instead of latents being extracted from it.

The in-play chain's own latents are NOT extracted: an in-game posterior is a set of
multipliers, and the quantity a consumer wants from it is `Λ(t)` for a specific match
state, which `remaining_intensity!` computes on demand. Materialising an OOS container
would mean fixing a state, and there is no single state to fix.
"""
function fit_model(ds::UIF_D.DataStore, config::InGameFitConfig; quiet::Bool = false,
                   kwargs...)
    quiet || _uif_header(config.name * " (in-game)")

    quiet || _uif_step(1, "Generating data splits")
    boundaries = UIF_D.create_id_boundaries(ds, config.splitter)

    quiet || _uif_step(2, "Building feature sets")
    feature_sets = UIF_Feat.create_features(boundaries, ds, config.model, config.splitter)

    return fit_model(config; feature_sets = feature_sets, quiet = quiet, kwargs...)
end

"""
    fit_model(config::InGameFitConfig; feature_sets, kwargs...) -> InGameFit

The `DataStore`-free in-game entry point. Same seam, same reason, as its pre-game
counterpart (`l03_engine.jl` header).
"""
function fit_model(config::InGameFitConfig;
                   feature_sets,
                   gates::ConvergenceGates = ConvergenceGates(),
                   checkpoint_dir::Union{Nothing, String} = nothing,
                   cleanup_checkpoints::Bool = false,
                   quiet::Bool = false)
    start = time()
    pre = pregame_latents(config.pregame)

    n = length(feature_sets)
    n > 0 || error("fit_model: `feature_sets` is empty — the splitter produced no folds.")

    results = load_checkpoints(checkpoint_dir, n)
    pending = findall(isnothing, results)

    if !isempty(pending)
        quiet || _uif_step(3, "Sampling $(length(pending)) of $n in-game folds")
        pending_fs = [feature_sets[i] for i in pending]
        prog = quiet ? _uif_noop : _uif_progress(start)
        fresh = run_folds(config.model, config.sampler, config.execution, pending_fs;
                          on_progress = prog)
        for (k, i) in enumerate(pending)
            results[i] = fresh[k]
            results[i] === nothing && continue
            checkpoint_dir === nothing ||
                save_checkpoint(checkpoint_dir, i, (results[i], feature_sets[i][2]))
        end
    end

    n_failed = count(isnothing, results)
    folds = _uif_narrow(FoldFit[FoldFit(i, results[i], feature_sets[i][2])
                                for i in 1:n if results[i] !== nothing])
    isempty(folds) && error("fit_model: every one of $n in-game folds failed to sample.")

    quiet || _uif_step(4, "Auditing convergence")
    diagnostics = audit_convergence(folds; gates = gates,
                                    max_depth = sampler_max_depth(config.sampler))
    quiet || _uif_info(_uif_diag_line(diagnostics))

    meta = capture_metadata(start)
    tags = copy(config.tags)
    push!(tags, "time:" * format_elapsed(meta.elapsed_seconds))
    push!(tags, "baseline:$(nameof(typeof(pre)))×$(n_matches(pre))")
    n_failed > 0 && push!(tags, "folds_failed:$n_failed")
    diagnostics.passed || push!(tags, "convergence:FAIL")

    stamped = InGameFitConfig(name = config.name, model = config.model,
                              pregame = config.pregame, splitter = config.splitter,
                              sampler = config.sampler, execution = config.execution,
                              tags = tags, description = config.description,
                              save_dir = config.save_dir)

    save_path = default_save_path(stamped, meta)

    if checkpoint_dir !== nothing && cleanup_checkpoints && n_failed == 0
        for i in 1:n
            p = checkpoint_path(checkpoint_dir, i)
            isfile(p) && rm(p; force = true)
        end
    end

    quiet || _uif_footer(meta, n_failed)
    return InGameFit(stamped, folds, pre, diagnostics, meta, save_path)
end

"""
    live_kernel(fit::InGameFit; fold = 1, seed = 20240601) -> LiveKernel

The pricing kernel for one fold of a completed in-game fit, already paired to the fit's
own pre-game baseline.

`fold` defaults to 1 because the common in-play case is a single global fit. For a
walk-forward in-game fit, pass the fold whose training window ENDS BEFORE the match
being priced — pricing a match with a kernel fitted on data that includes it is
look-ahead, and nothing in the type system can catch it for you.
"""
function live_kernel(f::InGameFit; fold::Integer = 1, seed::Integer = 20240601)
    fd = getfield(f, :folds)
    1 <= fold <= length(fd) || error(
        "live_kernel: fold $fold is outside 1:$(length(fd)).")
    fd[fold].chain isa Chains || error(
        "live_kernel: fold $fold holds a $(typeof(fd[fold].chain)), not a `Chains`. " *
        "A point estimate has no posterior to pair.")
    pre = getfield(f, :pregame_latents)
    return build_live_kernel(fd[fold].chain, getfield(f, :config).model,
                             n_draws(pre); seed = seed)
end

"""
    live_book(fit, markets; fold = 1) -> (kernel, Λ_h, Λ_a, books)

Everything a repricing loop needs, allocated ONCE:

```julia
K, Λh, Λa, books = live_book(fit, (Market1X2(), MarketOverUnder(2.5)))
i = match_index(fit.pregame_latents, match_id)

# ... then, on every tick, allocating nothing:
remaining_intensity!(Λh, Λa, K, fit.pregame_latents, i, state)
for (b, m) in zip(books, markets)
    price_live_market!(b, Λh, Λa, state, m)
end
```
"""
function live_book(f::InGameFit, markets; fold::Integer = 1,
                   max_goals::Integer = TPL_MAX_GOALS, seed::Integer = 20240601)
    K = live_kernel(f; fold = fold, seed = seed)
    Λ_h, Λ_a = alloc_intensity(K)
    books = Tuple(alloc_live_book(m, kernel_n_draws(K); max_goals = max_goals)
                  for m in markets)
    return (K, Λ_h, Λ_a, books)
end
