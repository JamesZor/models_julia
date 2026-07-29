#=
l09_ingame.jl — the in-game model, as a usable object.

ONE QUESTION, ONE ANSWER: given (match, current score, red cards, match minute), what is the
goal intensity now, and what are the fair prices for the staking system?

    m = InGameModel("funnel_apm_xg", chain, NHPPXConfig(), draws)

    ingame_rate(m, mid, 62.0; gh=1, ga=0)        # λ per minute, per side, RIGHT NOW
    ingame_remaining(m, mid, 62.0; gh=1, ga=0)   # Λ over the rest of the match (draws)
    ingame_book(m, mid, 62.0;  gh=1, ga=0)       # fair prob per selection → staking layer

WHICH MODEL THIS IS, AND WHY. The incumbent NHPP form:

    log λ_side(t) = α + log λ_side^pregame + β·z(t) + γ_tr·trailing + γ_ld·leading
                    + γ_man·man_adv + δ_time[bin]

i.e. the pregame rate scaled by an in-play multiplier that depends only on TIME and GAME
STATE. That is the specification this stream validated, and the two extensions tried since
were both null on Scottish data (WP-C shot-flow, WP-D shots-so-far nowcast), so nothing has
earned a place in it. Game state itself is a WEAK effect here — `state − time` t = 0.22
match-clustered, consistent with r01/r01b — while **red cards are the dominant repricing
event** (γ_man ≈ 0.5 ⇒ ×1.6–1.7). Time shape and red cards do the work; lead/trail barely
moves the rate.

CALIBRATION OFFSET YOU ARE CARRYING. The composed book sits ~1.4–1.6% BELOW the pregame
book in total intensity (Gate B: K = 0.9856 for funnel_apm_xg, 0.9835 for funnel_winner).
That is the NHPP, fitted on realised goal times, correcting a slightly hot pregame engine.
`kernel_scale` reports it so any price claim can state which K it carries.

Requires l01, l02, l04, l05 to be included first.
=#

using DataFrames, Statistics, Random, Distributions

struct InGameModel
    name::String
    chain::Any
    config::NHPPXConfig
    draws::PregameDraws
end

"Does this model have a pregame posterior for the match?"
Base.haskey(m::InGameModel, mid::Integer) = haskey(m.draws, Int(mid))

"""
    kernel_scale(m; t_now = 0.0) -> Float64

The t=0 kernel K — total composed intensity per unit pregame rate. K < 1 means the in-play
module prices slightly fewer goals than the pregame engine (see the header).
"""
function kernel_scale(m::InGameModel; t_now = 0.0)
    K_h, _ = intensity_kernels(m.chain, m.config; gh = 0, ga = 0, t_now = t_now)
    return mean(K_h)
end

# ---------------------------------------------------------------------------
# 1. The intensity itself
# ---------------------------------------------------------------------------

"""
    ingame_rate(m, mid, t; gh = 0, ga = 0, rh = 0, ra = 0) -> (λ_h, λ_a)

INSTANTANEOUS goal rate per minute for each side at match minute `t`, given the score
(`gh`-`ga`) and red cards (`rh`, `ra`). Posterior means.

This is the quantity the staking layer wants when it asks "how fast are goals coming now" —
multiply by remaining minutes for a crude total, or use `ingame_remaining` for the exact
integral over the non-flat time profile.
"""
function ingame_rate(m::InGameModel, mid::Integer, t::Real;
                     gh::Int = 0, ga::Int = 0, rh::Int = 0, ra::Int = 0)
    d = m.draws[Int(mid)]
    c = m.config
    αv = _cv(m.chain, :α); βv = _cv(m.chain, :β)
    gtr = _cv(m.chain, :γ_tr); gld = _cv(m.chain, :γ_ld); gman = _cv(m.chain, :γ_man)
    nb = Int(cld(c.Tend, c.Δt))
    zt = _has(m.chain, "z_time") ?
         (_cm(m.chain, :z_time, nb) .* _cv(m.chain, :σ_time)) : nothing
    b = clamp(Int(fld(t, c.Δt)) + 1, 1, nb)
    zc = (t - 45) / 45
    gd = gh - ga; man_h = Float64(ra - rh); man_a = -man_h
    bh = αv .+ βv .* zc .+ gtr .* (gd < 0) .+ gld .* (gd > 0) .+ gman .* man_h
    ba = αv .+ βv .* zc .+ gtr .* (gd > 0) .+ gld .* (gd < 0) .+ gman .* man_a
    if zt !== nothing; bh = bh .+ zt[:, b]; ba = ba .+ zt[:, b]; end
    return (λ_h = mean(d.λ_h) * mean(exp.(bh)), λ_a = mean(d.λ_a) * mean(exp.(ba)))
end

"""
    ingame_remaining(m, mid, t; gh, ga, rh, ra) -> (Λ_h, Λ_a)

Integrated remaining goal intensity per side from `t` to full time, as posterior DRAW
VECTORS (pregame draws paired with multiplier draws). Mean these for a point estimate;
keep them for Kelly staking, which needs the uncertainty.
"""
function ingame_remaining(m::InGameModel, mid::Integer, t::Real;
                          gh::Int = 0, ga::Int = 0, rh::Int = 0, ra::Int = 0,
                          n_pairs::Int = 2000, rng = Xoshiro(Int(mid)))
    d = m.draws[Int(mid)]
    K_h, K_a = intensity_kernels(m.chain, m.config; gh = gh, ga = ga,
                                 reds_h = rh, reds_a = ra, t_now = Float64(t))
    npg, nm = length(d.λ_h), length(K_h)
    Λh = Vector{Float64}(undef, n_pairs); Λa = similar(Λh)
    for s in 1:n_pairs
        i = rand(rng, 1:npg); j = rand(rng, 1:nm)
        Λh[s] = d.λ_h[i] * K_h[j]; Λa[s] = d.λ_a[i] * K_a[j]
    end
    return (Λ_h = Λh, Λ_a = Λa)
end

# ---------------------------------------------------------------------------
# 2. The book, for the staking system
# ---------------------------------------------------------------------------

const INGAME_MARKETS = vcat([DataM.Market1X2(), DataM.MarketBTTS()],
                            [DataM.MarketOverUnder(k + 0.5) for k in 0:5])

"""
    ingame_book(m, mid, t; gh, ga, rh, ra, n_pairs, draws = false)
        -> Dict{Symbol, Float64}   (or Dict{Symbol, Vector{Float64}} when draws = true)

Fair probability per selection at the given state — 1X2, BTTS and the O/U ladder, over
FINAL scores (so no line shifting: `over_25` means the match ends with 3+ goals, including
those already scored).

`draws = true` returns the full posterior per selection, which is what a Kelly staking layer
should consume; the scalar version is the point price.
"""
function ingame_book(m::InGameModel, mid::Integer, t::Real;
                     gh::Int = 0, ga::Int = 0, rh::Int = 0, ra::Int = 0,
                     n_pairs::Int = 2000, max_goals::Int = 12, draws::Bool = false,
                     rng = Xoshiro(Int(mid)))
    d = m.draws[Int(mid)]
    ppd = inplay_ppd(m.chain, m.config, d.λ_h, d.λ_a; gh = gh, ga = ga,
                     reds_h = rh, reds_a = ra, t_now = Float64(t),
                     markets = INGAME_MARKETS, n_pairs = n_pairs,
                     max_goals = max_goals, rng = rng)
    return draws ? ppd : Dict(k => mean(v) for (k, v) in ppd)
end

"""
    ingame_state(seqs_or_ms, mid, t) -> (gh, ga, rh, ra)

Current state from an event source — either a `build_event_seqs` dict (BBC) or a single
assembled match tuple. Counts events strictly before `t`.
"""
function ingame_state(src, mid::Integer, t::Real)
    s = src isa AbstractDict ? src[Int(mid)] : src
    (gh = count(g ->  g.home && g.t < t, s.goals),
     ga = count(g -> !g.home && g.t < t, s.goals),
     rh = count(c ->  c.home && c.t < t, s.reds),
     ra = count(c -> !c.home && c.t < t, s.reds))
end

"""
    ingame_trajectory(m, ms; grid = 0:5:85) -> DataFrame

The model's whole in-play path for one match: state, instantaneous rate, remaining Λ and the
main prices at each point on the grid. This is the object to plot against Betfair.
"""
function ingame_trajectory(m::InGameModel, ms; grid = 0.0:5.0:85.0, n_pairs::Int = 800)
    rows = DataFrame(mid = Int[], t = Float64[], gh = Int[], ga = Int[],
                     rh = Int[], ra = Int[], λ_h = Float64[], λ_a = Float64[],
                     Λ_rem = Float64[], sel = Symbol[], p_model = Float64[])
    for t in grid
        st = ingame_state(ms, ms.mid, t)
        r = ingame_rate(m, ms.mid, t; gh = st.gh, ga = st.ga, rh = st.rh, ra = st.ra)
        rem = ingame_remaining(m, ms.mid, t; gh = st.gh, ga = st.ga, rh = st.rh, ra = st.ra,
                               n_pairs = 400)
        book = ingame_book(m, ms.mid, t; gh = st.gh, ga = st.ga, rh = st.rh, ra = st.ra,
                           n_pairs = n_pairs)
        Λ = mean(rem.Λ_h) + mean(rem.Λ_a)
        for (sel, p) in book
            push!(rows, (ms.mid, t, st.gh, st.ga, st.rh, st.ra,
                         r.λ_h, r.λ_a, Λ, sel, p))
        end
    end
    return rows
end
