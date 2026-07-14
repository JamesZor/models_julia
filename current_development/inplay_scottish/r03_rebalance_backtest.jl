#=
r03_rebalance_backtest.jl — WP4: does in-play position management of the pregame book
add growth? Race per match: HOLD vs EXIT (τ=−0.05 full exit, the validated Ireland
rule) vs REBAL (l03 convex program, ℓ1 crossing cost).

Book v1 (pregame): curated per staking findings — totals (0.5–5.5) + BTTS backs only,
model edge ≥ 0.03 vs Betfair pregame close (LOCF −60..0 min), stakes from the l03
log-growth solver at t=0 (joint Kelly), rescaled to Σ ≤ 0.2 (portfolio cap).
Swap in the smile-grid winner's book when that lands.

EXECUTION: decisions price off LOCF fair value, but ALL in-play fills are FORWARD —
first actual print in (t_w+lag, t_w+lag+window] per selection (l04's lesson: as-of
fills at stale LTP give the optimiser fake post-goal edge; first run here produced an
absurd e^2.44/match REBAL under as-of fills — kept as the :asof rows for reference).
Other caveats: no spread/size data (c models crossing), lay commission ~2nd-order.
Eval set: 56 24/25 (betfair + incidents + latents). W0 = 1 per match.
=#

using Serialization, Statistics, DataFrames, Random, Distributions

const BF = BayesianFootball
!(@isdefined ds) && (ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower()))

include(joinpath(dirname(@__DIR__), "match_inplay_explore", "l01_inplay_inverse.jl"))
include(joinpath(@__DIR__, "l01_nhpp_scottish.jl"))
include(joinpath(@__DIR__, "l02_ppd_compose.jl"))
include(joinpath(@__DIR__, "l03_rebalancer.jl"))

OUT = joinpath(@__DIR__, "out")
chain  = deserialize(joinpath(OUT, "r01_chain.jls"))
mseqs  = deserialize(joinpath(OUT, "r01_mseqs.jls"))
config = NHPPXConfig()
latents_df = deserialize(abspath(joinpath(dirname(@__DIR__), "..", "data",
                          "scottish_decay_grid", "latents_hl365_hs2.jls")))
lat = Dict(r.match_id => (collect(r.λ_h), collect(r.λ_a)) for r in eachrow(latents_df))

const G = 13                       # score grid (0..12 goals)
const COMM = 0.02
const CURATED = vcat([Symbol("over_$(k)5") for k in 0:5],
                     [Symbol("under_$(k)5") for k in 0:5], [:btts_yes, :btts_no])
const MIN_EDGE = 0.03
const CAP = 0.2                    # pregame portfolio cap Σ|a|
const CROSS = 0.01                 # ℓ1 crossing cost for in-play trades
const TAU_EXIT = -0.05

eval_ms = [m for m in mseqs if m.tournament_id == 56 && m.season == "24/25"]
bf = ds.betfair_odds

"Mean composed final-score matrix + per-sel mean probs at a state."
function fair_state(ms, t_m, gh, ga, rh, ra; n_pairs = 800)
    λh, λa = lat[ms.mid]
    K_h, K_a = intensity_kernels(chain, config; gh = gh, ga = ga,
                                 reds_h = rh, reds_a = ra, t_now = t_m)
    S = compose_score_matrix(λh, λa, K_h, K_a; gh = gh, ga = ga, n_pairs = n_pairs,
                             max_goals = G - 1, rng = Xoshiro(ms.mid + Int(round(t_m * 10))))
    P̄ = dropdims(mean(S.data; dims = 3); dims = 3)
    return P̄
end

sel_prob(P̄, sel) = sum(P̄[cells_for(sel, G)])

"Equivalent lay stake that flattens a net back exposure s@o_e at current odds o_n."
exit_stake(s, o_e, o_n; comm = COMM) =
    s * (1 + (o_e - 1) * (1 - comm)) / (1 + (o_n - 1) * (1 - comm))

"FORWARD fill prices: first print per selection in (t_w+lag, t_w+lag+window]."
function forward_prices(bf_match::AbstractDataFrame, t_w::Real; lag = 1.0, window = 5.0)
    out = Dict{Symbol, Float64}()
    sub = filter(r -> (t_w + lag) < r.minutes_to_kickoff <= (t_w + lag + window), bf_match)
    isempty(sub) && return out
    for gdf in groupby(sub, :selection)
        r = first(sort(gdf, :minutes_to_kickoff))
        out[r.selection] = r.traded_price
    end
    return out
end

"Terminal wealth of a trade list given the final score."
function settle(trades, fh, fa; W0 = 1.0)
    fh = min(fh, G - 1); fa = min(fa, G - 1)
    W0 + sum((t.stake * (cells_for(t.sel, G)[fh + 1, fa + 1] ?
              (t.price - 1) * (1 - COMM) : -1.0) for t in trades); init = 0.0)
end

# ---------------------------------------------------------------------------
# per-match simulation
# ---------------------------------------------------------------------------

function run_match(ms)
    bfm = subset(bf, :match_id => ByRow(==(ms.mid)))
    isempty(bfm) && return nothing
    pre = latest_prices(subset(bfm, :minutes_to_kickoff => ByRow(x -> -60.0 < x <= 0.0)),
                        0.0; staleness = 60.0)
    isempty(pre) && return nothing

    P̄0 = fair_state(ms, 0.0, 0, 0, 0, 0; n_pairs = 1500)
    cands = Contract[]
    for sel in CURATED
        haskey(pre, sel) || continue
        p = sel_prob(P̄0, sel)
        (p * pre[sel] * (1 - COMM) - 1.0) >= MIN_EDGE || continue
        push!(cands, Contract(sel, pre[sel], cells_for(sel, G)))
    end
    isempty(cands) && return (mid = ms.mid, n_bets = 0)

    sol = rebalance(P̄0, zeros(G, G), cands; W0 = 1.0, c = 0.002, comm = COMM)
    a0 = max.(sol.Δa, 0.0)                          # backs only pregame
    tot = sum(a0)
    tot > CAP && (a0 .*= CAP / tot)
    book0 = [Trade(c.sel, c.price, a0[i]) for (i, c) in enumerate(cands) if a0[i] > 1e-6]
    isempty(book0) && return (mid = ms.mid, n_bets = 0)

    anchors = anchor_goals(bf, ds, ms.mid)
    cm = make_clock_map(anchors)
    trades = Dict(:hold => copy(book0), :exit => copy(book0), :rebal => copy(book0))
    exited = Set{Symbol}(); n_rebal_trades = 0

    for t_w in 10.0:5.0:110.0
        prices = latest_prices(bfm, t_w; staleness = 4.0)
        isempty(prices) && continue
        t_m = cm(t_w); (1.0 <= t_m <= 88.0) || continue
        gh = count(g ->  g.home && g.t < t_m, ms.goals)
        ga = count(g -> !g.home && g.t < t_m, ms.goals)
        rh = count(c ->  c.home && c.t < t_m, ms.reds)
        ra = count(c -> !c.home && c.t < t_m, ms.reds)
        P̄ = fair_state(ms, t_m, gh, ga, rh, ra)

        raw_fills = forward_prices(bfm, t_w)      # next actual prints
        # limit-order discipline: reject fills > 10% (log) from the decision quote —
        # the solver never authorized the post-goal repriced print
        fills = Dict(s => p for (s, p) in raw_fills
                     if haskey(prices, s) && abs(log(p / prices[s])) <= 0.10)

        # EXIT: decision on LOCF fair value, FILL at the next accepted print
        for b in book0
            (b.sel in exited || !haskey(prices, b.sel)) && continue
            e = sel_prob(P̄, b.sel) - 1.0 / prices[b.sel]
            if e <= TAU_EXIT && haskey(fills, b.sel)
                push!(trades[:exit], Trade(b.sel, fills[b.sel],
                                           -exit_stake(b.stake, b.price, fills[b.sel])))
                push!(exited, b.sel)
            end
        end

        # REBAL: solve on LOCF quotes, fill each leg at the next accepted print
        avail = [Contract(s, prices[s], cells_for(s, G)) for s in CURATED if haskey(prices, s)]
        isempty(avail) && continue
        π_now = payoff_vector(trades[:rebal], G; comm = COMM)
        r = rebalance(P̄, π_now, avail; W0 = 1.0, c = CROSS, comm = COMM)
        for (i, c) in enumerate(avail)
            (abs(r.Δa[i]) > 1e-4 && haskey(fills, c.sel)) || continue
            push!(trades[:rebal], Trade(c.sel, fills[c.sel], r.Δa[i]))
            n_rebal_trades += 1
        end
    end

    fh = count(g -> g.home, ms.goals); fa = count(g -> !g.home, ms.goals)
    (mid = ms.mid, n_bets = length(book0), stake0 = sum(b.stake for b in book0),
     n_exits = length(exited), n_rebal = n_rebal_trades,
     G_hold  = log(max(settle(trades[:hold],  fh, fa), 1e-6)),
     G_exit  = log(max(settle(trades[:exit],  fh, fa), 1e-6)),
     G_rebal = log(max(settle(trades[:rebal], fh, fa), 1e-6)))   # 1e-6 = ruin floor
end

results = NamedTuple[]
for ms in eval_ms
    r = run_match(ms)
    r === nothing || push!(results, r)
end
booked = [r for r in results if r.n_bets > 0]
gdf = DataFrame(booked)

function line(gs, name)
    (strategy = name, G_mean = mean(gs), se = std(gs) / sqrt(length(gs)),
     total_growth = exp(sum(gs)), worst = minimum(gs), ruined = any(gs .<= log(1e-6)))
end
race = DataFrame([line(gdf.G_hold, "HOLD"), line(gdf.G_exit, "EXIT τ=-0.05"),
                  line(gdf.G_rebal, "REBAL c=$(CROSS)")])
d_exit = gdf.G_exit .- gdf.G_hold; d_reb = gdf.G_rebal .- gdf.G_hold
uplift = (exit_vs_hold = (mean = mean(d_exit), t = mean(d_exit)/(std(d_exit)/sqrt(nrow(gdf)))),
          rebal_vs_hold = (mean = mean(d_reb), t = mean(d_reb)/(std(d_reb)/sqrt(nrow(gdf)))),
          n = nrow(gdf), skipped_no_book = length(results) - nrow(gdf),
          avg_bets = mean(gdf.n_bets), avg_stake = mean(gdf.stake0),
          avg_exits = mean(gdf.n_exits), avg_rebal_trades = mean(gdf.n_rebal))

serialize(joinpath(OUT, "r03_results.jls"), (results = results, race = race, uplift = uplift))
R03 = (race = race, uplift = uplift)
@info "r03 done" nrow(gdf)
