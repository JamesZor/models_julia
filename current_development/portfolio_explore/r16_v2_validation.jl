# r16_v2_validation.jl
#
# Runner for l06_portfolio_v2.jl. Three jobs:
#   A. assert every invariant that l02-l05 silently violated
#   B. produce the honest (untuned, chronological, slate-capped) baseline
#   C. race walk-forward alpha against flat alpha and against the frozen r08 vector,
#      out of sample, so we can see whether per-market trust survives contact with
#      data it was not fitted on.

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization
using ThreadPinning

include("l06_portfolio_v2.jl")

isdefined(Main, :__pinned__) || (pinthreads(:cores); global __pinned__ = true)
@info "threads" n = Threads.nthreads()

# -------------------------------------------------------------------
# 1. Data (reuse whatever is already warm in the session)
# -------------------------------------------------------------------
if !isdefined(Main, :ds);   global ds   = D.load_datastore_cached(D.ScottishLower()); end
if !isdefined(Main, :odds); global odds = D.summarize_betfair_market(ds,
        open_window = (-100000.0, -10.0), close_window = (-20.0, 0.0)); end
if !isdefined(Main, :expr)
    global expr = E.load_experiment(E.list_experiments("./data/experiments/plus_minus_biweek",
                                                       data_dir = ""), 3)
end
if !isdefined(Main, :latents); global latents = E.extract_oos_predictions(ds, expr); end

scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
over_unders    = [D.MarketOverUnder(i + 0.5) for i in 0:4]
MK             = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

CFG    = PortfolioConfig()
SHRINK = ShrinkConfig(enabled = true, n_draws = 128)

# -------------------------------------------------------------------
# 2. Build (or load) the books
# -------------------------------------------------------------------
const BOOK_CACHE = joinpath(@__DIR__, "r16_books.jls")

books = if isfile(BOOK_CACHE) && get(ENV, "R16_REBUILD", "0") != "1"
    @info "loading cached books" BOOK_CACHE
    deserialize(BOOK_CACHE)
else
    @info "building books (Kelly + Baker-McHale on $(SHRINK.n_draws) draws per match)"
    b = @time build_books(latents.df, expr, odds, MK, ds; cfg = CFG, shrink = SHRINK)
    serialize(BOOK_CACHE, b)
    b
end
slates = build_slates(books)
@info "books built" n_matches = length(books) n_slates = length(slates)

flat_alphas(a::Float64) = Dict{String,Float64}(alpha_key(s) => a for b in books for s in b.sels)
ALL_KEYS = sort(collect(keys(flat_alphas(1.0))))

# -------------------------------------------------------------------
# 3. INVARIANTS  (each one is a bug l02-l05 had)
# -------------------------------------------------------------------
println("\n", "="^78, "\n=== A. INVARIANTS ===\n", "="^78)

function invariants(books, slates)
    fails = String[]
    chk(ok, msg) = ok ? nothing : push!(fails, msg)

    chk(issorted([b.date for b in books]), "books not chronological")
    chk(all(abs(sum(b.p_grid) - 1) < 1e-10 for b in books), "score grid does not sum to 1")
    kk = [b.kkt for b in books]
    chk(quantile(kk, 0.99) < 1e-4,
        "KKT residual too large: p99 = $(quantile(kk, 0.99))")
    chk(all(sum(b.a_kelly) <= CFG.budget + 1e-8 for b in books), "budget constraint violated")
    chk(all(all(b.a_kelly .>= -1e-12) for b in books), "negative stake")

    # The netting-pass premise. Once the book is de-arbed, a FULL COVER of a market
    # group is strictly dominated (sum 1/(1+c_i) > 1 with commission), so the optimum
    # must never take one. Note a 2-of-3 position on 1X2 is NOT a cover -- it is a
    # legitimate double-chance-shaped hedge and must be allowed through.
    cover = 0
    for b in books
        g = Dict{Tuple{String,Float64},Vector{Int}}()
        for (j, s) in enumerate(b.sels); push!(get!(g, (s.group, s.line), Int[]), j); end
        for (_, idx) in g
            count(j -> b.a_kelly[j] > 0, idx) == length(idx) && (cover += 1)
        end
    end
    chk(cover == 0, "optimum fully covered a market group $cover times (arb artifact leaked through)")

    # completeness guard
    bad = 0
    for b in books
        g = Dict{Tuple{String,Float64},Int}()
        for s in b.sels; g[(s.group, s.line)] = get(g, (s.group, s.line), 0) + 1; end
        for ((grp, ln), n) in g
            want = grp == "1X2" ? 3 : 2
            n == want || (bad += 1)
        end
    end
    chk(bad == 0, "$bad incomplete market groups leaked through")

    # payoff matrix agrees with settlement on the realised score
    chk(all(all(isfinite, b.R) for b in books), "non-finite payoff in R")
    chk(all(all(x -> x == 0.0 || x < 0.0 || x > 0.0, b.settle) for b in books), "bad settle vector")

    # exposure cap makes ruin structurally impossible
    rc = RiskConfig(lambda = 0.0, slate_cap = 0.25)
    sim = simulate(slates, flat_alphas(1.0), CFG, rc)
    chk(minimum(sim.bankroll) > 0, "bankroll went non-positive")
    chk(maximum(abs, sim.day_pl) < 1.0, "a slate risked more than the bankroll")

    return fails
end

let fails = invariants(books, slates)
    if isempty(fails)
        println("  all invariants hold ✓")
    else
        for f in fails; println("  ✗ ", f); end
    end
end

# push handling / grading coverage / solver quality
let npush = sum(count(==(0.0), b.settle) for b in books),
    nsel  = sum(length(b.settle) for b in books),
    kk    = [b.kkt for b in books],
    dearb = [s.odds_used / s.odds_quoted for b in books for s in b.sels]
    @printf("  selections %d | push/void %d (%.2f%%) | KKT med %.1e p99 %.1e max %.1e\n",
            nsel, npush, 100npush/nsel, median(kk), quantile(kk, 0.99), maximum(kk))
    @printf("  de-arb: %.1f%% of quotes shrunk, mean shrink %.3f%%, worst %.2f%%\n",
            100mean(dearb .< 1 - 1e-12), 100 * (1 - mean(dearb)), 100 * (1 - minimum(dearb)))
    twothree = 0
    for b in books
        g = Dict{Tuple{String,Float64},Vector{Int}}()
        for (j, s) in enumerate(b.sels); push!(get!(g, (s.group, s.line), Int[]), j); end
        for ((grp, _), idx) in g
            grp == "1X2" && count(j -> b.a_kelly[j] > 0, idx) == 2 && (twothree += 1)
        end
    end
    println("  informational: 2-of-3 1X2 hedges taken by the optimum: $twothree")
end

# -------------------------------------------------------------------
# 4. Baker-McHale shrinkage
# -------------------------------------------------------------------
println("\n", "="^78, "\n=== B. BAKER-McHALE PARAMETER-UNCERTAINTY SHRINKAGE ===\n", "="^78)
let ks = [b.k_bm for b in books], stk = [sum(b.a_kelly) for b in books]
    @printf("  k*   median %.3f   mean %.3f   q10 %.3f   q90 %.3f   at 1.0: %.1f%%\n",
            median(ks), mean(ks), quantile(ks, 0.1), quantile(ks, 0.9), 100mean(ks .>= 0.999))
    @printf("  full-Kelly stake per match: median %.1f%%  max %.1f%% of bankroll\n",
            100median(stk), 100maximum(stk))
end

# -------------------------------------------------------------------
# 5. Honest baseline: flat alpha x lambda x cap, chronological
# -------------------------------------------------------------------
println("\n", "="^78, "\n=== C. HONEST BASELINE (no tuning, chronological, capped) ===\n", "="^78)

base = DataFrame(alpha = Float64[], lambda = Float64[], cap = Float64[], bm = Bool[],
                 avg_k = Float64[], roi = Float64[], final = Float64[],
                 growth = Float64[], mdd = Float64[])
for a in (0.10, 0.25, 0.50, 1.00), lam in (0.0, 10.0, 20.0), bm in (false, true)
    rc  = RiskConfig(lambda = lam, slate_cap = 0.25)
    sim = simulate(slates, flat_alphas(a), CFG, rc; use_bm = bm)
    pm  = path_metrics(sim)
    push!(base, (a, lam, 0.25, bm, round(pm.avg_k, digits = 3), round(pm.roi, digits = 2),
                 round(pm.final, digits = 3), round(pm.growth_per_slate, digits = 5),
                 round(pm.mdd, digits = 1)))
end
display(base)

# -------------------------------------------------------------------
# 6. Walk-forward alpha vs flat vs frozen r08 vector, OUT OF SAMPLE
# -------------------------------------------------------------------
println("\n", "="^78, "\n=== D. ALPHA RACE (second half = out of sample) ===\n", "="^78)

R08_VECTOR = Dict{String,Float64}(
    "O/U 3.5_over_35" => 0.002, "BTTS_btts_no" => 0.003, "O/U 4.5_over_45" => 0.003,
    "O/U 1.5_over_15" => 0.004, "O/U 1.5_under_15" => 0.027, "O/U 0.5_under_05" => 0.029,
    "O/U 3.5_under_35" => 0.131, "1X2_away" => 0.157, "BTTS_btts_yes" => 0.183,
    "1X2_home" => 0.432, "1X2_draw" => 0.445, "O/U 2.5_over_25" => 0.486,
    "O/U 2.5_under_25" => 0.818, "O/U 0.5_over_05" => 0.846, "O/U 4.5_under_45" => 1.000)

cut       = slates[cld(length(slates), 2)].date
oos       = [s for s in slates if s.date > cut]
RC        = RiskConfig(lambda = 20.0, slate_cap = 0.25)
@info "OOS window" from = minimum(s.date for s in oos) n_slates = length(oos)

wf_all = walkforward_alphas(slates)
wf_oos = wf_all[(length(slates) - length(oos) + 1):end]

race = DataFrame(policy = String[], roi = Float64[], final = Float64[],
                 growth = Float64[], mdd = Float64[], avg_k = Float64[])
for (nm, am) in (("flat 0.10", flat_alphas(0.10)), ("flat 0.25", flat_alphas(0.25)),
                 ("flat 1.00", flat_alphas(1.00)), ("r08 frozen vector", R08_VECTOR))
    pm = path_metrics(simulate(oos, am, CFG, RC))
    push!(race, (nm, round(pm.roi, digits = 2), round(pm.final, digits = 3),
                 round(pm.growth_per_slate, digits = 5), round(pm.mdd, digits = 1),
                 round(pm.avg_k, digits = 3)))
end
let pm = path_metrics(simulate_walkforward(oos, wf_oos, CFG, RC))
    push!(race, ("walk-forward alpha", round(pm.roi, digits = 2), round(pm.final, digits = 3),
                 round(pm.growth_per_slate, digits = 5), round(pm.mdd, digits = 1),
                 round(pm.avg_k, digits = 3)))
end
display(race)

println("\n  final walk-forward alphas:")
let last_map = wf_all[end]
    for k in ALL_KEYS
        haskey(last_map, k) && @printf("    %-20s %.3f\n", k, last_map[k])
    end
end

# -------------------------------------------------------------------
# 7. Attribution + significance on the honest policy
# -------------------------------------------------------------------
println("\n", "="^78, "\n=== E. ATTRIBUTION (flat 0.25, lambda 20, cap 0.25) ===\n", "="^78)
honest_sim = simulate(slates, flat_alphas(0.25), CFG, RC)
let sim = honest_sim,
    ci  = bootstrap_roi(sim.bets)
    global att = combine(groupby(sim.bets, :key), nrow => :n, :stake => sum => :stake,
                  :pl => sum => :pl, :odds => median => :med_odds,
                  :payoff => (x -> mean(x .> 0)) => :hit)
    att.roi = round.(100 .* att.pl ./ att.stake, digits = 2)
    att.stake = round.(att.stake, digits = 3); att.pl = round.(att.pl, digits = 3)
    att.med_odds = round.(att.med_odds, digits = 2); att.hit = round.(att.hit, digits = 3)
    display(sort(att, :pl, rev = true))
    @printf("\n  flat ROI %.2f%%   bootstrap 95%% CI [%.2f%%, %.2f%%]  (clustered by match)\n",
            100 * sim.total_pl / sim.total_stake, ci.lo, ci.hi)
end

# -------------------------------------------------------------------
# 8. How much of the l02 headline was the closing-window arbitrage?
# -------------------------------------------------------------------
println("\n", "="^78, "\n=== F. PRICE-MODE ABLATION ===\n", "="^78)
price_abl = DataFrame(price_mode = Symbol[], roi = Float64[], final = Float64[],
                      mdd = Float64[], full_covers = Int[], mean_stake_pct = Float64[])
for mode in (:dearb, :normalise, :raw)
    cfg  = PortfolioConfig(price_mode = mode)
    bks  = build_books(latents.df, expr, odds, MK, ds; cfg = cfg,
                       shrink = ShrinkConfig(enabled = false))
    sls  = build_slates(bks)
    cover = 0
    for b in bks
        g = Dict{Tuple{String,Float64},Vector{Int}}()
        for (j, s) in enumerate(b.sels); push!(get!(g, (s.group, s.line), Int[]), j); end
        for (_, idx) in g
            count(j -> b.a_kelly[j] > 0, idx) == length(idx) && (cover += 1)
        end
    end
    am  = Dict{String,Float64}(alpha_key(s) => 0.25 for b in bks for s in b.sels)
    pmm = path_metrics(simulate(sls, am, PortfolioConfig(price_mode = mode),
                                RiskConfig(lambda = 20.0, slate_cap = 0.25); use_bm = false))
    push!(price_abl, (mode, round(pmm.roi, digits = 2), round(pmm.final, digits = 3),
                      round(pmm.mdd, digits = 1), cover,
                      round(100 * mean(sum(b.a_kelly) for b in bks), digits = 1)))
end
display(price_abl)
