# r04_diagnostics.jl -- what to check before you believe a number.
#
# Every check here corresponds to a bug the prototype actually shipped with. Run this whenever
# you change a BookSpec component, add a market, or move to a new league.

include("_setup.jl")

spec   = PF.BookSpec(markets = MARKETS)
policy = PF.PolicySpec(trust = PF.FlatTrust(0.25), risk = PF.SlateDrawdown(23.0),
                       cap = PF.FixedCap(0.25))

CACHE = joinpath(@__DIR__, "books_$(string(PF.book_cache_key(spec), base = 16)).jls")
books = isfile(CACHE) ? deserialize(CACHE) :
        (b = PF.build_books(spec, latents_df, expr, odds, ds); serialize(CACHE, b); b)
slates = PF.group(policy.grouping, books)
traj   = PF.simulate(policy, slates)

# ===================================================================
# 1. Is the solver actually solving?
# ===================================================================
#
# `kkt` is the worst first-order-condition violation. It is NOT a convergence flag -- Optim can
# report success at a point that does not satisfy KKT. Want p99 < 1e-4.

println("\n", "="^78, "\n=== 1. SOLVER ===\n", "="^78)
let kk = [b.kkt for b in books]
    @printf("  KKT residual   median %.1e | p99 %.1e | max %.1e | above 1e-5: %.1f%%\n",
            median(kk), quantile(kk, 0.99), maximum(kk), 100mean(kk .> 1e-5))
    @printf("  not converged  %d of %d\n", count(!b -> b, [b.converged for b in books]), length(books))
end

# ===================================================================
# 2. Is the price you settle at defensible?
# ===================================================================
#
# DeArb only ever SHRINKS a price. If `worst` is large you are looking at a market whose closing
# book was badly impossible -- usually a liquidity artifact worth investigating, not a free bet.

println("\n", "="^78, "\n=== 2. PRICING ===\n", "="^78)
let de = [s.odds_used / s.odds_quoted for b in books for s in b.sels]
    @printf("  quotes shrunk by de-arb  %.1f%%\n", 100mean(de .< 1 - 1e-12))
    @printf("  mean shrink              %.3f%%   worst %.2f%%\n",
            100(1 - mean(de)), 100(1 - minimum(de)))
    @printf("  settles above traded?    %s   <- must be false\n", any(de .> 1 + 1e-12))
end

# A market group the optimum fully covers means an arbitrage got through the pricing layer.
# After de-arb, with commission, a full cover is dominated in every state -- so this must be 0.
let cover = 0
    for b in books
        g = Dict{Tuple{String,Float64},Vector{Int}}()
        for (j, s) in enumerate(b.sels); push!(get!(g, (s.group, s.line), Int[]), j); end
        for (_, idx) in g
            count(j -> b.a_kelly[j] > 0, idx) == length(idx) && (cover += 1)
        end
    end
    @printf("  market groups fully covered by the optimum: %d   <- must be 0\n", cover)
end

# ===================================================================
# 3. Is the risk layer doing its job?
# ===================================================================

println("\n", "="^78, "\n=== 3. RISK ===\n", "="^78)
let m = PF.path_metrics(traj)
    @printf("  mean / max exposure    %.1f%% / %.1f%% of bankroll per slate\n",
            100m.mean_exposure, 100m.max_exposure)
    @printf("  worst slate            %.1f%%   <- must be > -100%%\n", 100m.worst_slate)
    @printf("  min bankroll           %.4f   <- must be > 0\n", minimum(traj.bankroll))
    @printf("  slates hitting the cap %d of %d\n", traj.n_capped, length(slates))
    # Does the drawdown promise hold? It is made under the model's measure, so expect a miss.
    nominal = 100 * (1 - exp(log(0.01) / policy.risk.lambda))
    @printf("  nominal drawdown @1%%   %.1f%%   realised %.1f%%   overshoot %.2fx\n",
            nominal, -m.mdd, -m.mdd / nominal)
end

# ===================================================================
# 4. Where does the money actually come from?
# ===================================================================
#
# Concentration is the thing to look at. If one family carries the P/L, your result is a bet on
# that family, however diversified the book looks.

println("\n", "="^78, "\n=== 4. ATTRIBUTION ===\n", "="^78)
att = PF.attribution(traj)
att.stake = round.(att.stake, digits = 3); att.pnl = round.(att.pnl, digits = 3)
att.roi = round.(att.roi, digits = 2); att.hit = round.(att.hit, digits = 3)
att.med_odds = round.(att.med_odds, digits = 2)
println(att)
@printf("\n  top family carries %.0f%% of total P/L on %.0f%% of the stake\n",
        100 * first(att.pnl) / sum(att.pnl), 100 * first(att.stake) / sum(att.stake))

# ===================================================================
# 5. Is any of it distinguishable from luck?
# ===================================================================
#
# The interval is resampled BY MATCH. Bets on the same match share a scoreline, so resampling
# individual bets would make it look far tighter than it is.

println("\n", "="^78, "\n=== 5. SIGNIFICANCE ===\n", "="^78)
let ci = PF.bootstrap_roi(traj.bets), m = PF.path_metrics(traj)
    @printf("  flat ROI  %.2f%%   95%% CI [%.2f%%, %.2f%%]  (sd %.2f)\n",
            m.roi, ci.lo, ci.hi, ci.sd)
    println(ci.lo > 0 ? "  interval excludes zero." :
                        "  INTERVAL INCLUDES ZERO -- not distinguishable from luck on this sample.")
end

# ===================================================================
# 6. Model versus market
# ===================================================================
#
# A sanity anchor. If the model has no log-loss edge over the vig-removed market, any profit is
# either coming from the tails (plausible, and the intended design) or from luck.

println("\n", "="^78, "\n=== 6. FORECAST SKILL vs THE MARKET ===\n", "="^78)
let rows = NamedTuple[]
    for b in books, (j, s) in enumerate(b.sels)
        y  = b.settle[j] > 0
        pm = clamp(s.p_model, 1e-6, 1 - 1e-6)
        pk = clamp(s.p_market, 1e-6, 1 - 1e-6)
        push!(rows, (fam = s.group * (s.group == "OverUnder" ? " $(s.line)" : ""),
                     ll_model = -(y ? log(pm) : log1p(-pm)),
                     ll_mkt   = -(y ? log(pk) : log1p(-pk))))
    end
    df = DataFrame(rows)
    g = combine(groupby(df, :fam), nrow => :n,
                :ll_model => mean => :model, :ll_mkt => mean => :market)
    g.diff = round.(g.model .- g.market, digits = 5)
    g.model = round.(g.model, digits = 4); g.market = round.(g.market, digits = 4)
    println(g)
    println("\n  diff < 0 means the model beats the market. Near zero means it does not, which is")
    println("  not fatal -- edge can live in per-match deviations rather than in the mean.")
end
