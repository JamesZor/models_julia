# r17_risk_scope_ab.jl
#
# A/B: WHERE should the Busseti drawdown budget be spent?
#
#   :match  -- l03's engine. Each match gets its own k from its own return distribution.
#              Bounds the drawdown of every bet in isolation; says nothing about the
#              ~6 bets that all settle at 3pm.
#   :slate  -- l06's default. One k for the whole day, solved against all L matches.
#
# The two engines lever differently, so comparing them at their natural settings just
# measures which one bets bigger. `calibrate_scale` puts both on the SAME mean realised
# slate exposure first, so what is left is allocation shape.

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

include("l06_portfolio_v2.jl")

# -------------------------------------------------------------------
# Data / books  (reuse r16's cache; run r16 first if it is missing)
# -------------------------------------------------------------------
if !isdefined(Main, :ds);   global ds   = D.load_datastore_cached(D.ScottishLower()); end
if !isdefined(Main, :odds); global odds = D.summarize_betfair_market(ds,
        open_window = (-100000.0, -10.0), close_window = (-20.0, 0.0)); end
if !isdefined(Main, :expr)
    global expr = E.load_experiment(E.list_experiments("./data/experiments/plus_minus_biweek",
                                                       data_dir = ""), 3)
end
if !isdefined(Main, :latents); global latents = E.extract_oos_predictions(ds, expr); end

MK17 = D.MarketConfig(reduce(vcat, (D.AbstractMarket[D.Market1X2(), D.MarketBTTS()],
                                    [D.MarketOverUnder(i + 0.5) for i in 0:4])))
CFG17 = PortfolioConfig()

const CACHE17 = joinpath(@__DIR__, "r16_books.jls")
books17 = if isfile(CACHE17)
    deserialize(CACHE17)
else
    b = build_books(latents.df, expr, odds, MK17, ds; cfg = CFG17,
                    shrink = ShrinkConfig(enabled = true, n_draws = 128))
    serialize(CACHE17, b); b
end
slates17 = build_slates(books17)
ALPHAS17 = Dict{String,Float64}(alpha_key(s) => 1.0 for b in books17 for s in b.sels)

# matches-per-slate context: the whole point of the comparison
let L = [length(s.books) for s in slates17]
    @printf("\n%d matches over %d slates | matches/slate: median %d, mean %.1f, max %d\n",
            length(books17), length(slates17), median(L), mean(L), maximum(L))
end

# -------------------------------------------------------------------
# 1. Natural settings (no exposure matching) -- what each engine does left alone
# -------------------------------------------------------------------
println("\n", "="^92, "\n=== 1. NATURAL SETTINGS (alpha = 1, cap 0.99 so only lambda binds) ===\n", "="^92)
nat = DataFrame(scope = Symbol[], lambda = Float64[], mean_expo = Float64[], max_expo = Float64[],
                roi = Float64[], final = Float64[], mdd = Float64[], worst_slate = Float64[])
for scope in (:match, :slate), lam in (10.0, 20.0, 30.0)
    rc  = RiskConfig(lambda = lam, slate_cap = 0.99, scope = scope)
    sim = simulate(slates17, ALPHAS17, CFG17, rc; use_bm = false)
    pm  = path_metrics(sim)
    push!(nat, (scope, lam, round(mean(sim.exposure), digits = 3),
                round(maximum(sim.exposure), digits = 3), round(pm.roi, digits = 2),
                round(pm.final, digits = 3), round(pm.mdd, digits = 1),
                round(pm.worst_slate_pl, digits = 3)))
end
display(nat)

# -------------------------------------------------------------------
# 2. Exposure-matched: same mean daily exposure, both scopes
# -------------------------------------------------------------------
println("\n", "="^92, "\n=== 2. EXPOSURE-MATCHED (lambda tuned per scope to hit the same mean exposure) ===\n", "="^92)
# NOTE: exposure must be matched by moving LAMBDA, not by scaling stakes. An active
# Busseti constraint solves k against whatever stakes it is handed, so a stake
# multiplier is very nearly a no-op -- exposure saturates at the lambda-implied level.
# That scale-invariance is the same mechanism that makes lambda subsume alpha.
ab = DataFrame(target_expo = Float64[], scope = Symbol[], lambda = Float64[],
               mean_expo = Float64[], max_expo = Float64[], roi = Float64[],
               final = Float64[], growth = Float64[], mdd = Float64[],
               worst_slate = Float64[], n_capped = Int[])
for tgt in (0.05, 0.10, 0.15, 0.20), scope in (:match, :slate)
    rc0 = RiskConfig(lambda = 20.0, slate_cap = 0.99, scope = scope)
    lam = calibrate_lambda(slates17, ALPHAS17, CFG17, rc0;
                           target_exposure = tgt, use_bm = false)
    rc  = RiskConfig(lambda = lam, slate_cap = 0.99, scope = scope)
    sim = simulate(slates17, ALPHAS17, CFG17, rc; use_bm = false)
    pm  = path_metrics(sim)
    push!(ab, (tgt, scope, round(lam, digits = 2), round(mean(sim.exposure), digits = 4),
               round(maximum(sim.exposure), digits = 3), round(pm.roi, digits = 2),
               round(pm.final, digits = 3), round(pm.growth_per_slate, digits = 5),
               round(pm.mdd, digits = 1), round(pm.worst_slate_pl, digits = 3), sim.n_capped))
end
display(ab)

# scale-invariance, demonstrated rather than asserted
println("\n  stake multiplier vs realised exposure at lambda = 20 (slate scope):")
let rc = RiskConfig(lambda = 20.0, slate_cap = 0.99, scope = :slate)
    for scx in (0.25, 1.0, 4.0, 16.0)
        e = mean(simulate(slates17, ALPHAS17, CFG17, rc; use_bm = false, global_scale = scx).exposure)
        @printf("    scale %6.2f  ->  mean exposure %.4f\n", scx, e)
    end
end

# -------------------------------------------------------------------
# 3. Does either scope deliver the drawdown it promises?
#    lambda = log(beta)/log(D) is a claim: P(bankroll ever < D) <= beta.
# -------------------------------------------------------------------
println("\n", "="^92, "\n=== 3. IS THE DRAWDOWN PROMISE KEPT? (beta = 0.01) ===\n", "="^92)
promise = DataFrame(scope = Symbol[], lambda = Float64[], nominal_dd = Float64[],
                    realised_mdd = Float64[], mean_expo = Float64[], kept = Bool[])
for scope in (:match, :slate), lam in (10.0, 20.0, 30.0, 50.0)
    rc  = RiskConfig(lambda = lam, slate_cap = 0.99, scope = scope)
    sim = simulate(slates17, ALPHAS17, CFG17, rc; use_bm = false)
    pm  = path_metrics(sim)
    nominal = 100 * (1 - exp(log(0.01) / lam))          # D = beta^(1/lambda)
    push!(promise, (scope, lam, round(nominal, digits = 1), round(-pm.mdd, digits = 1),
                    round(mean(sim.exposure), digits = 3), -pm.mdd <= nominal))
end
display(promise)
println("\n  nominal_dd is the drawdown the lambda claims to bound at 1% probability;")
println("  realised_mdd is what the single historical path actually did.")
