# r18_src_parity.jl
#
# Acceptance gate for the graduation of l06 into src/Portfolio.
#
# Rather than assert against numbers copied out of a previous session, this builds BOTH
# implementations on the same data in the same process and diffs them element by element. A port
# bug shows up as a disagreeing stake, not as a headline that drifted.
#
# l06's types live in Main; the module's live in Portfolio, so the two coexist without clashing
# provided we never `using` the module.

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const PF = BayesianFootball.Portfolio
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments

include("l06_portfolio_v2.jl")          # the reference implementation

# -------------------------------------------------------------------
# Data
# -------------------------------------------------------------------
const WARM = "/root/bf_review/warm.jls"

if !isdefined(Main, :ds); global ds = DD.load_datastore_cached(DD.ScottishLower()); end
if !isdefined(Main, :expr)
    global expr = EE.load_experiment(
        EE.list_experiments("./data/experiments/plus_minus_biweek", data_dir = ""), 3)
end
if !isdefined(Main, :odds) || !isdefined(Main, :latents_df)
    if isfile(WARM)
        @info "restoring warm odds + latents" WARM
        w = deserialize(WARM)
        global odds = w.odds
        global latents_df = w.latents_df
    else
        global odds = DD.summarize_betfair_market(ds, open_window = (-100000.0, -10.0),
                                                  close_window = (-20.0, 0.0))
        global latents_df = EE.extract_oos_predictions(ds, expr).df
    end
end

MK = DD.MarketConfig(reduce(vcat, (DD.AbstractMarket[DD.Market1X2(), DD.MarketBTTS()],
                                   [DD.MarketOverUnder(i + 0.5) for i in 0:4])))

# -------------------------------------------------------------------
# Build both
# -------------------------------------------------------------------
@info "building l06 reference books"
ref = @time build_books(latents_df, expr, odds, MK, ds;
                        cfg = PortfolioConfig(), shrink = ShrinkConfig(enabled = true, n_draws = 128))

@info "building src/Portfolio books"
spec = PF.BookSpec(markets = MK)                      # defaults match l06 exactly
new = @time PF.build_books(spec, latents_df, expr, odds, ds)

# -------------------------------------------------------------------
# 1. Book-level diff
# -------------------------------------------------------------------
println("\n", "="^80, "\n=== 1. BOOK PARITY ===\n", "="^80)

fails = String[]
chk(ok, msg) = ok ? nothing : push!(fails, msg)

chk(length(ref) == length(new), "book count $(length(ref)) vs $(length(new))")
chk([b.m_id for b in ref] == [b.m_id for b in new], "match ids / ordering differ")

d_stake, d_k, d_settle, d_odds, d_R = 0.0, 0.0, 0.0, 0.0, 0.0
for (r, n) in zip(ref, new)
    r.m_id == n.m_id || continue
    chk(length(r.sels) == length(n.sels), "selection count differs on $(r.m_id)")
    length(r.sels) == length(n.sels) || continue
    # l06 and the module iterate a Dict, so selection ORDER can differ -- compare by key
    perm = [findfirst(s -> (s.group, s.line, s.selection) ==
                           (t.group, t.line, t.selection), n.sels) for t in r.sels]
    chk(all(!isnothing, perm), "selection sets differ on $(r.m_id)")
    all(!isnothing, perm) || continue
    p = Int.(perm)
    d_stake  = max(d_stake,  maximum(abs.(r.a_kelly .- n.a_kelly[p]); init = 0.0))
    d_k      = max(d_k,      abs(r.k_bm - n.k_shrink))
    d_settle = max(d_settle, maximum(abs.(r.settle .- n.settle[p]); init = 0.0))
    d_odds   = max(d_odds,   maximum(abs.([s.odds_used for s in r.sels] .-
                                          [n.sels[i].odds_used for i in p]); init = 0.0))
    d_R      = max(d_R,      maximum(abs.(r.R .- n.R[:, p]); init = 0.0))
end

@printf("  max |Δ odds_used|   %.3e\n", d_odds)
@printf("  max |Δ payoff R|    %.3e\n", d_R)
@printf("  max |Δ settle|      %.3e\n", d_settle)
@printf("  max |Δ a_kelly|     %.3e\n", d_stake)
@printf("  max |Δ shrink k|    %.3e\n", d_k)
chk(d_odds  < 1e-12, "odds_used differ by $d_odds")
chk(d_R     < 1e-12, "payoff matrix differs by $d_R")
chk(d_settle< 1e-12, "settlement differs by $d_settle")
chk(d_stake < 1e-4,  "allocations differ by $d_stake")
chk(d_k     < 1e-9,  "shrinkage differs by $d_k")

# -------------------------------------------------------------------
# 2. Simulation diff across a policy grid
# -------------------------------------------------------------------
println("\n", "="^80, "\n=== 2. SIMULATION PARITY ===\n", "="^80)

ref_slates = build_slates(ref)
new_slates = PF.group(PF.DailySlate(), new)
chk(length(ref_slates) == length(new_slates), "slate count differs")

cmp = DataFrame(trust = Float64[], lambda = Float64[], bm = Bool[],
                ref_final = Float64[], new_final = Float64[], d_final = Float64[],
                ref_roi = Float64[], new_roi = Float64[], d_roi = Float64[],
                ref_mdd = Float64[], new_mdd = Float64[])

for w in (0.10, 0.25, 1.00), lam in (0.0, 10.0, 20.0), bm in (false, true)
    a_ref = Dict{String,Float64}(alpha_key(s) => w for b in ref for s in b.sels)
    r_sim = simulate(ref_slates, a_ref, PortfolioConfig(),
                     RiskConfig(lambda = lam, slate_cap = 0.25); use_bm = bm)
    r_m   = path_metrics(r_sim)

    pol = PF.PolicySpec(trust = PF.FlatTrust(w),
                        risk = lam > 0 ? PF.SlateDrawdown(lam) : PF.NoRisk(),
                        cap = PF.FixedCap(0.25))
    n_sim = PF.simulate(pol, new_slates; use_shrink = bm)
    n_m   = PF.path_metrics(n_sim)

    push!(cmp, (w, lam, bm, r_m.final, n_m.final, abs(r_m.final - n_m.final),
                r_m.roi, n_m.roi, abs(r_m.roi - n_m.roi), r_m.mdd, n_m.mdd))
end

for c in (:ref_final, :new_final, :ref_roi, :new_roi, :ref_mdd, :new_mdd)
    cmp[!, c] = round.(cmp[!, c], digits = 4)
end
cmp.d_final = round.(cmp.d_final, sigdigits = 3)
cmp.d_roi   = round.(cmp.d_roi, sigdigits = 3)
println(cmp)

chk(maximum(cmp.d_final) < 1e-3, "final bankroll differs by $(maximum(cmp.d_final))")
chk(maximum(cmp.d_roi)   < 1e-2, "ROI differs by $(maximum(cmp.d_roi)) pp")

# -------------------------------------------------------------------
# 3. Reference values from the audit session
# -------------------------------------------------------------------
println("\n", "="^80, "\n=== 3. HEADLINE NUMBERS ===\n", "="^80)

let kk = [b.kkt for b in new], ks = [b.k_shrink for b in new],
    de = [s.odds_used / s.odds_quoted for b in new for s in b.sels],
    stk = [sum(b.a_kelly) for b in new]
    @printf("  books / slates            %d / %d            (expect 628 / 99)\n",
            length(new), length(new_slates))
    @printf("  KKT median / p99          %.1e / %.1e   (expect 1.2e-6 / 3.3e-6)\n",
            median(kk), quantile(kk, 0.99))
    @printf("  de-arb shrunk / mean      %.1f%% / %.3f%%     (expect 41.5%% / 0.216%%)\n",
            100mean(de .< 1 - 1e-12), 100 * (1 - mean(de)))
    @printf("  Baker-McHale k* med/mean  %.3f / %.3f      (expect 0.640 / 0.584)\n",
            median(ks), mean(ks))
    @printf("  full-Kelly stake med/max  %.1f%% / %.1f%%     (expect 16.2%% / 97.1%%)\n",
            100median(stk), 100maximum(stk))
end

let pol = PF.PolicySpec(trust = PF.FlatTrust(0.25), risk = PF.SlateDrawdown(20.0),
                        cap = PF.FixedCap(0.25)),
    m = PF.path_metrics(PF.simulate(pol, new_slates; use_shrink = false))
    @printf("  flat .25 / λ20 / cap .25  ROI %.2f%%  final %.3fx  MDD %.1f%%",
            m.roi, m.final, m.mdd)
    println("   (expect 9.25% / 2.315x / -24.1%)")
end

# -------------------------------------------------------------------
println("\n", "="^80)
if isempty(fails)
    println("PARITY GATE: PASS")
else
    println("PARITY GATE: FAIL")
    for f in fails; println("  ✗ ", f); end
end
println("="^80)
