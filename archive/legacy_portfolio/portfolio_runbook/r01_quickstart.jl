# r01_quickstart.jl -- the whole pipeline, end to end.
#
# Run this first. Everything else in the runbook is a variation on these five calls.
#
#     latents + quotes --build_books--> books --group--> slates --simulate--> trajectory --report
#
# Read the printed output alongside the comments; the point is to see what each stage HOLDS,
# not just what it returns.

include("_setup.jl")
include("current_development/portfolio_runbook/_setup.jl")

# ===================================================================
# 1. Describe the system you want to run
# ===================================================================
#
# Two configs, and the split between them is the design (see README).

spec = PF.BookSpec(
    markets   = MARKETS,
    price     = PF.DeArb(),              # how a traded price becomes a settlement price
    allocator = PF.KellyLogUtility(),    # how a belief + payoffs become stakes
    shrink    = PF.BakerMcHale(),        # parameter-uncertainty correction (128 posterior draws)
    exec      = PF.ExecutionConfig(      # exchange + sizing constraints
                    commission = PF.PerBetCommission(0.02),
                    max_selection_stake = 0.50,   # no single bet above 50% of bankroll
                    budget = 0.99),               # no single MATCH above 99%
)

policy = PF.PolicySpec(
    trust    = PF.FlatTrust(0.25),       # believe the model 25%, the market 75%
    risk     = PF.SlateDrawdown(23.0),   # lambda; ~= 20% drawdown at 1% probability, see below
    cap      = PF.FixedCap(0.10),        # never risk >25% of bankroll on one settlement window
    filter   = PF.KeepAll(),             # no curation
    grouping = PF.DailySlate(),          # matches on the same date settle together
)

# Where does lambda = 23 come from? risk_lambda(D, beta) = log(beta)/log(D).
# D = 0.8 (tolerate a 20% drawdown), beta = 0.01 (with at most 1% probability) gives 20.6.
# We use 23 because realised drawdown overshoots the nominal by a stable ~1.15x -- the bound
# holds under the MODEL's measure, and the model is optimistic.
@printf("\nlambda for a 20%% drawdown at 1%% probability: %.1f  (we use %.1f, see comment)\n",
        PF.risk_lambda(0.8, 0.01), policy.risk.lambda)

# ===================================================================
# 2. Build the books  -- THE EXPENSIVE STEP
# ===================================================================
#
# One MatchBook per match. Per match this: computes the posterior score grid, prices every
# quoted market, builds the payoff matrix, solves the Kelly portfolio, and then re-solves it
# on 128 posterior draws to find the Baker-McHale shrinkage. Threaded over matches.
#
# NOTE what is NOT in here: nothing from `policy`. That is what makes books cacheable.

@info "building books" cache_key = PF.book_cache_key(spec)
books = @time PF.build_books(spec, latents_df, expr, odds, ds)

println("\n", "="^70, "\n=== WHAT A MatchBook HOLDS ===\n", "="^70)
b = books[1]
println("  match           ", b.m_id, "  on ", b.date)
println("  selections      ", length(b.sels), "  ", [s.family for s in b.sels])
println("  p_grid          ", length(b.p_grid), "-vector, sums to ", round(sum(b.p_grid), digits=12))
println("  R (payoffs)     ", size(b.R), "  = (score states) x (selections)")
println("  settle          ", round.(b.settle, digits=3), "   <- realised payoff per unit staked")
println("  a_kelly         ", round.(b.a_kelly, digits=4), "   <- FULL Kelly, before any policy")
println("  k_shrink        ", b.k_shrink, "   <- Baker-McHale factor for this match")
println("  kkt             ", round(b.kkt, sigdigits=3), "   <- solver quality; want ~1e-6")

# a_kelly is deliberately enormous -- it is undiluted Kelly on a 144-state grid. The policy
# layer is what makes it survivable.
@printf("\n  full-Kelly stake per match: median %.1f%%, max %.1f%% of bankroll\n",
        100median(sum(bk.a_kelly) for bk in books), 100maximum(sum(bk.a_kelly) for bk in books))

# ===================================================================
# 3. Group into settlement windows
# ===================================================================
#
# A "slate" is a set of matches that resolve against the SAME bankroll. This is the boundary
# that makes simultaneous exposure meaningful -- 8 matches at 3pm on Saturday are one bet, not
# eight sequential ones.

slates = PF.group(policy.grouping, books)
@printf("\n%d books -> %d slates | matches per slate: median %d, max %d\n",
        length(books), length(slates),
        median(length(s.books) for s in slates), maximum(length(s.books) for s in slates))

# ===================================================================
# 4. Simulate
# ===================================================================
#
# Walks the slates forward in time. For each one: apply trust, apply shrinkage, solve the
# drawdown budget, cap exposure, filter, settle, compound once.

traj = PF.simulate(policy, slates)

println("\n", "="^70, "\n=== WHAT A Trajectory HOLDS ===\n", "="^70)
println("  bankroll     ", length(traj.bankroll), " points, ends at ", round(traj.bankroll[end], digits=3))
println("  slate_pl     per-slate P/L as a fraction of bankroll")
println("  k_risk       the drawdown factor chosen for each slate")
println("  exposure     total stake settled simultaneously, per slate")
println("  n_capped     ", traj.n_capped, " slates hit the hard exposure cap")
println("  bets         ", nrow(traj.bets), " rows -- every stake placed")
println("\n  first three bets:")
println(first(traj.bets, 3))

# ===================================================================
# 5. Report
# ===================================================================

m = PF.path_metrics(traj)
println("\n", "="^70, "\n=== RESULT ===\n", "="^70)
@printf("  final bankroll     %.3fx\n", m.final)
@printf("  flat ROI           %.2f%%\n", m.roi)
@printf("  log-growth / slate %.5f\n", m.growth_per_slate)
@printf("  max drawdown       %.1f%%\n", m.mdd)
@printf("  mean exposure      %.1f%% of bankroll per slate (max %.1f%%)\n",
        100m.mean_exposure, 100m.max_exposure)
@printf("  worst slate        %.1f%%\n", 100m.worst_slate)
@printf("  mean risk factor   %.3f   <- how hard the drawdown budget is squeezing\n", m.mean_k_risk)

# `report` adds any BackTesting wealth metric, evaluated on the slate-level bankroll curve.
r = PF.report(traj, [BT.SharpeRatio(), BT.CalmarRatio(), BT.SortinoRatio()])
@printf("\n  Sharpe %.3f | Calmar %.3f | Sortino %.3f\n",
        r.SharpeRatio, r.CalmarRatio, r.SortinoRatio)
@printf("  ROI 95%% CI (clustered by match): [%.2f%%, %.2f%%]\n", r.roi_ci_lo, r.roi_ci_hi)
println("""
  ^ that interval is the number that matters. It is resampled BY MATCH, because several bets
    on one match share one scoreline. If it straddles zero, the strategy is not distinguishable
    from luck on this sample -- no matter how good the final bankroll looks.""")

# Keep `books` and `slates` in the REPL: r02 reuses them.
@info "done -- `books` and `slates` are now in scope for r02_policy_sweep.jl"
