# current_development/matchday_2026_08_08/r03_replay_scot_lower.jl
#
# POST-MATCH-DAY FORENSICS -- Scottish League One + Two, Saturday 2026-08-08.
#
#   julia --project -t 16 current_development/matchday_2026_08_08/r03_replay_scot_lower.jl
#
# Or, in a warm REPL:
#   include("current_development/matchday_2026_08_08/r03_replay_scot_lower.jl")
#
# ---------------------------------------------------------------------------------------------
# WHAT THIS IS
# ---------------------------------------------------------------------------------------------
#
# `r02` priced Friday's card live and wrote a paper sheet. Nothing priced Saturday's -- but the
# order book was archived, the crosswalk was healthy, and the results are in. So Saturday can be
# re-decided at 21 instants in its final hour and every decision graded against what happened.
#
# THE SLATE. 10 fixtures, 5 in tournament 56 and 5 in 57, ALL kicking off 13:00 UTC. That is the
# ideal shape for a portfolio question: one settlement window, so `SlateDrawdown` and `FixedCap`
# bind across the entire card rather than across an arbitrary calendar grouping.
#
# THE MODEL. `scot_lower_funnel_20260807_012812`, trained the night before on 710 history matches
# plus the ONE match week of 26/27 that had been played. Read that number again before reading
# any edge on this sheet -- see view 5.
#
# ---------------------------------------------------------------------------------------------
# HOW TO READ THE OUTPUT, IN ORDER
# ---------------------------------------------------------------------------------------------
#
#   0  provenance      what the replay can and cannot see. Read first, always.
#   1  slate trace     exposure / k_risk / P&L as a function of time-to-kick-off
#   2  churn           would re-pricing every 3 minutes have whipsawed the book?
#   3  CLV             did the market move toward the model or away from it?
#   4  fill            could the sizes have been filled at the top of the book?
#   5  cold start      is the biggest edge just the model's ignorance?
#   6  policy sweep    was the policy the problem, or the model?
#
# ---------------------------------------------------------------------------------------------
# THE HONEST POWER STATEMENT, UP FRONT
# ---------------------------------------------------------------------------------------------
#
# n = 10 matches, one slate, one hour. The P&L numbers below are NOT evidence about edge; the
# Portfolio backtest's ROI confidence interval still includes zero over 628 matches, and one
# Saturday cannot do better than that. What this run IS evidence about: whether the pipeline
# produces a fillable, correctly-ticketed, sanely-sized book, and how that book behaves as a
# function of when you fire. Those are mechanical properties and one slate measures them fine.

using BayesianFootball
using DataFrames, Dates, CSV, Statistics, Printf

const Data = BayesianFootball.Data
const EXPR = BayesianFootball.Experiments
const MDX  = BayesianFootball.MatchDay
const PFX  = BayesianFootball.Portfolio

include(joinpath(@__DIR__, "l02_slate_replay.jl"))

const MATCH_DAY  = Date(2026, 8, 8)
const SEGMENT    = Data.ScottishLower()
const EXP_PATH   = "./data/matchday_wknd_0808/scot_lower_funnel_20260807_012812"
const OUT_DIR    = "./data/replays/scot_lower_20260808/"
const BANKROLL   = 1000.0
const LOOKBACK   = Minute(60)     # measured: the feed's first pre-KO tick is T-60. Not a choice.
const STEP       = Minute(3)      # measured: `order_book_1m` carries this slate every 3 minutes.
mkpath(OUT_DIR)

# ===================================================================
# 0. Provenance
# ===================================================================

fixtures, results = slate_from_db(Data.tournament_ids(SEGMENT), MATCH_DAY)
@info "slate" day = MATCH_DAY fixtures = length(fixtures) graded = length(results)

ds   = Data.load_datastore_cached(SEGMENT)
expr = EXPR.load_experiment(EXP_PATH)

println("\n", "="^100, "\n0. PROVENANCE -- what this replay can and cannot see\n", "="^100)

cov = book_coverage(fixtures)
show(cov, allrows = true, allcols = true); println()

println("""
  `n_snaps` is the number of distinct order-book instants at or before kick-off. If it is ~21 the
  feed covers the final hour at 3-minute resolution and the grid below is right. If it is ~240
  the drain was writing every minute and STEP should be Minute(1). If it is single digits the
  collector was struggling and every number downstream is thinner than it looks.""")

# --- the split the chain is actually paired with -------------------------------------------
#
# THIS IS THE CHECK THAT MATTERS MOST FOR LEAKAGE, and it is worth doing by hand rather than
# trusting the warning. `select_split` picks `idx = min(n_trained, n_bounds)`. The DataStore
# cache expires after 48h and rebuilds itself on load, so by the time you run a post-mortem the
# rebuilt boundary list has GROWN by one fold -- and that new fold's target window contains the
# very matches being priced.
#
# On this run: 2 training results, 3 rebuilt boundaries, so idx = 2, and boundary 2's target
# window holds match week 1 with none of Saturday's fixtures in it. The replay is leak-free.
#
# But notice WHY it is leak-free: because `min(2, 3) = 2`, not because anything compared a
# boundary against `as_of`. Retrain today and n_trained becomes 3, idx becomes 3, and the model
# prices Saturday off a fold whose target window IS Saturday. The principled fix is to select the
# split by time -- the latest boundary whose target window closes before `as_of` -- and it is the
# single most valuable change to `MatchDay.inference.jl`.
bounds = Data.create_id_boundaries(ds, expr.config.splitter)
idx    = min(length(expr.training_results), length(bounds))
slate_ids = Set(f.m_id for f in fixtures)
leak = length(intersect(slate_ids, Set(bounds[idx][1].target_match_ids)))

@printf("\n  trained folds %d | rebuilt boundaries %d | conditioning on split %d\n",
        length(expr.training_results), length(bounds), idx)
@printf("  target sizes %s\n", string([length(b[1].target_match_ids) for b in bounds]))
@printf("  slate fixtures inside split %d's TARGET window: %d  %s\n", idx, leak,
        leak == 0 ? "<- leak-free" : "<- LEAKAGE, the model was fitted on today's results")
leak == 0 || error("aborting: the chain was trained on the matches this replay is pricing")

# ===================================================================
# 1. The spec, the system, and the invariance assertion
# ===================================================================

spec = replay_spec(fixtures)

# Carried over verbatim from `r02`, so the replay measures the policy that was actually going to
# be run rather than a policy invented for the post-mortem:
#   FlatTrust(0.5)   the simulation's flat-w verdict
#   SlateDrawdown(23) ~20% drawdown budget at 1% probability
#   FixedCap(0.25)   the dominant lever; per-bet Kelly went bankrupt on the same book
#   KeepAll          NO curation on the primary run -- the curated cell is in the sweep at view 6,
#                    where it can be compared against its own control instead of asserted
sys = PFX.PortfolioSystem(
    PFX.BookSpec(markets = MDX.MatchDaySpec().markets),
    PFX.PolicySpec(trust  = PFX.FlatTrust(0.5),
                   risk   = PFX.SlateDrawdown(lambda = 23.0, mode = :sequential),
                   cap    = PFX.FixedCap(0.25),
                   filter = PFX.KeepAll()))

snaps = snapshot_grid(fixtures; lookback = LOOKBACK, step = STEP)
@info "snapshot grid" n = length(snaps) from = first(snaps) to = last(snaps)

# The funnel engine reads no lineup and no rating, so its latents MUST be identical at T-60 and
# T-0. Assert it: every number in views 1-4 is otherwise ambiguous between "the market moved" and
# "the model moved", and on this engine only one of those is possible.
ok, diff, col = latents_invariant(spec, expr, ds, fixtures, first(snaps), last(snaps))
@printf("\n  latents time-invariant: %s   (max |Δ| = %.3e on :%s)\n", ok, diff, col)
ok || @warn """the model's own view of these fixtures MOVED between T-60 and T-0. On the funnel
   engine that should be impossible. Every trace below now confounds model drift with market
   drift and must be read as the sum of the two."""

# ===================================================================
# 2. Replay
# ===================================================================

@info "replaying" snapshots = length(snaps) est_minutes = round(9.2 * length(snaps) / 60, digits = 1)
t0 = time()
out = replay(spec, sys, SEGMENT, expr, ds, snaps; bankroll = BANKROLL, results = results)
@info "replay done" minutes = round((time() - t0) / 60, digits = 1) legs = nrow(out.legs)

if isempty(out.legs)
    println("\nNO LEGS AT ANY SNAPSHOT. Read the blocked report before concluding 'no edge':")
    isempty(out.blocked) ? println("  ...and it is empty, so the gate passed and the model " *
                                   "genuinely wanted nothing.") :
                           show(out.blocked, allrows = true, allcols = true)
    error("nothing to analyse")
end

isempty(out.blocked) || begin
    println("\n--- GATE REFUSALS (a refusal is a value; read this before anything else) ---")
    show(combine(groupby(out.blocked, [:gate, :reason]), nrow => :snapshots), allcols = true)
    println()
end

# ===================================================================
# VIEW 1 -- how the portfolio adapts as t -> kick-off
# ===================================================================

println("\n", "="^100, "\n1. SLATE TRACE -- the portfolio as a function of time-to-kick-off\n", "="^100)

tr = copy(out.slate)
for c in (:exposure, :k_risk, :mean_edge, :risk_wtd_edge, :roi); tr[!, c] = round.(tr[!, c], digits = 4); end
for c in (:staked, :pnl); tr[!, c] = round.(tr[!, c], digits = 2); end
show(select(tr, :mins_to_ko, :fixtures, :legs, :lays, :exposure, :k_risk, :capped,
            :risk_wtd_edge, :staked, :pnl, :roi), allrows = true, allcols = true)
println()

println("""
  READING IT.
  * `exposure` is what fraction of the bankroll is simultaneously live. `capped = false` with a
    low `k_risk` means the DRAWDOWN BUDGET is what is holding the book down, not the hard cap.
    That distinction decides which knob to turn: trust and shrinkage cannot resize a book once
    `risk_factor` binds (it is homogeneous of degree 0) -- they only reshape it. Move `lambda`.
  * `risk_wtd_edge` falling toward kick-off is the market absorbing information the model does
    not have. Rising is the market drifting away from a model that has not changed. On this
    engine the model genuinely has not changed, so the whole series is market movement.
  * `pnl` by entry time is the closest thing here to an operational answer, and it is one draw
    from a very wide distribution. Do not tune on it.""")

println("\n--- by market family ---")
ft = family_trace(out.legs)
for c in (:risk, :pnl); ft[!, c] = round.(ft[!, c], digits = 2); end
ft.roi = round.(ft.roi, digits = 3)
show(unstack(select(ft, :as_of, :group, :roi), :as_of, :group, :roi), allrows = true, allcols = true)
println()
println("\n--- family totals at the CLOSE ---")
show(combine(groupby(out.legs[out.legs.as_of .== last(snaps), :], :group),
             nrow => :legs, :risk => (x -> round(sum(x), digits = 2)) => :risk,
             :pnl => (x -> round(sum(x), digits = 2)) => :pnl), allcols = true)
println()

# ===================================================================
# VIEW 2 -- churn
# ===================================================================

println("\n", "="^100, "\n2. CHURN -- would re-pricing every 3 minutes have whipsawed the book?\n", "="^100)

ch = churn(out.legs)
ch.jaccard = round.(ch.jaccard, digits = 3)
ch.risk_turnover = round.(ch.risk_turnover, digits = 3)
show(ch, allrows = true, allcols = true); println()

@printf("\n  median leg-set overlap between consecutive snapshots : %.3f\n", median(ch.jaccard))
@printf("  median risk turnover per 3 minutes                   : %.3f\n", median(ch.risk_turnover))
println("""
  A high overlap with high turnover means the SELECTIONS are stable but the SIZES are not -- the
  allocator is chasing small price moves. That costs spread on every re-price and buys nothing,
  and it is an argument for firing once rather than for a tighter cadence.""")

# ===================================================================
# VIEW 3 -- CLV
# ===================================================================

println("\n", "="^100, "\n3. CLV -- did the market come toward the model?\n", "="^100)

clv_legs, clv_fam = clv_vs_close(out)
if isempty(clv_legs)
    println("  no leg was priced at both an early snapshot and the close.")
else
    show(clv_fam, allcols = true); println()
    @printf("\n  overall: %.0f%% of legs moved toward us, median move %.2f%%, risk-weighted %.2f%%\n",
            100 * mean(clv_legs.toward_us), median(clv_legs.move_pct),
            sum(clv_legs.move_pct .* clv_legs.risk) / sum(clv_legs.risk))
    println("\n  by entry time (does firing early get a better price than firing at the bell?):")
    bt = combine(groupby(clv_legs, :mins_to_ko), nrow => :legs,
                 :toward_us => (x -> round(100mean(x), digits = 1)) => :pct_toward_us,
                 :move_pct => (x -> round(median(x), digits = 2)) => :median_move)
    show(sort(bt, :mins_to_ko, rev = true), allrows = true, allcols = true); println()
    println("""
  CONVENTION: a price that SHORTENS after we took it (`move_pct < 0`) means the market came
  toward us -- positive CLV. Both sides of the comparison are EFFECTIVE odds, so a leg that
  flipped from a direct back to a synthetic between the two instants is still comparable; a
  venue-price comparison would not be.

  At n = 10 matches CLV is the higher-powered measurement and P&L is close to uninformative.
  Weight your conclusions accordingly.""")
end

# ===================================================================
# VIEW 4 -- fill feasibility
# ===================================================================

println("\n", "="^100, "\n4. FILL -- could these sizes actually have been filled?\n", "="^100)

fill_legs, fill_snap = fill_report(out)
if isempty(fill_legs)
    println("  no depth captured.")
else
    fs = copy(fill_snap)
    for c in (:mean_fill, :risk_wtd_fill); fs[!, c] = round.(fs[!, c], digits = 3); end
    show(fs, allrows = true, allcols = true); println()

    cl = fill_legs[fill_legs.as_of .== last(snaps), :]
    println("\n  --- worst-filled legs at the close ---")
    w = sort(select(cl, :match_id, :group, :line, :selection, :venue_selection, :side,
                    :venue_stake, :available, :fill_ratio), :fill_ratio)
    for c in (:venue_stake, :available, :fill_ratio); w[!, c] = round.(w[!, c], digits = 2); end
    show(first(w, 10), allrows = true, allcols = true); println()

    @printf("\n  at the close: %d/%d legs fully fillable at the top of book, risk-weighted fill %.1f%%\n",
            count(x -> !isnan(x) && x >= 0.999, cl.fill_ratio), nrow(cl),
            100 * sum(ifelse.(isnan.(cl.fill_ratio), 0.0, cl.fill_ratio) .* cl.risk) / sum(cl.risk))
    println("""
  `available` is the size at the TOP of book only, so this is a lower bound -- walking one or two
  levels down gets more on at a worse price. But on Scottish League One and Two the totals and
  BTTS books are thin enough that the curated policy (totals + BTTS only) is precisely the part
  of the book that cannot absorb size, while the 1X2 market that curation removes is the only
  liquid one. That tension is real and this table is where you see it.

  NOTE what is NOT modelled anywhere in src: nothing reads `back_size`. `BestAvailable` takes the
  price and ignores the depth, and `max_leverage` rejects implausible synthetics on PRICE alone.
  A per-selection size cap belongs in `AbstractQuoteRule` or as a new `AbstractSelectionFilter`.""")
end

# ===================================================================
# VIEW 5 -- cold start
# ===================================================================

println("\n", "="^100, "\n5. COLD START -- is the biggest edge the model's ignorance?\n", "="^100)

dv = divergence_vs_experience(out, ds, fixtures, last(snaps))
dv.max_abs_edge = round.(dv.max_abs_edge, digits = 3)
dv.risk = round.(dv.risk, digits = 2); dv.pnl = round.(dv.pnl, digits = 2)
show(dv, allrows = true, allcols = true); println()

println("""
  `min_team_matches` counts appearances inside the DataStore BEFORE match day. A fixture at the
  top of this table (largest disagreement) with a small `min_team_matches` is the model pricing a
  team it has barely seen, not an edge -- and in August that is every promoted and relegated club.
  Ross County and Airdrieonians came down 55 -> 56 and their entire pre-Saturday history inside a
  ScottishLower [56, 57] store is one match week.

  This is a real structural limitation of segment-scoped pooling, and the fix is upstream of
  everything in this file: either pool the tier boundary into the DataStore (as `ScottishUpper`
  and `IrelandAll` do for their own promotions), or carry a prior from the higher tier. Until
  then, an August slate should probably not be staked at all on cold-start fixtures, and a filter
  on `min_team_matches` is a one-line policy that would express that.""")

# ===================================================================
# VIEW 6 -- policy sweep at the close
# ===================================================================

println("\n", "="^100, "\n6. POLICY SWEEP at the closing book\n", "="^100)

wl   = totals_btts_whitelist()
grid = Pair{String,Any}[
    "base trust.5 cap.25 λ23"        => sys.policy,
    "curated totals+BTTS"            => PFX.PolicySpec(trust = PFX.FlatTrust(0.5),
                                                       risk = PFX.SlateDrawdown(lambda = 23.0, mode = :sequential),
                                                       cap = PFX.FixedCap(0.25), filter = wl),
    "cap 0.10"                       => PFX.PolicySpec(trust = PFX.FlatTrust(0.5),
                                                       risk = PFX.SlateDrawdown(lambda = 23.0, mode = :sequential),
                                                       cap = PFX.FixedCap(0.10), filter = PFX.KeepAll()),
    "cap 0.50"                       => PFX.PolicySpec(trust = PFX.FlatTrust(0.5),
                                                       risk = PFX.SlateDrawdown(lambda = 23.0, mode = :sequential),
                                                       cap = PFX.FixedCap(0.50), filter = PFX.KeepAll()),
    "λ 10 (tighter drawdown)"        => PFX.PolicySpec(trust = PFX.FlatTrust(0.5),
                                                       risk = PFX.SlateDrawdown(lambda = 10.0, mode = :sequential),
                                                       cap = PFX.FixedCap(0.25), filter = PFX.KeepAll()),
    "λ 40 (looser drawdown)"         => PFX.PolicySpec(trust = PFX.FlatTrust(0.5),
                                                       risk = PFX.SlateDrawdown(lambda = 40.0, mode = :sequential),
                                                       cap = PFX.FixedCap(0.25), filter = PFX.KeepAll()),
    "trust 0.25"                     => PFX.PolicySpec(trust = PFX.FlatTrust(0.25),
                                                       risk = PFX.SlateDrawdown(lambda = 23.0, mode = :sequential),
                                                       cap = PFX.FixedCap(0.25), filter = PFX.KeepAll()),
    "isolated drawdown (per match)"  => PFX.PolicySpec(trust = PFX.FlatTrust(0.5),
                                                       risk = PFX.IsolatedDrawdown(23.0),
                                                       cap = PFX.FixedCap(0.25), filter = PFX.KeepAll()),
    # The two cells the code review argues for, rather than the ones r02 happens to run:
    #
    # :joint  -- every fixture on this slate kicks off at 13:00, so there is no rebalancing
    #            between them. `:sequential` solves sum_t log E[(1+kR_t)^-λ] <= 0, which is the
    #            constraint for matches compounding one AFTER the other. `:joint` Monte-Carlos
    #            the simultaneous sum, which is what actually happens here.
    #
    # trust   -- `MarketWhitelist` runs AFTER the cap, so curating 1X2 out TRUNCATES the book:
    #            the survivors keep sizes solved for a portfolio that still contained 1X2, and
    #            the freed capacity is simply not used. Per-family trust is applied BEFORE the
    #            allocator, so the drawdown budget re-expands what is left. The staking-sim
    #            "curated per-line w" result is a trust model, not a filter, and this is the
    #            cell that tests whether the distinction is worth anything.
    "joint drawdown (simultaneous)"  => PFX.PolicySpec(trust = PFX.FlatTrust(0.5),
                                                       risk = PFX.SlateDrawdown(lambda = 23.0, mode = :joint),
                                                       cap = PFX.FixedCap(0.25), filter = PFX.KeepAll()),
    "per-family trust (1X2 -> 0)"    => PFX.PolicySpec(trust = family_trust(),
                                                       risk = PFX.SlateDrawdown(lambda = 23.0, mode = :sequential),
                                                       cap = PFX.FixedCap(0.25), filter = PFX.KeepAll()),
]

sw = policy_sweep(sys, expr, out.close, results, grid;
                  bankroll = BANKROLL, instruments = out.close.instruments)
for c in (:staked, :pnl); sw[!, c] = round.(sw[!, c], digits = 2); end
for c in (:exposure, :roi, :growth, :k_risk); sw[!, c] = round.(sw[!, c], digits = 4); end
show(sw, allrows = true, allcols = true); println()

println("""
  JUDGE ON `growth`, NOT `roi`. ROI is P/L divided by stake, so a uniform scaling of every stake
  cancels out of it exactly -- every flat-trust cell will report the same ROI and a very different
  bankroll outcome. `trust 0.25` versus the base cell is the control for that: if their ROIs match
  and their growths do not, the arithmetic is behaving as documented.

  Also expect `trust 0.25` to change NOTHING at all if the drawdown constraint is binding, for
  the homogeneity reason in view 1. Two cells that are identical is information, not a bug.""")

# ===================================================================
# 7. Artefacts
# ===================================================================

stamp = Dates.format(MATCH_DAY, "yyyymmdd")
CSV.write(joinpath(OUT_DIR, "legs_$stamp.csv"),     out.legs)
CSV.write(joinpath(OUT_DIR, "quotes_$stamp.csv"),   out.quotes)
CSV.write(joinpath(OUT_DIR, "depth_$stamp.csv"),    out.depth)
CSV.write(joinpath(OUT_DIR, "slate_$stamp.csv"),    out.slate)
isempty(clv_legs)  || CSV.write(joinpath(OUT_DIR, "clv_$stamp.csv"),  clv_legs)
isempty(fill_legs) || CSV.write(joinpath(OUT_DIR, "fill_$stamp.csv"), fill_legs)
CSV.write(joinpath(OUT_DIR, "policy_sweep_$stamp.csv"), sw)

# The CORRECTED order tickets for the closing sheet. Deliberately not `MatchDay.order_ticket`:
# that function names a synthetic leg by the position wanted while quoting the complement's side
# and price, so acting on it places the opposite bet. See `venue_leg` in l02.
close_sheet = out.legs[out.legs.as_of .== last(snaps), :]
tickets = DataFrame([merge((match_id = r.match_id, market = r.group, line = r.line,
                            model_selection = r.selection), venue_leg(r))
                     for r in eachrow(close_sheet)])
CSV.write(joinpath(OUT_DIR, "tickets_corrected_$stamp.csv"), tickets)

n_synth = count(==(:lay), close_sheet.side)
println("\n", "="^100)
@printf("  corrected tickets written: %d legs, of which %d (%.0f%%) are SYNTHETIC and are named\n",
        nrow(tickets), n_synth, 100 * n_synth / nrow(tickets))
println("  differently by MatchDay.order_ticket than by venue_leg. On a synthetic, order_ticket")
println("  emits (selection = the position you want, side = :lay, price = the COMPLEMENT's lay")
println("  price). Executing that places the opposite position. Fix: give `Instrument` a")
println("  `venue_key` field and have `order_ticket` read it.")
@info "written" dir = OUT_DIR
