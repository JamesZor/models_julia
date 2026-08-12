# current_development/orderbook_layer2/r04_entry_timing.jl
#
# WP4. When, before kick-off, should the book be touched?
#
# ---------------------------------------------------------------------------------------------
# PRE-REGISTRATION  (written before the runner was executed; see NOTES.md for the dated entry)
# ---------------------------------------------------------------------------------------------
#
# The plan's original hypothesis was: "executable cost falls monotonically toward kickoff
# (spread 4.6% -> 2.6%, matched £295 -> £10.6k) while the model's edge is largest early, so the
# optimum is INTERIOR, expected T-120 to T-30."
#
# **That hypothesis is retired before it is tested.** It rested on a liquidity table built with
# `avg()` over `market_matched`, a column that is NULL for 23.8% of rows before 2026-08-02 — so
# the table described 18 August fixtures, not the corpus (WP0 Correction 2). Re-measured with
# medians over all 81 matches (WP0 Correction 3):
#
#   MATCH_ODDS  relative spread   2.86% at T-60  ->  2.27% at the off
#   O/U 2.5     relative spread   ~1.9-2.1%, FLAT from T-240 to KO
#   top-of-book size              £1,906  ->  £7,641   (quadruples)
#
# So the revised, and much less exciting, pre-registration:
#
#   H1  PRICE is nearly free to wait for. Spread compresses by ~0.6pp on 1X2 and not at all on
#       totals. `PriceDrift` should be ~0 with a CI spanning zero, i.e. the book has no
#       systematic pre-kickoff drift a backer can harvest.
#   H2  CAPACITY is what the clock buys. `FillCost` should be the only estimator with a strong,
#       monotone entry-time gradient — early entry cannot absorb a stake that late entry can.
#   H3  Because of H1, the `BestPrice` oracle gap should be almost entirely HINDSIGHT NOISE.
#       This is the prediction that could most easily fool us, so it gets its own control:
#       `RandomEntry` (see below). Predicted: BestPrice >> AtClose, RandomEntry ~= AtClose.
#   H4  `live - frozen` (team news) should favour LATER entry, and more in 1X2 than in totals,
#       because an announced XI moves supremacy more than it moves total goals.
#
# **If H1-H3 hold, the headline of this work package is a negative result**: entry time is not a
# price lever on this corpus, only a size lever. That is worth pre-committing to, because the
# oracle gap will be large and positive and it would be very easy to write it up as an
# opportunity.
#
# ---------------------------------------------------------------------------------------------
# THE ORACLE TRAP, AND THE CONTROL FOR IT
# ---------------------------------------------------------------------------------------------
#
# `BestPrice` takes the maximum odds over every snapshot of a leg. Under a driftless random walk
# the maximum of N draws exceeds the last draw ALWAYS, and by an amount that grows with N. Since
# N here is set by our own grid (~28 instants/slate), a naive oracle gap measures the grid, not
# the market.
#
# `RandomEntry(seed)` fires each leg at a uniformly random instant, so under a random walk it
# matches the close in expectation. The contrast identifies the trend:
#
#     BestPrice >> AtClose,  RandomEntry ~= AtClose   =>  no drift; the gap is unharvestable
#     BestPrice >> AtClose,  RandomEntry >  AtClose   =>  a real drift toward kickoff
#
# Several seeds are averaged, because a single draw of a deliberately noisy rule is one sample.
#
# ---------------------------------------------------------------------------------------------
# THE THREE ESTIMATORS, IN DESCENDING ORDER OF POWER
# ---------------------------------------------------------------------------------------------
#
#   PriceDrift        ~100k quote-level obs, NO MODEL      -> when is the book cheapest
#   ClosingLineValue  ~1-2k legs                           -> does the market confirm our picks
#   wealth + hurdle   ~500 graded legs, directional only   -> did it make money
#
# The corpus is 81 matches. Every CI is a match-clustered bootstrap and every one of them will be
# wide. Read the top row and treat the bottom one as confirmation of sign at best.
#
# ---------------------------------------------------------------------------------------------
# USAGE (server; needs WP2's pins and WP3's gate green)
# ---------------------------------------------------------------------------------------------
#
#   include("current_development/orderbook_layer2/r04_entry_timing.jl")

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const PF = BayesianFootball.Portfolio
const MD = BayesianFootball.MatchDay
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments
const BT = BayesianFootball.BackTesting

include(joinpath(@__DIR__, "l00_corpus.jl"))
include(joinpath(@__DIR__, "l01_l2_experiment.jl"))
include(joinpath(@__DIR__, "l02_l2_ledger.jl"))
include(joinpath(@__DIR__, "l03_l2_metrics.jl"))
include(joinpath(@__DIR__, "..", "matchday_2026_08_08", "l02_slate_replay.jl"))
include(joinpath(@__DIR__, "l04_corpus_replay.jl"))

const ENGINE_DIR = "./data/l2_ireland_engines"
const OUT_DIR    = "./data/l2_entry_timing"

# ===================================================================
# 0. Setup
# ===================================================================

pinned(tag) = deserialize(joinpath(ENGINE_DIR, "ds_$(tag).jls"))

function newest_experiment(prefix::String)
    dirs = filter(d -> startswith(basename(d), prefix),
                  [joinpath(ENGINE_DIR, d) for d in readdir(ENGINE_DIR)
                   if isdir(joinpath(ENGINE_DIR, d))])
    isempty(dirs) && error("r04: no experiment under $ENGINE_DIR matching $prefix")
    return EE.load_experiment(sort(dirs, by = mtime, rev = true)[1])
end

"""
The reference book and policy. Held FIXED across every entry rule — this work package varies the
clock and nothing else. WP5 varies `trust`, WP6 varies the filter; mixing axes here would repeat
`r08_vector_alpha_optim.jl`'s mistake at smaller scale.

`SlateDrawdown(23.0)` is the runbook default and it BINDS, which means absolute stake levels are
set by the risk model rather than by the entry rule. That is fine for a timing comparison — every
arm is capped identically — but it is why WP5 has to run two risk regimes and this one does not.
"""
reference_system() = PF.PortfolioSystem(
    PF.BookSpec(markets = DD.MarketConfig(DD.AbstractMarket[
                    DD.Market1X2(), DD.MarketBTTS(),
                    (DD.MarketOverUnder(l) for l in (0.5, 1.5, 2.5, 3.5, 4.5))...])),
    PF.PolicySpec(trust = PF.FlatTrust(0.25), risk = PF.SlateDrawdown(23.0),
                  cap = PF.FixedCap(0.25)))

"""
The entry ladder.

`FixedLead` rungs are placed where the WP0 measurements say something might change: dense inside
the last hour (where spread does compress), sparse outside it (where it is flat). `AtClose` is
the baseline; `BestPrice` is the upper bound; `RandomEntry` is the null that tells them apart.
"""
entry_ladder() = AbstractEntryRule[
    AtClose(),
    FixedLead(Minute(5)),   FixedLead(Minute(15)),  FixedLead(Minute(30)),
    FixedLead(Minute(60)),  FixedLead(Minute(90)),  FixedLead(Minute(120)),
    FixedLead(Minute(180)),
    FirstQualifying(0.02; max_lead = Minute(180)),
    FirstQualifying(0.05; max_lead = Minute(180)),
    RandomEntry(1; max_lead = Minute(180)),
    RandomEntry(2; max_lead = Minute(180)),
    RandomEntry(3; max_lead = Minute(180)),
    BestPrice(; max_lead = Minute(180)),
]

# ===================================================================
# 1. Build the Tier-1 caches
# ===================================================================

"""
    build_arm(corpus, expr, ds, arm) -> L2Snapshots

Tier 1 for one league and one latent arm. This is the only expensive call in the file; everything
downstream is a `groupby` over its output.
"""
function build_arm(corpus, expr, ds, arm::Symbol)
    @printf("\n--- Tier 1: %s, arm :%s ---\n", corpus.name, arm)
    t0 = time()
    s = build_snapshots(corpus, expr, ds; arm = arm, lookback = Minute(180), verbose = true)
    @printf("    %d snapshots over %d slates in %.1f min\n",
            length(s), length(unique(x.slate_day for x in s.snaps)), (time() - t0) / 60)
    return s
end

# ===================================================================
# 2. Tier 2/3: one ledger, many readings
# ===================================================================

"""
    entry_ledger(snaps, sys, expr) -> Layer2Ledger

Stake ONCE, then apply every entry rule to the same staked ledger and stack the results.

Staking is deterministic given `(snapshot, system)`, so re-staking per rule would produce
identical numbers at 14x the cost. Stacking instead means every arm is compared on literally the
same rows, which removes one way the comparison could go wrong.
"""
function entry_ledger(snaps::L2Snapshots, sys, expr)
    full = stake_snapshots(snaps, sys, expr; bankroll = 1.0, policy_name = "reference")
    isempty(full) && error("entry_ledger: staking produced no legs")
    @printf("    staked %d leg-instants over %d legs\n", nrow(full.df),
            nrow(unique(full.df, [:match_id, :group, :line, :selection])))

    parts = DataFrame[]
    for rule in entry_ladder()
        picked = apply_entry(rule, full.df)
        isempty(picked) && continue
        recap_slates!(picked, cap_fraction(sys.policy.cap))
        picked.entry_name = fill(entry_name(rule), nrow(picked))
        picked.is_oracle  = fill(rule isa BestPrice, nrow(picked))
        picked.is_control = fill(rule isa RandomEntry, nrow(picked))
        push!(parts, picked)
    end
    return Layer2Ledger(reduce(vcat, parts; cols = :union))
end

# ===================================================================
# 3. The four readings
# ===================================================================

"""
    reading_1_drift(snaps) -> DataFrame

H1, at quote level and with no model in the loop.

Every quoted selection at every instant is one observation of `log(odds_close_final / odds_t)`,
so this is ~100k rows rather than the ~500 the P&L reading gets. If the book has no pre-kickoff
drift, `drift_mean` is ~0 in every bucket and its CI spans zero.

Deliberately built from the SNAPSHOT odds rather than from the staked ledger: the staked ledger
only contains legs the model wanted, which is a model-conditioned sample of the book.
"""
function reading_1_drift(snaps::L2Snapshots)
    rows = DataFrame[]
    for s in snaps.snaps
        d = select(s.odds, :match_id, :market_name => :group, :market_line => :line,
                   :selection, :odds_close => :odds)
        d.as_of = fill(s.as_of, nrow(d))
        push!(rows, d)
    end
    q = reduce(vcat, rows; cols = :union)

    ko = Dict{Int,DateTime}()
    for s in snaps.snaps, (mid, fi) in s.fixtures
        haskey(ko, mid) || (ko[mid] = _fixture_kickoff(fi, s))
    end
    q = filter(r -> haskey(ko, r.match_id), q)
    q.mins_to_ko = [Dates.value(ko[m] - a) / 60_000 for (m, a) in zip(q.match_id, q.as_of)]
    add_entry_buckets!(q)

    # stamp the close the same way the ledger does, so the two readings cannot disagree
    lastq, lastt = Dict{NTuple{4,Any},Float64}(), Dict{NTuple{4,Any},DateTime}()
    for r in eachrow(q)
        k = (r.match_id, String(r.group), Float64(r.line), Symbol(r.selection))
        if !haskey(lastt, k) || r.as_of > lastt[k]
            lastt[k] = r.as_of; lastq[k] = Float64(r.odds)
        end
    end
    q.odds_close_final = [get(lastq, (r.match_id, String(r.group), Float64(r.line),
                                      Symbol(r.selection)), NaN) for r in eachrow(q)]
    q.side = fill(:back, nrow(q))

    out = combine(groupby(q, :entry_bucket)) do sub
        merge((n_quotes = nrow(sub),),
              BT.compute_distributional_metric(PriceDrift(), sub))
    end
    return sort!(out, :entry_bucket)
end

"""
    reading_2_clv(led) -> DataFrame

H1/H4 at leg level: does the market subsequently move TOWARD the model's picks?

Grouped by entry rule and by market family, because the prior from r21 is that the two disagree —
totals/BTTS positive, 1X2 flat-to-negative (`hurdle_G` = -0.042 on home).
"""
reading_2_clv(led::Layer2Ledger) =
    l2_tearsheet(led; groupby_cols = [:entry_name, :family],
                 dist_metrics = [ClosingLineValue()], bootstrap = false)

"""
    reading_3_wealth(led) -> DataFrame

H2/H3 with money attached — the weakest reading, and the one whose CI matters more than its
point estimate.
"""
reading_3_wealth(led::Layer2Ledger) = l2_tearsheet(led; groupby_cols = [:entry_name])

"""
    reading_4_oracle(ts) -> DataFrame

H3. Lines up the oracle, the null control and the baseline so the trap in the header is settled
on one row rather than by eye across a table.

`oracle_gap` is what a naive read would call the timing opportunity. `control_gap` is how much of
it a rule with NO information also collects. The difference is the only part that could be real.
"""
function reading_4_oracle(ts::DataFrame)
    getrow(n) = (i = findfirst(==(n), ts.entry_name); i === nothing ? nothing : ts[i, :])
    base = getrow("AtClose")
    orc  = getrow("BestPrice(oracle)")
    ctl  = filter(r -> startswith(r.entry_name, "RandomEntry"), ts)
    (base === nothing || orc === nothing) && return DataFrame()

    ctl_roi = isempty(ctl) ? NaN : mean(ctl.roi)
    return DataFrame(
        at_close_roi   = base.roi,
        oracle_roi     = orc.roi,
        control_roi    = ctl_roi,
        oracle_gap     = orc.roi - base.roi,
        control_gap    = ctl_roi - base.roi,
        harvestable    = (orc.roi - base.roi) - (ctl_roi - base.roi),
        n_control_seeds = nrow(ctl),
        verdict = abs(ctl_roi - base.roi) < 0.25 * abs(orc.roi - base.roi) ?
                  "oracle gap is mostly HINDSIGHT (H3 holds)" :
                  "control also beats close -> a real drift toward kickoff (H3 fails)")
end

"""
    reading_5_teamnews(frozen_ts, live_ts) -> DataFrame

H4. `live - frozen`, per entry rule.

The frozen arm holds latents fixed so all movement is the book; the live arm recomputes them, so
the difference is exactly the value of having waited for the announced XI. Nothing else in the
two runs differs.
"""
function reading_5_teamnews(fz::DataFrame, lv::DataFrame)
    j = innerjoin(select(fz, :entry_name, :roi => :roi_frozen, :final => :final_frozen),
                  select(lv, :entry_name, :roi => :roi_live,   :final => :final_live),
                  on = :entry_name)
    j.roi_delta   = j.roi_live   .- j.roi_frozen
    j.final_ratio = j.final_live ./ j.final_frozen
    return sort!(j, :roi_delta, rev = true)
end

# ===================================================================
# 4. Per-league driver
# ===================================================================

function run_league(tag::String, tid::Int, corpus_all)
    println("\n", "="^90)
    @printf("WP4 ENTRY TIMING — %s (tournament %d)\n", tag, tid)
    println("="^90)

    ds   = pinned(tag == "ire718" ? "ire718_bfpillar" : tag)
    expr = newest_experiment("l2_$(tag)_sup40_sw40")
    c    = subset_corpus(corpus_all, tid)
    sys  = reference_system()

    fz = build_arm(c, expr, ds, :frozen)
    lv = build_arm(c, expr, ds, :live)

    led_fz = entry_ledger(fz, sys, expr)
    led_lv = entry_ledger(lv, sys, expr)

    drift  = reading_1_drift(fz)
    clv    = reading_2_clv(led_fz)
    ts_fz  = reading_3_wealth(led_fz)
    ts_lv  = reading_3_wealth(led_lv)
    oracle = reading_4_oracle(ts_fz)
    news   = reading_5_teamnews(ts_fz, ts_lv)

    println("\n[R1] PriceDrift by entry bucket — H1: expect ~0 everywhere")
    show(stdout, MIME"text/plain"(),
         select(drift, :entry_bucket, :n_quotes, :drift_mean, :drift_ci_lo, :drift_ci_hi,
                :drift_wait_paid))
    println("\n\n[R3] wealth by entry rule (frozen arm)")
    show(stdout, MIME"text/plain"(),
         select(ts_fz, :entry_name, :legs, :bets, :med_lead, :roi, :roi_ci_lo, :roi_ci_hi,
                :final, :hurdle_G_emp))
    println("\n\n[R4] oracle vs control — H3")
    show(stdout, MIME"text/plain"(), oracle)
    println("\n\n[R5] live - frozen (team news) — H4")
    show(stdout, MIME"text/plain"(), first(news, 8))
    println()
    pw = path_warning(ts_fz)
    isempty(pw) || println("\n", pw)

    mkpath(OUT_DIR)
    res = (tag = tag, drift = drift, clv = clv, ts_frozen = ts_fz, ts_live = ts_lv,
           oracle = oracle, teamnews = news,
           n_snap_frozen = length(fz), n_snap_live = length(lv))
    serialize(joinpath(OUT_DIR, "$(tag)_entry.jls"), res)
    return res
end

# ===================================================================
# 5. Run
# ===================================================================

println("\n", "="^90)
println("WP4 — entry timing on the Ireland order book")
println("="^90)

corpus_all = build_corpus("ireland", [79, 718]; from = Date(2026, 5, 20), to = Date(2026, 8, 10))
@printf("corpus: %d fixtures, %d slates\n",
        length(corpus_all.fixtures), length(corpus_slates(corpus_all)))

r79  = run_league("ire79",  79,  corpus_all)
r718 = run_league("ire718", 718, corpus_all)

println("\n", "="^90)
println("WP4 complete — results in $OUT_DIR")
println("Read the two leagues as independent replications, never pooled: their market pillars")
println("differ (79 SofaScore ~100%, 718 Betfair ~54%), so a pooled contrast is confounded.")
println("="^90)
