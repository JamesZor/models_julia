# current_development/orderbook_layer2/r06_fullbook.jl
#
# WP8. Is the model uninformative, or is the BOOK selected badly?
#
# ---------------------------------------------------------------------------------------------
# PRE-REGISTRATION
# ---------------------------------------------------------------------------------------------
#
# WP5's `w* = 0` was measured on 530 legs the staking layer chose. Those legs are selected on
# `p_model > p_market`, which is the same quantity the test evaluates — so the result cannot
# separate "the model knows nothing" from "the selection rule picks the model's errors".
#
#   F1  The full book carries MORE skill than the staked subset. Equivalently `w*` is higher on
#       all quoted selections than on the ones actually bet.
#       If F1 fails — if skill is the same or worse on the full book — the problem is Layer 1 and
#       no staking intervention can help.
#
#   F2  Skill declines monotonically in the claim `p_model − p_market`, and the staked legs are
#       concentrated in the declining end. This is the optimizer's curse stated as a measurement
#       rather than as an explanation.
#
#   F3  `w*` on the full book is still well below 1. The model does not beat the closing market
#       outright even unselected; the interesting question is only whether it is worth a nonzero
#       weight.
#
# Prior, stated before running: F2 holds (the C3 tail pattern was clean and replicated), F1 holds
# weakly, F3 holds. The outcome that would most change the plan is F1 failing.
#
# ---------------------------------------------------------------------------------------------
# WHAT EACH OUTCOME MEANS
# ---------------------------------------------------------------------------------------------
#
#   full-book w* ~ 0, staked w* ~ 0     Layer 1. The engine adds nothing anywhere. Stop staking
#                                       it and go fix the model.
#   full-book w* > 0, staked w* = 0     Layer 2, and fixable. The engine has information; the
#                                       book is built out of its errors. Shrink toward the market
#                                       or abstain above a claim threshold.
#   full-book w* ~ 1                    Would contradict WP5 outright and mean something is wrong
#                                       with one of the two measurements. Investigate before
#                                       believing it.
#
# ---------------------------------------------------------------------------------------------
# USAGE (server; rebuilds Tier 1, ~3 min per league on 16 threads)
# ---------------------------------------------------------------------------------------------
#
#   include("current_development/orderbook_layer2/r06_fullbook.jl")

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const PF = BayesianFootball.Portfolio
const MD = BayesianFootball.MatchDay
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments

include(joinpath(@__DIR__, "l00_corpus.jl"))
include(joinpath(@__DIR__, "l01_l2_experiment.jl"))
include(joinpath(@__DIR__, "l02_l2_ledger.jl"))
include(joinpath(@__DIR__, "l03_l2_metrics.jl"))
include(joinpath(@__DIR__, "..", "matchday_2026_08_08", "l02_slate_replay.jl"))
include(joinpath(@__DIR__, "l04_corpus_replay.jl"))
include(joinpath(@__DIR__, "l05_curation.jl"))
include(joinpath(@__DIR__, "l06_fullbook.jl"))

const ENGINE_DIR = "./data/l2_ireland_engines"
const LEDGER_DIR = "./data/l2_entry_timing"
const OUT_DIR    = "./data/l2_fullbook"

pinned(tag) = deserialize(joinpath(ENGINE_DIR, "ds_$(tag).jls"))

function newest_experiment(prefix::String)
    dirs = filter(d -> startswith(basename(d), prefix),
                  [joinpath(ENGINE_DIR, d) for d in readdir(ENGINE_DIR) if isdir(joinpath(ENGINE_DIR, d))])
    isempty(dirs) && error("r06: no experiment matching $prefix")
    return EE.load_experiment(sort(dirs, by = mtime, rev = true)[1])
end

"The same book and policy WP4 staked with — the staked subset has to come from the same spec."
reference_system() = PF.PortfolioSystem(
    PF.BookSpec(markets = DD.MarketConfig(DD.AbstractMarket[
                    DD.Market1X2(), DD.MarketBTTS(),
                    (DD.MarketOverUnder(l) for l in (0.5, 1.5, 2.5, 3.5, 4.5))...])),
    PF.PolicySpec(trust = PF.FlatTrust(0.25), risk = PF.SlateDrawdown(23.0),
                  cap = PF.FixedCap(0.25)))

banner(s) = (println("\n", "="^92); println(s); println("="^92))
shw(t, d; n = 20) = (println("\n", t);
                     isempty(d) ? println("  (empty)") :
                         show(stdout, MIME"text/plain"(), first(d, min(n, nrow(d)))); println())

# ===================================================================

function run_league(tag::String, tid::Int, corpus_all)
    banner("WP8 FULL BOOK — $tag (tournament $tid)")

    ds   = pinned(tag == "ire718" ? "ire718_bfpillar" : tag)
    expr = newest_experiment("l2_$(tag)_sup40_sw40")
    c    = subset_corpus(corpus_all, tid)
    sys  = reference_system()
    g    = recommend_grid(c; coverage = 0.80)

    snaps = build_snapshots(c, expr, ds; arm = :frozen, lookback = g.lookback,
                            fine_step = g.fine_step, coarse_step = g.coarse_step, verbose = true)

    full = full_book_close(snaps, expr, sys.book)
    isempty(full) && (println("  no book built"); return nothing)

    led  = deserialize(joinpath(LEDGER_DIR, "$(tag)_entry.jls")).ledger
    full = mark_staked!(full, led)

    n_staked_led = nrow(unique(filter(r -> r.entry_name == "AtClose", led),
                               [:match_id, :group, :line, :selection]))
    @printf("\nfull book: %d selections over %d matches;  staked flag set on %d;  ledger had %d\n",
            nrow(full), length(unique(full.match_id)), count(full.staked), n_staked_led)
    if count(full.staked) < n_staked_led
        # single literal — @printf will not take a concatenated format string
        @printf("  WARNING %d staked legs have NO counterpart in the full book; the two paths disagree about the book itself and the comparison below is not clean\n",
                n_staked_led - count(full.staked))
    end

    # ---- F1 ----
    cmp = DataFrame([book_skill(full, "ALL quoted"),
                     book_skill(filter(r -> r.staked, full), "STAKED only"),
                     book_skill(filter(r -> !r.staked, full), "NOT staked")])
    shw("[F1] skill vs the de-vigged close — the whole point", cmp)

    ws = DataFrame([merge((set = "ALL quoted",), w_star(full)),
                    merge((set = "STAKED only",), w_star(filter(r -> r.staked, full))),
                    merge((set = "NOT staked",), w_star(filter(r -> !r.staked, full)))])
    shw("[F1b] optimal weight on the model, w*", ws)

    # ---- F2 ----
    shw("[F2] the curse curve — skill by claimed disagreement, FULL book", curse_curve(full))
    shw("[F2b] same bands, STAKED subset only", curse_curve(filter(r -> r.staked, full)))

    # ---- supporting ----
    shw("[S1] full-book skill by market", combine(groupby(full, :group)) do sub
            merge((group = sub.group[1],), book_skill(sub, sub.group[1]))
        end)

    serialize(joinpath(OUT_DIR, "$(tag)_fullbook.jls"),
              (tag = tag, full = full, cmp = cmp, wstar = ws,
               curve_all = curse_curve(full),
               curve_staked = curse_curve(filter(r -> r.staked, full))))
    return full
end

# ===================================================================

banner("WP8 — full-book test: Layer 1 problem or selection problem?")
println("WP5 measured w* = 0 on legs the staking layer CHOSE. Those legs are selected on")
println("p_model > p_market, the same quantity under test. This scores every quoted selection.")

mkpath(OUT_DIR)
corpus_all = build_corpus("ireland", [79, 718]; from = Date(2026, 5, 20), to = Date(2026, 8, 10))

f79  = run_league("ire79",  79,  corpus_all)
f718 = run_league("ire718", 718, corpus_all)

# ===================================================================
# Pooled — one parameter, both leagues
# ===================================================================
if f79 !== nothing && f718 !== nothing
    banner("POOLED (both leagues)")
    pool = vcat(f79, f718; cols = :union)
    shw("skill", DataFrame([book_skill(pool, "ALL quoted"),
                            book_skill(filter(r -> r.staked, pool), "STAKED only"),
                            book_skill(filter(r -> !r.staked, pool), "NOT staked")]))
    shw("w*", DataFrame([merge((set = "ALL quoted",), w_star(pool)),
                         merge((set = "STAKED only",), w_star(filter(r -> r.staked, pool))),
                         merge((set = "NOT staked",), w_star(filter(r -> !r.staked, pool)))]))
    shw("curse curve", curse_curve(pool))
    serialize(joinpath(OUT_DIR, "pooled_fullbook.jls"), pool)

    wa, wsk = w_star(pool), w_star(filter(r -> r.staked, pool))
    println("\nVERDICT")
    if wa.w > wsk.w + 0.05
        @printf("  w* is %.2f on the full book against %.2f on the staked subset.\n", wa.w, wsk.w)
        println("  => LAYER 2. The engine carries information the SELECTION RULE throws away.")
        println("     Shrink toward the market, or abstain above a claim threshold.")
    elseif wa.w <= 0.05
        @printf("  w* is %.2f on the full book and %.2f on the staked subset.\n", wa.w, wsk.w)
        println("  => LAYER 1. The engine adds nothing anywhere in the book, selected or not.")
        println("     No staking intervention can help; the model itself is the constraint.")
    else
        @printf("  w* is %.2f full book, %.2f staked — no clear separation.\n", wa.w, wsk.w)
        println("  => inconclusive at this sample size; read the curse curve rather than w*.")
    end
end

banner("WP8 complete — results in $OUT_DIR")
