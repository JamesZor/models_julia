# current_development/orderbook_layer2/r07_route2.jl
#
# WP9. The Layer-2 studies, on the 293-match sample where the model has a measured edge.
#
# ---------------------------------------------------------------------------------------------
# WHY THE EARLIER ANSWERS DO NOT COUNT
# ---------------------------------------------------------------------------------------------
#
# WP4-WP8 ran on the 81-match order-book corpus. Measured with r21's own metric and benchmark on
# the experiment this stream trained, that corpus is the only slice of the OOS set where the
# model loses to the market (+0.0054 on 21 overlapping matches, against -0.0148 on 2025 and
# -0.0301 on the rest of 2026). It is a sharp market on predictable fixtures.
#
# Asking "which markets should we trust" or "what is the optimal model weight" on a sample where
# the model has no edge answers neither question. This file re-asks them where there is an edge.
#
# ---------------------------------------------------------------------------------------------
# PRE-REGISTRATION
# ---------------------------------------------------------------------------------------------
#
#   R1  `w*` is clearly positive on Route 2 — the WP8 zero was a property of that window, not of
#       the engine. Expected in the 0.3-0.7 range given the log-loss gap of ~0.018 nats.
#
#   R2  The curse curve still bends: skill highest near agreement, falling in both tails. The
#       WP8 shape was measured on a no-edge sample, so its SHAPE may survive even though its
#       LEVEL was meaningless. If the shape holds here it is a real property of the engine.
#
#   R3  Per-market curation, derived on 2025 and applied to 2026, beats no curation on growth.
#       This is the WP5 question re-asked where it is answerable. WP5's r = -0.647 was measured
#       across two leagues in a no-edge window and should not be treated as evidence either way.
#
#   R4  Entering at the CLOSE beats entering at the OPEN. WP4 measured this on the order book and
#       the mechanism (overround compression) is a market fact independent of model quality, so
#       it should reproduce on the coarse two-point contrast.
#
#   R5  Flat trust is a NO-OP under a binding `SlateDrawdown` and does something under a slack
#       risk model. This is the homogeneity property of `risk_factor`, and Route 2 is the first
#       sample in this stream where `simulate` re-solves the drawdown budget properly rather than
#       being approximated by a rescale.
#
# Falsifier for the whole file: if `w*` is ~0 here too, then WP8's verdict was right after all
# and the order-book window was not the problem.
#
# ---------------------------------------------------------------------------------------------
# WHAT THIS PATH CANNOT ANSWER
# ---------------------------------------------------------------------------------------------
#
# No ladder, so no fill or size question — the order-book stream keeps that. Two prices per
# selection, so the entry-time result is a two-point check, not a curve. And these are summary
# closes, not the best back price a stake would have hit, so ROI here is an ALLOCATION number and
# is optimistic as an execution number.
#
# ---------------------------------------------------------------------------------------------
# USAGE (server)
# ---------------------------------------------------------------------------------------------
#
#   include("current_development/orderbook_layer2/r07_route2.jl")

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const PF = BayesianFootball.Portfolio
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments
const BT = BayesianFootball.BackTesting

include(joinpath(@__DIR__, "l01_l2_experiment.jl"))
include(joinpath(@__DIR__, "l02_l2_ledger.jl"))
include(joinpath(@__DIR__, "l03_l2_metrics.jl"))
include(joinpath(@__DIR__, "l05_curation.jl"))
include(joinpath(@__DIR__, "l06_fullbook.jl"))
include(joinpath(@__DIR__, "l07_route2.jl"))

const ENGINE_DIR = "./data/l2_ireland_engines"
const OUT_DIR    = "./data/l2_route2"

pinned(tag) = deserialize(joinpath(ENGINE_DIR, "ds_$(tag).jls"))
function newest_experiment(prefix::String)
    dirs = filter(d -> startswith(basename(d), prefix),
                  [joinpath(ENGINE_DIR, d) for d in readdir(ENGINE_DIR) if isdir(joinpath(ENGINE_DIR, d))])
    isempty(dirs) && error("r07: no experiment matching $prefix")
    return EE.load_experiment(sort(dirs, by = mtime, rev = true)[1])
end

banner(s) = (println("\n", "="^92); println(s); println("="^92))
shw(t, d; n = 25) = (println("\n", t);
                     isempty(d) ? println("  (empty)") :
                         show(stdout, MIME"text/plain"(), first(d, min(n, nrow(d)))); println())

const WEALTH = [BT.CumulativeWealth(), BT.SharpeRatio(), BT.CalmarRatio(), BT.SortinoRatio()]

# ===================================================================

function run_league(tag::String)
    banner("WP9 ROUTE 2 — $tag")
    ds   = pinned(tag == "ire718" ? "ire718_bfpillar" : tag)
    expr = newest_experiment("l2_$(tag)_sup40_sw40")

    st    = route2_setup(ds, expr; price = :close)
    books = st.books
    frame = books_frame(books, st.ds1)
    @printf("\n%d books, %d scored selections, %d matches, %s .. %s\n",
            length(books), nrow(frame), length(unique(frame.match_id)),
            minimum(frame.date), maximum(frame.date))

    # ---------- R1: is there an edge to allocate? ----------
    shw("[R1] skill and w* — is there anything here at all",
        DataFrame([merge((set = "ALL selections",), w_star(frame))]))
    shw("[R1b] skill vs the de-vigged close", DataFrame([book_skill(frame, "ALL selections")]))
    shw("[R1c] by season", combine(groupby(frame, :season)) do sub
            merge((season = sub.season[1],), w_star(sub), (skill = book_skill(sub, "").skill,))
        end)

    # ---------- R2: does the curse curve shape survive? ----------
    shw("[R2] curse curve on a sample WITH edge", curse_curve(frame))
    shw("[R2b] skill by family", skill_table(frame, [:family]))
    shw("[R2c] skill by odds band", sort!(skill_table(frame, [:odds_band]; min_legs = 20),
                                          :odds_band))

    # ---------- baseline allocation ----------
    base = run_policy(books, reference_policy(); label = "reference", metrics = WEALTH)
    @printf("\nbaseline: %d slates, %d bets, final %.3f, roi %.2f%% [%.2f, %.2f], growth %.5f, mdd %.1f%%\n",
            base.n_slates, base.n_bets, base.final, base.roi, base.roi_lo, base.roi_hi,
            base.growth, base.mdd)
    shw("attribution by family", PF.attribution(base.traj))
    base.n_slates >= MIN_SLATES_FOR_PATH ||
        @printf("  NOTE only %d slates — path metrics (Calmar/Sortino) below the %d-slate bar\n",
                base.n_slates, MIN_SLATES_FOR_PATH)

    return (tag = tag, ds = ds, expr = expr, st = st, books = books,
            frame = frame, base = base)
end

# ===================================================================
# Studies that need the books
# ===================================================================

function studies(L)
    tag, books, frame = L.tag, L.books, L.frame

    # ---------- R5: homogeneity, on real re-simulated policies ----------
    banner("[R5] $tag — flat trust under a BINDING vs a SLACK risk model")
    binding = [("FlatTrust($w)", reference_policy(trust = PF.FlatTrust(w)))
               for w in (0.10, 0.25, 0.50, 1.00)]
    shw("binding: SlateDrawdown(23) + FixedCap(0.25)", race(books, binding; metrics = WEALTH))

    slack = [("FlatTrust($w)", reference_policy(trust = PF.FlatTrust(w),
                                                risk = PF.NoRisk(), cap = PF.FixedCap(0.05)))
             for w in (0.10, 0.25, 0.50, 1.00)]
    shw("slack: NoRisk + FixedCap(0.05)", race(books, slack; metrics = WEALTH))
    println("  R5 expects the first table FLAT (homogeneity) and the second MONOTONE.")

    # ---------- R3: curation derived on 2025, tested on 2026 ----------
    banner("[R3] $tag — per-market trust, derived on 2025 and applied to 2026")
    cut = Date(2026, 1, 1)
    b25, b26 = split_books(books, cut)
    f25 = filter(r -> r.date < cut, frame)
    @printf("  derive on %d books (<%s), test on %d books\n", length(b25), cut, length(b26))

    if length(b25) >= 30 && length(b26) >= 30
        tr = trust_from_frame(f25; min_legs = 20)
        println("  derived $(length(tr.table)) keyed selections, default $(tr.default)")
        for (k, v) in sort([(k, v) for (k, v) in tr.table], by = x -> -x[2])
            @printf("     %-12s line %5.1f  %-10s  w = %.2f\n", k[1], k[2], String(k[3]), v)
        end
        arms = [("uncurated (flat 0.25)", reference_policy()),
                ("CURATED (2025-derived)", reference_policy(trust = tr))]
        shw("held-out 2026", race(b26, arms; metrics = WEALTH))
        shw("in-sample 2025 (for contrast only)", race(b25, arms; metrics = WEALTH))
    else
        println("  not enough books either side of $cut")
    end

    # ---------- R2 applied: claim and odds filters ----------
    banner("[R2-applied] $tag — filters implied by the curse curve")
    arms = [("no filter",           reference_policy()),
            ("MaxOdds(6.0)",        reference_policy(filter = MaxOdds(6.0))),
            ("MaxClaim(0.05)",      reference_policy(filter = MaxClaim(0.05))),
            ("MaxClaim(0.02)",      reference_policy(filter = MaxClaim(0.02))),
            ("both (claim .05)",    reference_policy(
                                        filter = PF.FilterChain(MaxOdds(6.0), MaxClaim(0.05)))),
            ("MinEdge(0.02) [opposite]", reference_policy(filter = PF.MinEdge(0.02)))]
    shw("filters — MinEdge is the OPPOSITE intervention, kept as a direction check",
        race(books, arms; metrics = WEALTH))
    return nothing
end

# ===================================================================
# R4: open vs close
# ===================================================================

function open_vs_close(L)
    banner("[R4] $(L.tag) — entering at the OPEN vs the CLOSE")
    st_open = route2_setup(L.ds, L.expr; price = :open)
    f_open  = books_frame(st_open.books, st_open.ds1)

    shw("skill and w* at each instant",
        DataFrame([merge((instant = "open",),  w_star(f_open)),
                   merge((instant = "close",), w_star(L.frame))]))

    # The allocation half. `race` already returns exactly the summary columns wanted, so the two
    # instants are run through it rather than hand-unpacking two NamedTuples into a frame.
    a = race(st_open.books, [("open",  reference_policy())]; metrics = WEALTH)
    b = race(L.books,       [("close", reference_policy())]; metrics = WEALTH)
    shw("allocation at each instant", vcat(a, b))
    return f_open
end

# ===================================================================

banner("WP9 — Layer-2 on the full OOS sample (Route 2)")
println("WP4-WP8 ran on the 81-match order-book corpus, the one slice of the OOS set where the")
println("model LOSES to the market. This re-asks the same questions where it wins.")
mkpath(OUT_DIR)

L79  = run_league("ire79")
L718 = run_league("ire718")

studies(L79);  studies(L718)
o79  = open_vs_close(L79)
o718 = open_vs_close(L718)

# ---------- pooled ----------
banner("POOLED")
pool = vcat(L79.frame, L718.frame; cols = :union)
shw("skill and w*", DataFrame([merge((set = "pooled",), w_star(pool))]))
shw("curse curve", curse_curve(pool))

serialize(joinpath(OUT_DIR, "route2.jls"),
          (ire79 = (frame = L79.frame, base = L79.base.report),
           ire718 = (frame = L718.frame, base = L718.base.report),
           pooled = pool))

banner("WP9 complete — results in $OUT_DIR")
println("Route 2 is the ALLOCATION answer. Execution — depth, fill, the entry-time curve —")
println("stays with the order-book stream, which remains the authority on what a stake costs.")
