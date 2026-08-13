# current_development/orderbook_layer2/r09_route2_noanchor.jl
#
# WP10 (continued). Judge the `market_on = false` engine trained by
# `r08_train_ireland_noanchor.jl` against the same Route 2 sample and the same estimators r07
# used for `src_sup40_sw40` — plus the one comparison r08 couldn't make: anchored vs unanchored,
# head to head, on identical matches and identical outcomes.
#
# ---------------------------------------------------------------------------------------------
# TWO QUESTIONS, IN ORDER
# ---------------------------------------------------------------------------------------------
#
#   Q1  Head-to-head accuracy. Same 293 matches, same de-vigged close, same realised outcomes —
#       only the engine differs. `ll_model` from `w_star` (the LL at w=1, i.e. the pure-model
#       score) is the number to read: does the unanchored engine beat, match, or lose to the
#       anchored one against the closing price?
#
#         much worse   -> the anchoring is carrying the accuracy (RESULTS.md's expectation)
#         comparable   -> the anchoring is suppressing a signal the xG/goals pillars would
#                          otherwise express; the weights, not the model, would be the problem
#
#   Q2  Does it change what Layer 2 should do. Re-run the full WP9 suite (R1-R5, R2-applied,
#       R4 open/close) on the unanchored engine's books, unconditionally, per the standing
#       instruction not to gate the allocation study on the accuracy result.
#
# ---------------------------------------------------------------------------------------------
# WHY THE ANCHORED SIDE IS LOADED, NOT RECOMPUTED
# ---------------------------------------------------------------------------------------------
#
# `r07_route2.jl` already serialised `(ire79 = (frame, base), ire718 = (frame, base), pooled)` to
# `data/l2_route2/route2.jls`. Recomputing it here would risk a second, subtly different
# `route2_setup` call producing numbers that don't quite match RESULTS.md's — loading the exact
# artifact r07 produced is what makes Q1 a comparison against the published verdict rather than
# against a fresh, possibly-drifted run of the same code.
#
# ---------------------------------------------------------------------------------------------
# USAGE (server; run AFTER r08's background job reports both engines PASS their gates)
# ---------------------------------------------------------------------------------------------
#
#   include("current_development/orderbook_layer2/r09_route2_noanchor.jl")

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
include(joinpath(@__DIR__, "l07_route2.jl"))     # route2_setup, w_star, book_skill, curse_curve,
                                                  # reference_spec, reference_policy, MaxOdds/MaxClaim

const ENGINE_DIR   = "./data/l2_ireland_engines"
const ROUTE2_DIR   = "./data/l2_route2"          # r07's output — read only
const OUT_DIR      = "./data/l2_route2_noanchor"

pinned(tag) = deserialize(joinpath(ENGINE_DIR, "ds_$(tag).jls"))
function newest_experiment(prefix::String)
    dirs = filter(d -> startswith(basename(d), prefix),
                  [joinpath(ENGINE_DIR, d) for d in readdir(ENGINE_DIR) if isdir(joinpath(ENGINE_DIR, d))])
    isempty(dirs) && error("r09: no experiment matching $prefix")
    return EE.load_experiment(sort(dirs, by = mtime, rev = true)[1])
end

banner(s) = (println("\n", "="^92); println(s); println("="^92))
shw(t, d; n = 25) = (println("\n", t);
                     isempty(d) ? println("  (empty)") :
                         show(stdout, MIME"text/plain"(), first(d, min(n, nrow(d)))); println())

const WEALTH = [BT.CumulativeWealth(), BT.SharpeRatio(), BT.CalmarRatio(), BT.SortinoRatio()]

# ===================================================================
# 0. The anchored side, loaded from r07's own artifact
# ===================================================================

function anchored_route2()
    path = joinpath(ROUTE2_DIR, "route2.jls")
    isfile(path) || error("r09: $path missing — run r07_route2.jl first, Q1 needs its artifact")
    return deserialize(path)
end

# ===================================================================
# 1. Books for the unanchored engine (mirrors r07's run_league)
# ===================================================================

function run_league_noanchor(tag::String)
    banner("WP10 ROUTE 2 (noanchor) — $tag")
    ds   = pinned(tag == "ire718" ? "ire718_bfpillar" : tag)
    expr = newest_experiment("l2_$(tag)_noanchor")

    st    = route2_setup(ds, expr; price = :close)
    books = st.books
    frame = books_frame(books, st.ds1)
    @printf("\n%d books, %d scored selections, %d matches, %s .. %s\n",
            length(books), nrow(frame), length(unique(frame.match_id)),
            minimum(frame.date), maximum(frame.date))

    shw("[R1] skill and w* — noanchor engine",
        DataFrame([merge((set = "ALL selections",), w_star(frame))]))
    shw("[R1b] skill vs the de-vigged close", DataFrame([book_skill(frame, "ALL selections")]))
    shw("[R1c] by season", combine(groupby(frame, :season)) do sub
            merge((season = sub.season[1],), w_star(sub), (skill = book_skill(sub, "").skill,))
        end)

    shw("[R2] curse curve — noanchor engine", curse_curve(frame))
    shw("[R2b] skill by family", skill_table(frame, [:family]))

    base = run_policy(books, reference_policy(); label = "reference (noanchor)", metrics = WEALTH)
    @printf("\nbaseline: %d slates, %d bets, final %.3f, roi %.2f%% [%.2f, %.2f], growth %.5f, mdd %.1f%%\n",
            base.n_slates, base.n_bets, base.final, base.roi, base.roi_lo, base.roi_hi,
            base.growth, base.mdd)
    shw("attribution by family", PF.attribution(base.traj))

    return (tag = tag, ds = ds, expr = expr, st = st, books = books, frame = frame, base = base)
end

# ===================================================================
# 2. Q1 — anchored vs unanchored, head to head, on the SAME matches
# ===================================================================

"""
    head_to_head(anchored_frame, noanchor_frame, label) -> NamedTuple

The comparison this file exists for. Both frames come from the same match set (same corpus,
same de-vigged close, same outcomes) via `route2_setup(...; price = :close)`, so `ll_market`
from `w_star` should agree between them to within rounding — printed as a sanity check, not
just an assumption.
"""
function head_to_head(af::AbstractDataFrame, nf::AbstractDataFrame, label::AbstractString)
    wa, wn = w_star(af), w_star(nf)
    sa, sn = book_skill(af, "anchored"), book_skill(nf, "noanchor")
    same_market = isapprox(wa.ll_market, wn.ll_market; atol = 1e-3)
    return (league = label,
            n_anchored = wa.n, n_noanchor = wn.n,
            ll_market = wa.ll_market, ll_market_check = wn.ll_market, same_market = same_market,
            ll_model_anchored = wa.ll_model, ll_model_noanchor = wn.ll_model,
            gap = round(wn.ll_model - wa.ll_model, digits = 5),   # + => noanchor WORSE (higher LL)
            w_anchored = wa.w, w_noanchor = wn.w,
            skill_anchored = sa.skill, skill_noanchor = sn.skill)
end

# ===================================================================

banner("WP10 — market_on=false control, judged against Route 2")
mkpath(OUT_DIR)

anchored = anchored_route2()

N79  = run_league_noanchor("ire79")
N718 = run_league_noanchor("ire718")

banner("[Q1] HEAD TO HEAD — anchored (src_sup40_sw40) vs unanchored, same matches")
h79  = head_to_head(anchored.ire79.frame,  N79.frame,  "79")
h718 = head_to_head(anchored.ire718.frame, N718.frame, "718")
h_df = DataFrame([h79, h718])
show(stdout, MIME"text/plain"(), h_df); println()
println("\n`gap` > 0 means the unanchored engine has HIGHER (worse) log loss than the anchored one.")
println("`same_market` should be true in both rows — it is the check that Q1 is comparing the")
println("same selections, not two different corpora.")

pool_a = anchored.pooled
pool_n = vcat(N79.frame, N718.frame; cols = :union)
hp = head_to_head(pool_a, pool_n, "pooled")
shw("[Q1 pooled]", DataFrame([hp]))

# ===================================================================
# 3. Q2 — the full WP9 suite, unconditionally
# ===================================================================

function studies_noanchor(L)
    tag, books, frame = L.tag, L.books, L.frame

    banner("[R5] $tag (noanchor) — flat trust under a BINDING vs a SLACK risk model")
    binding = [("FlatTrust($w)", reference_policy(trust = PF.FlatTrust(w)))
               for w in (0.10, 0.25, 0.50, 1.00)]
    shw("binding: SlateDrawdown(23) + FixedCap(0.25)", race(books, binding; metrics = WEALTH))

    slack = [("FlatTrust($w)", reference_policy(trust = PF.FlatTrust(w),
                                                risk = PF.NoRisk(), cap = PF.FixedCap(0.05)))
             for w in (0.10, 0.25, 0.50, 1.00)]
    shw("slack: NoRisk + FixedCap(0.05)", race(books, slack; metrics = WEALTH))

    banner("[R3] $tag (noanchor) — per-market trust, derived on 2025 and applied to 2026")
    cut = Date(2026, 1, 1)
    b25, b26 = split_books(books, cut)
    f25 = filter(r -> r.date < cut, frame)
    @printf("  derive on %d books (<%s), test on %d books\n", length(b25), cut, length(b26))
    if length(b25) >= 30 && length(b26) >= 30
        tr = trust_from_frame(f25; min_legs = 20)
        arms = [("uncurated (flat 0.25)", reference_policy()),
                ("CURATED (2025-derived)", reference_policy(trust = tr))]
        shw("held-out 2026", race(b26, arms; metrics = WEALTH))
    else
        println("  not enough books either side of $cut")
    end

    banner("[R2-applied] $tag (noanchor) — filters implied by the curse curve")
    arms = [("no filter",           reference_policy()),
            ("MaxOdds(6.0)",        reference_policy(filter = MaxOdds(6.0))),
            ("MaxClaim(0.05)",      reference_policy(filter = MaxClaim(0.05))),
            ("MaxClaim(0.02)",      reference_policy(filter = MaxClaim(0.02))),
            ("MinEdge(0.02)",       reference_policy(filter = PF.MinEdge(0.02)))]
    shw("filters", race(books, arms; metrics = WEALTH))
    return nothing
end

function open_vs_close_noanchor(L)
    banner("[R4] $(L.tag) (noanchor) — entering at the OPEN vs the CLOSE")
    st_open = route2_setup(L.ds, L.expr; price = :open)
    f_open  = books_frame(st_open.books, st_open.ds1)
    shw("skill and w* at each instant",
        DataFrame([merge((instant = "open",),  w_star(f_open)),
                   merge((instant = "close",), w_star(L.frame))]))
    a = race(st_open.books, [("open",  reference_policy())]; metrics = WEALTH)
    b = race(L.books,       [("close", reference_policy())]; metrics = WEALTH)
    shw("allocation at each instant", vcat(a, b))
    return f_open
end

studies_noanchor(N79)
studies_noanchor(N718)
open_vs_close_noanchor(N79)
open_vs_close_noanchor(N718)

serialize(joinpath(OUT_DIR, "route2_noanchor.jls"),
          (ire79 = (frame = N79.frame, base = N79.base.report),
           ire718 = (frame = N718.frame, base = N718.base.report),
           head_to_head = h_df))

banner("WP10 complete — results in $OUT_DIR")
println("Q1 (accuracy) answered by the head-to-head table above.")
println("Q2 (allocation) answered by the R1-R5/R2-applied/R4 tables, unconditionally, per instruction.")
