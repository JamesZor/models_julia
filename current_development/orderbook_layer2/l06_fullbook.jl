# current_development/orderbook_layer2/l06_fullbook.jl
#
# WP8. Score the model on EVERY quoted selection, not only the ones it wanted to bet.
#
# ---------------------------------------------------------------------------------------------
# THE QUESTION
# ---------------------------------------------------------------------------------------------
#
# WP5 measured `w* = 0`: on 530 legs the de-vigged Betfair close weakly dominates the model, and
# no two-parameter recalibration recovers it even fitted in-sample. But every one of those legs
# was a leg the staking layer CHOSE — a positive-edge selection. That is an adversarially selected
# subsample, and it is selected on exactly the quantity being tested.
#
# Two very different worlds produce that same measurement:
#
#   A. The model is uninformative everywhere.
#      -> Layer 1 problem. No staking rule, trust weight or filter can help.
#
#   B. The model is competitive across the book and bad only where it disagrees MOST with the
#      market — the optimizer's curse. Selecting on `p_model - p_market > 0` preferentially picks
#      legs whose model probability is high because of ERROR rather than because of information,
#      and the bigger the claimed edge the worse the ratio.
#      -> Layer 2 problem, and a fixable one: shrink toward the market, or abstain in the tails.
#
# Everything measured so far is consistent with both. The C3 tail result (beats the market on
# 53.8%/59.5% of legs claiming <2pp, and 37.3%/21.3% of legs claiming >+5pp) is the signature of
# B, but it was itself computed on selected legs, so it cannot distinguish them either.
#
# The full book can. If skill over ALL quoted selections is ~0 or positive while skill over the
# STAKED subset is negative, that is world B and the difference is the size of the curse.
#
# ---------------------------------------------------------------------------------------------
# WHY THIS IS CHEAP
# ---------------------------------------------------------------------------------------------
#
# WP3 measured serving latents to be bit-identical across `as_of` — a pure function of
# `(fixture, split)`. So the posterior score matrix, which is the expensive step, needs computing
# **once per match** rather than once per instant. Everything else is `extract_selections` on a
# quote frame.
#
# `build_book` is deliberately NOT used: it also runs `allocate` and `shrink_factor`, which are
# the costly parts and which exist to decide stakes. This file never stakes anything, so it takes
# the same two `Predictions` calls `build_book` takes and stops before the allocator.
#
# ---------------------------------------------------------------------------------------------
# WHAT IS REUSED, AND WHY THAT MATTERS
# ---------------------------------------------------------------------------------------------
#
#   Predictions.compute_score_matrix / compute_market_probs   the model's probabilities
#   Portfolio.extract_selections                              quoting + vig removal
#   Data.grade_selection                                      the outcome of ANY selection
#
# None of the probability or de-vigging maths is reimplemented here. That is the point: if this
# file computed `p_model` its own way, a discrepancy with the staked ledger would be
# indistinguishable from the effect under test.
#
# `extract_selections` also enforces `require_complete_markets`, which matters more here than
# anywhere else: vig removal divides by the sum over whatever legs are present, so a group
# missing a leg manufactures edge on the survivors — up to 20% on a 1X2 missing one way. On the
# full book that would be a systematic bias toward exactly the finding we are looking for.

using DataFrames, Dates, Statistics, Printf

# ===================================================================
# 1. The full book at each fixture's own close
# ===================================================================

"""
    full_book_close(snaps, expr, spec) -> DataFrame

Every quoted selection of every fixture, priced at THAT fixture's last pre-kickoff snapshot.

One row per `(match_id, group, line, selection)` with:

  * `p_model`   posterior-mean model probability
  * `p_market`  de-vigged closing probability — the same `fair_close` WP5 scored against
  * `odds`      the settlement price after the `AbstractPricePolicy`
  * `is_winner` realised outcome, from the corpus's FROZEN results

"That fixture's own close", not the slate's: slates here are staggered by up to four hours, so a
single instant would price the late fixtures two hours early. This mirrors how `AtClose` picks
per leg, which is what makes the staked subset a strict subset of these rows rather than a
differently-timed one.
"""
function full_book_close(snaps::L2Snapshots, expr, spec)
    PF, PR, DD = _pf(), BayesianFootball.Predictions, _dd()

    # last snapshot per match, and the latents to go with it
    best = Dict{Int,Any}()
    for s in snaps.snaps, r in eachrow(s.latents)
        mid = Int(r.match_id)
        if !haskey(best, mid) || s.as_of > best[mid].as_of
            best[mid] = (as_of = s.as_of, odds = s.odds, lat = r, day = s.slate_day)
        end
    end

    rows = NamedTuple[]
    for (mid, b) in best
        sc = get(snaps.results, mid, nothing)
        sc === nothing && continue

        # ONE score matrix per match — licensed by WP3's latent-invariance measurement.
        sm = try
            PR.compute_score_matrix(expr.config.model, PR.extract_params(expr.config.model, b.lat))
        catch
            continue
        end
        probs = Dict(string(m) => PR.compute_market_probs(sm, m) for m in spec.markets.markets)

        for sel in PF.extract_selections(b.odds, mid, spec, probs)
            push!(rows, (match_id   = mid,
                         slate_day  = b.day,
                         as_of      = b.as_of,
                         kickoff    = kickoff_of(snaps, mid),
                         family     = sel.family,
                         group      = sel.group,
                         line       = sel.line,
                         selection  = sel.selection,
                         odds       = sel.odds_used,
                         odds_quoted = sel.odds_quoted,
                         p_model    = sel.p_model,
                         p_market   = sel.p_market,
                         is_winner  = DD.grade_selection(sel.group, sel.line, sel.selection,
                                                         sc[1], sc[2])))
        end
    end

    isempty(rows) && return DataFrame()
    df = DataFrame(rows)
    df.mins_to_ko = [Dates.value(k - a) / 60_000 for (k, a) in zip(df.kickoff, df.as_of)]
    df.claim      = df.p_model .- df.p_market
    df.fair_close = df.p_market          # name it as WP5 does, so the same cuts apply unchanged
    return df
end

# ===================================================================
# 2. Marking which of them the staking layer actually took
# ===================================================================

"""
    mark_staked!(full, ledger) -> DataFrame

Flag the rows the staking layer chose, by joining the WP4 `AtClose` ledger on leg identity.

Joined rather than re-derived. The staking decision is a Kelly solve over the whole score grid
with a cap, a drawdown factor and a filter — "positive edge" is a necessary condition for it, not
a sufficient one, and reconstructing it here would compare the full book against a *guess* at
what was staked. The join compares it against what actually was.

`n_unmatched` is reported by the caller: a staked leg with no counterpart in the full book would
mean the two paths disagree about the book itself, which would invalidate the comparison.
"""
function mark_staked!(full::DataFrame, ledger::AbstractDataFrame)
    key = [:match_id, :group, :line, :selection]
    st  = unique(select(filter(r -> r.entry_name == "AtClose", ledger), key))
    st.staked = trues(nrow(st))
    out = leftjoin(full, st, on = key)
    out.staked = coalesce.(out.staked, false)
    return out
end

# ===================================================================
# 3. The comparison
# ===================================================================

"""
    book_skill(df, label) -> NamedTuple

Skill of `p_model` against `p_market`, with the same clustered interval WP5 used.
"""
function book_skill(df::AbstractDataFrame, label::AbstractString)
    isempty(df) && return (set = label, n = 0, skill = NaN, lo = NaN, hi = NaN, beats = NaN)
    s = leg_skill(df)
    ci = _cluster_boot(s, df.match_id)
    return (set = label, n = nrow(df),
            skill = round(mean(s), digits = 5),
            lo = round(ci.lo, digits = 5), hi = round(ci.hi, digits = 5),
            beats = round(100 * count(>(0), s) / length(s), digits = 1))
end

"""
    w_star(df) -> NamedTuple

The one-parameter blend fit: the weight on the model in `w·p_model + (1−w)·p_market` that
minimises log loss.

This is WP5's settling statistic, recomputed on whichever subset it is handed. Reported on the
full book and on the staked subset, the pair answers the question this file exists for: a `w*`
that is clearly positive on the full book and zero on the staked subset IS the optimizer's curse,
measured.
"""
function w_star(df::AbstractDataFrame)
    isempty(df) && return (n = 0, w = NaN, ll_at_w = NaN, ll_market = NaN, ll_model = NaN)
    p = clamp.(Float64.(df.p_model), _EPS, 1 - _EPS)
    q = clamp.(Float64.(df.p_market), _EPS, 1 - _EPS)
    y = Float64.(coalesce.(df.is_winner, false))
    _ll(v) = (v = clamp.(v, _EPS, 1 - _EPS); -mean(@. y * log(v) + (1 - y) * log1p(-v)))

    ws = 0.0:0.02:1.0
    ls = [_ll(w .* p .+ (1 - w) .* q) for w in ws]
    i  = argmin(ls)
    return (n = nrow(df), w = ws[i], ll_at_w = round(ls[i], digits = 5),
            ll_market = round(ls[1], digits = 5), ll_model = round(ls[end], digits = 5),
            gain = round(ls[1] - ls[i], digits = 5))
end

"""
    curse_curve(df) -> DataFrame

Skill as a function of how far the model's probability sits above the market's.

The direct measurement of the curse. Under world A this is flat and negative — the model is
uninformative wherever you look. Under world B it declines with the claim: fine where the model
agrees with the market, bad where it disagrees upward, and the staked book is drawn from the bad
end because that is what "positive edge" selects.

Bands are the WP5 ones, so the full-book and staked-book versions line up row for row.
"""
function curse_curve(df::AbstractDataFrame)
    d = copy(df)
    d.edge_band = edge_band.(d.claim)
    out = combine(groupby(d, :edge_band)) do sub
        s = leg_skill(sub)
        ci = _cluster_boot(s, sub.match_id)
        (n = nrow(sub),
         staked = hasproperty(sub, :staked) ? count(sub.staked) : 0,
         skill = round(mean(s), digits = 5),
         lo = round(ci.lo, digits = 5), hi = round(ci.hi, digits = 5),
         beats = round(100 * count(>(0), s) / length(s), digits = 1),
         med_odds = round(median(sub.odds), digits = 2))
    end
    return sort!(out, :edge_band)
end
