# current_development/orderbook_layer2/l05_curation.jl
#
# WP5. Which markets does this model actually know something about, and which matches should it
# not be betting at all?
#
# ---------------------------------------------------------------------------------------------
# WHY NOT CLV, AND WHY NOT ROI
# ---------------------------------------------------------------------------------------------
#
# WP4 established the entry time (the close) and, in doing so, disqualified both of the estimators
# that stream leaned on:
#
#   * **CLV is degenerate at the close.** `clv = log(odds_entry * fair_close)`, and at the close
#     `odds_entry == odds_close`, so CLV collapses to `log(odds_close * fair_close)` — exactly
#     minus the market's margin on that selection. It carries no information about the model at
#     all. WP4's per-family CLV table is a ranking of SPREAD, not of edge, and reading it as a
#     curation signal would curate toward whichever markets happen to be tightest.
#
#   * **ROI cannot resolve anything at this sample size.** 267 legs per league, match-clustered
#     intervals spanning roughly ±40 percentage points. WP4 watched an ROI ordering reverse the
#     true answer on the oracle test.
#
# So curation needs a third estimator, and the honest one is the direct question:
#
#       does `p_model` beat `fair_close` at predicting what actually happened?
#
# ---------------------------------------------------------------------------------------------
# THE DISCRIMINATOR
# ---------------------------------------------------------------------------------------------
#
#   skill = logscore(p_model) - logscore(fair_close)         per leg, in nats
#
# `fair_close` is the market's own de-vigged closing probability — the strongest benchmark
# available, and the one the model has to beat for a bet to be more than a transfer of margin.
# Positive skill means the model knows something the closing market did not. It is measured in
# nats per leg, which puts it on the same scale as log-growth, so "is this family worth trusting"
# and "is this family worth staking" are answered in the same units.
#
# This is deliberately NOT a proper-scoring-rule argument about calibration. The standing
# instruction is to judge on growth, and growth is reported alongside. But at 267 legs growth
# cannot separate fifteen families, and skill can: it uses every leg's outcome against a fixed
# benchmark rather than through a stake that the risk model has already flattened.
#
# ⚠️ Scored on the SELECTION, not on the side taken. `p_model`, `fair_close` and `is_winner` all
# refer to the selection itself, so a laid leg contributes the same information as a backed one.
# That is what curation wants to know — whether the model reads this market — and it keeps the
# Double Chance scoring defect out of the picture entirely (DC is not in the book here).
#
# ---------------------------------------------------------------------------------------------
# VALIDATION: CROSS-LEAGUE, NOT SPLIT-HALF
# ---------------------------------------------------------------------------------------------
#
# Picking families by their own skill and then reporting that skill is circular — it is
# `r08_vector_alpha_optim.jl`'s mistake (15 fitted alphas, 0.089x out of sample) at smaller scale.
#
# The corpus offers a better answer than a temporal split. **Tournaments 79 and 718 are disjoint
# in fixtures, teams and season phase, and were fitted with different market pillars** (79 on
# SofaScore, 718 on early Betfair). So: derive the curation on one league, apply it to the other,
# and report the held-out number. Then swap. Two genuine out-of-sample tests, neither of which
# costs a single leg of sample the way halving does.
#
# If a family only looks good in the league it was chosen in, that is the answer.

using DataFrames, Dates, Statistics, Printf

const _EPS = 1e-6

# ===================================================================
# 1. The per-leg score
# ===================================================================

"""
    leg_skill(df) -> Vector{Float64}

Per-leg log-score improvement of `p_model` over `fair_close`, in nats.

    skill_i = [y log p + (1-y) log(1-p)] - [y log q + (1-y) log(1-q)]

with `p = p_model`, `q = fair_close`, `y = is_winner`. Positive means the model was closer to what
happened than the de-vigged closing market was.

Probabilities are clamped away from 0 and 1: a model probability of exactly 0 on an outcome that
occurred contributes -Inf and would silently take the whole family's mean with it.
"""
function leg_skill(df::AbstractDataFrame)
    p = clamp.(Float64.(df.p_model), _EPS, 1 - _EPS)
    q = clamp.(Float64.(df.fair_close), _EPS, 1 - _EPS)
    y = Float64.(coalesce.(df.is_winner, false))
    return @. (y * log(p) + (1 - y) * log1p(-p)) - (y * log(q) + (1 - y) * log1p(-q))
end

"""
    usable(df) -> SubDataFrame

Legs a skill score can be computed on: graded, with a finite de-vigged close.

`fair_close` is NaN wherever the closing group was incomplete, and an ungraded leg has no `y`.
Dropping them here rather than inside each cut keeps every table on the same denominator.
"""
usable(df::AbstractDataFrame) =
    filter(r -> !ismissing(r.is_winner) && isfinite(r.fair_close) &&
                isfinite(r.p_model) && 0 < r.fair_close < 1, df)

# ===================================================================
# 2. Cuts
# ===================================================================

"""
    skill_table(df, by; min_legs = 8) -> DataFrame

Skill, growth and a match-clustered interval, grouped by `by`.

`min_legs` exists because the family split is uneven by an order of magnitude — 35 legs on
O/U 4.5 over versus 2 on O/U 4.5 under — and a two-leg family will produce whichever answer its
two legs happened to give. Small groups are kept in the table with `enough = false` rather than
dropped, so the reader can see what was set aside.
"""
function skill_table(df::AbstractDataFrame, by; min_legs::Int = 8)
    u = usable(df)
    isempty(u) && return DataFrame()
    u = copy(u)
    u.skill = leg_skill(u)

    out = combine(groupby(u, by)) do sub
        ci = _cluster_boot(sub.skill, sub.match_id)
        stk = sum(sub.stake)
        (legs      = nrow(sub),
         matches   = length(unique(sub.match_id)),
         skill     = round(mean(sub.skill), digits = 5),
         skill_lo  = round(ci.lo, digits = 5),
         skill_hi  = round(ci.hi, digits = 5),
         beats_mkt = round(100 * count(>(0), sub.skill) / nrow(sub), digits = 1),
         med_odds  = round(median(sub.odds), digits = 2),
         roi       = stk > 0 ? round(100 * sum(sub.pnl) / stk, digits = 2) : 0.0,
         enough    = nrow(sub) >= min_legs)
    end
    return sort!(out, :skill, rev = true)
end

"""
    ODDS_BANDS

Bands for the favourite–longshot test, cut on the taken price.

Chosen on the market's own structure rather than on quantiles of this sample: 1.5 is roughly the
heavy-favourite boundary, 2.2 the coin-flip region, 3.5 the point at which 1X2 draws and O/U tails
start to dominate, and 6.0 the threshold the standing prior names ("skipping odds > ~6 helps").
Quantile bands would move with the sample and could not be compared across leagues.
"""
const ODDS_BANDS = [(1.0, 1.5), (1.5, 2.2), (2.2, 3.5), (3.5, 6.0), (6.0, Inf)]
const ODDS_BAND_NAMES = ["1_1.00-1.50", "2_1.50-2.20", "3_2.20-3.50", "4_3.50-6.00", "5_6.00+"]

odds_band(o::Real) = ODDS_BAND_NAMES[something(findfirst(b -> b[1] <= o < b[2], ODDS_BANDS),
                                               length(ODDS_BANDS))]

"""
    EDGE_BANDS

Bands on the model's claimed disagreement with the de-vigged close, `p_model - fair_close`.

This is the cut that tests the house philosophy directly. The standing position is that the
per-line expectation should sit on the market and the edge lives in per-match deviations — so
the tails are where the money is supposed to be. The alternative reading is that a large
disagreement is a large model error. Bucketing realised skill by claimed disagreement separates
the two, and it is cheap: it needs no extra data, only the column the staking layer already uses.

Signed, not absolute: over-confidence upward and downward are different failures.
"""
const EDGE_BANDS = [(-Inf, -0.05), (-0.05, -0.02), (-0.02, 0.02), (0.02, 0.05), (0.05, Inf)]
const EDGE_BAND_NAMES = ["1_below -5pp", "2_-5..-2pp", "3_within 2pp", "4_+2..+5pp", "5_above +5pp"]

edge_band(d::Real) = EDGE_BAND_NAMES[something(findfirst(b -> b[1] <= d < b[2], EDGE_BANDS),
                                               length(EDGE_BANDS))]

"""
    annotate!(df) -> df

Add the cut columns every table below groups on.

`claim` is measured against `fair_close`, NOT against `p_market`. The staking layer's own `edge`
column is computed against the vigged quote, so a leg can show a positive `edge` purely because
the market's margin sits on the other side of it. Curation must not reward that.
"""
function annotate!(df::AbstractDataFrame)
    df.claim      = Float64.(df.p_model) .- Float64.(df.fair_close)
    df.odds_band  = odds_band.(Float64.(df.odds))
    df.edge_band  = edge_band.(df.claim)
    df.market     = String.(df.group)
    return df
end

# ===================================================================
# 3. Match-level avoidance
# ===================================================================

"""
    match_table(df) -> DataFrame

One row per match: how wide its book was, how boldly the model disagreed, and what that was worth.

The question this serves is "which matches should we not have bet", so the candidate predictors
are all things knowable BEFORE the result:

  * `spread`    — mean relative spread across the match's legs. A wide book is a market that does
                  not know either, and it is where the margin is largest.
  * `max_claim` — the model's boldest disagreement with the de-vigged close on that match.
  * `n_legs`    — how much of the book the model wanted. A match where it likes everything is a
                  match where its overall level, not its relative reads, is driving the sheet.
  * `longshot`  — share of the match's legs priced above 6.0.
"""
function match_table(df::AbstractDataFrame)
    u = copy(usable(df))
    isempty(u) && return DataFrame()
    u.skill = leg_skill(u)
    out = combine(groupby(u, :match_id)) do sub
        stk = sum(sub.stake)
        (legs      = nrow(sub),
         spread    = round(mean(filter(isfinite, sub.rel_spread); init = NaN), digits = 4),
         max_claim = round(maximum(abs.(sub.claim)), digits = 4),
         longshot  = round(100 * count(>(6.0), sub.odds) / nrow(sub), digits = 1),
         skill     = round(mean(sub.skill), digits = 5),
         roi       = stk > 0 ? round(100 * sum(sub.pnl) / stk, digits = 2) : 0.0,
         pnl       = round(sum(sub.pnl), digits = 5))
    end
    return sort!(out, :skill)
end

"""
    tercile_cut(matches, col) -> DataFrame

Split matches into terciles on one pre-match observable and report what each tercile was worth.

Terciles rather than a regression because there are ~38 matches per league: a slope fitted on 38
points with this much noise is a number, not a finding, and a monotone tercile pattern that
repeats in the other league is much harder to produce by chance than a significant coefficient.
"""
function tercile_cut(m::DataFrame, col::Symbol)
    v = filter(isfinite, m[!, col])
    (isempty(v) || length(unique(v)) < 3) && return DataFrame()
    lo, hi = quantile(v, 1/3), quantile(v, 2/3)
    band(x) = !isfinite(x) ? "na" : x <= lo ? "1_low" : x <= hi ? "2_mid" : "3_high"
    d = copy(m); d.band = band.(d[!, col])
    out = combine(groupby(d, :band),
                  nrow => :matches,
                  :legs  => sum => :legs,
                  col    => (x -> round(median(filter(isfinite, x); init = NaN), digits = 4)) => :median,
                  :skill => (x -> round(mean(x), digits = 5)) => :skill,
                  :pnl   => (x -> round(sum(x), digits = 4)) => :pnl)
    return sort!(out, :band)
end

# ===================================================================
# 4. Turning skill into a trust vector
# ===================================================================

"""
    derive_trust(df; min_legs = 8, default = 0.25, w_hi = 0.5, w_lo = 0.0) -> SelectionTrust

Build a `Portfolio.SelectionTrust` from measured per-selection skill.

Deliberately a THREE-LEVEL step, not a fitted continuum:

    skill significantly > 0   ->  w_hi     (trust the model here)
    interval straddles 0      ->  default  (no evidence either way)
    skill significantly < 0   ->  w_lo     (the market is better; do not trust the model)

Fitting a weight per family is the r08 failure mode — 15 free parameters against 267 legs will
find structure in noise every time. A step function has one decision per family and each decision
is a sign test with a clustered interval, which is about the most this sample can support. The
prior from the staking-layer stream already points here: curated per-line weights (0 on 1X2, ~0.5
on totals and BTTS) beat both empirical-Bayes and flat weights.

Keyed on `(group, line, selection)` to match `SelectionTrust`'s table exactly.
"""
function derive_trust(df::AbstractDataFrame; min_legs::Int = 8, default::Float64 = 0.25,
                      w_hi::Float64 = 0.5, w_lo::Float64 = 0.0)
    t = skill_table(df, [:group, :line, :selection]; min_legs = min_legs)
    table = Dict{Tuple{String,Float64,Symbol},Float64}()
    isempty(t) && return BayesianFootball.Portfolio.SelectionTrust(table; default = default,
                                                                  strict = false)
    for r in eachrow(t)
        r.enough || continue
        w = r.skill_lo > 0 ? w_hi : r.skill_hi < 0 ? w_lo : default
        table[(String(r.group), Float64(r.line), Symbol(r.selection))] = w
    end
    return BayesianFootball.Portfolio.SelectionTrust(table; default = default, strict = false)
end

"""
    apply_trust_oos(trust, df) -> DataFrame

Re-weight a HELD-OUT ledger by a trust vector derived elsewhere, and report what it bought.

The stakes are rescaled rather than re-solved. That is exact under this policy for the ordering
question being asked: `risk_factor` is homogeneous of degree 0, so once the drawdown constraint
binds a uniform rescale changes nothing and only the RELATIVE weights across families move the
book — which is precisely the intervention under test. It is not exact if the cap re-binds, so
`recapped` is reported.
"""
function apply_trust_oos(trust, df::AbstractDataFrame; cap_frac::Float64 = 0.25)
    d = copy(usable(df))
    isempty(d) && return DataFrame()
    w = [get(trust.table, (String(r.group), Float64(r.line), Symbol(r.selection)), trust.default)
         for r in eachrow(d)]
    d.trust_w = w
    d.stake   = d.stake .* w
    d.pnl     = d.stake .* d.payoff
    recap_slates!(d, cap_frac)
    return d
end
