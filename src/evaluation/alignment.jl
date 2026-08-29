# src/evaluation/alignment.jl
#
# Building the dense indexes, declaring what a metric needs, and verifying that the
# alignment those indexes produce is the one the legacy `innerjoin` would have produced.
#
# THE ALIGNMENT IS THE PART THAT CAN BE SILENTLY WRONG. Every number downstream is a
# reduction over (fixture, selection) pairs, and a pairing that is off by one fixture
# produces a metric that is plausible, stable across reruns, and meaningless. So the
# join is a named, inspectable object (`EvaluationRow`) rather than a `DataFrame`
# operation buried inside six separate kernels, and `verify_alignment` re-derives it.

export build_odds_view, extract_match_outcomes, build_evaluation_context,
       evaluation_rows, verify_alignment, AlignmentReport
export market_for_selection, selections_to_markets, market_selections,
       DEFAULT_SCORED_MARKETS, MIQ_DEFAULT_MARKETS
export scored_markets, scored_selections, needs_outcomes, needs_draws
export pit_values, pit_uniformity, PITReport


# ==============================================================================
# 1. WHAT A METRIC NEEDS
# ==============================================================================
#
# The legacy triggers name a metric's scope with a `Vector{Symbol}` of SELECTIONS —
# `LogLoss([:over_25])`. That is a POST-HOC FILTER: the pipeline prices every market in
# `DEFAULT_MARKET_CONFIG`, materialises a PPD row per fixture per selection, joins the
# lot against `ds.odds`, and only then throws away the 39 markets the metric never
# wanted. On a 900-fixture fold at 1,200 draws that is ~780 MB of posterior vectors
# built to answer a question about three of them.
#
# These four functions turn the same filter into a PRICING INSTRUCTION, without touching
# the trigger structs: `scored_markets` says what to build, `scored_selections` says what
# to score, and the two flags say which of the other two indexes are wanted at all.

"""
    scored_markets(rule) -> Vector{<:AbstractMarket}

The markets the probability pricer must build for this rule. Empty for rules that read
realised goal counts off the score grid rather than market prices (`CRPS`, `RQR`).
"""
scored_markets(::AbstractScoringRule) = AbstractMarket[]

"""
    scored_selections(rule) -> Vector{Symbol}

The selections the rule actually scores. Empty means "every outcome of every market in
`scored_markets(rule)`" — the legacy meaning of an empty `selections` vector.
"""
scored_selections(m::AbstractScoringRule) =
    hasproperty(m, :selections) ? Symbol[Symbol(s) for s in m.selections] : Symbol[]

"""
    needs_outcomes(rule) -> Bool

Whether the kernel reads realised `(home_score, away_score)` from `ds.matches`. Lets the
context skip building the outcome index for a batch that does not want it.
"""
needs_outcomes(::AbstractScoringRule) = false

"Whether the kernel needs the full posterior draw vector, or only its mean."
needs_draws(::AbstractScoringRule) = true

"""
    DEFAULT_SCORED_MARKETS

1X2, Over/Under 2.5 and BTTS.

NOT `Data.DEFAULT_MARKET_CONFIG`, which is 40-odd markets including the whole
Asian-handicap ladder. A default that prices that ladder to report a 1X2 log-loss is how
the legacy evaluation path came to spend most of its time inside `model_inference`.
"""
const DEFAULT_SCORED_MARKETS =
    AbstractMarket[Market1X2(), MarketOverUnder(2.5), MarketBTTS()]

"""
    MIQ_DEFAULT_MARKETS

The five markets covering the twelve selections `MIQResult` reports: 1X2, BTTS and the
three central Over/Under lines.
"""
const MIQ_DEFAULT_MARKETS = AbstractMarket[
    Market1X2(), MarketBTTS(),
    MarketOverUnder(1.5), MarketOverUnder(2.5), MarketOverUnder(3.5),
]

# --- the six legacy triggers, given a scope -----------------------------------
#
# Methods on the EXISTING trigger types in `metrics_methods/`, not new structs. The
# scope is derived from the trigger's own `selections` filter where it has one, so
# `LogLoss(:over_25)` prices `MarketOverUnder(2.5)` and nothing else while continuing to
# mean exactly what it meant before.

_scope_markets(sels) = isempty(sels) ? copy(DEFAULT_SCORED_MARKETS) :
                                       selections_to_markets(sels)

scored_markets(m::LogLoss) = _scope_markets(scored_selections(m))
needs_draws(::LogLoss)     = false

scored_markets(m::LPD) = _scope_markets(scored_selections(m))

scored_markets(m::GLMEdge) = _scope_markets(scored_selections(m))
needs_draws(::GLMEdge)     = false

scored_markets(::MIQ) = copy(MIQ_DEFAULT_MARKETS)

needs_outcomes(::CRPS) = true
needs_draws(::CRPS)    = false

needs_outcomes(::RQR) = true
needs_draws(::RQR)    = false


# ==============================================================================
# 2. SELECTION ↔ MARKET
# ==============================================================================
#
# The inverse that lets a legacy `LogLoss([:over_25])` tell the pricer it needs exactly
# `MarketOverUnder(2.5)`.
#
# Well-defined because `Data.Markets.outcomes` mints a DISTINCT symbol per market per
# line: `Market1X2()` owns `:home/:draw/:away`, `MarketBTTS()` owns
# `:btts_yes/:btts_no`, and `MarketOverUnder(L)` owns `:over_<L>/:under_<L>` with the
# decimal point deleted. No two markets in the standard config share a selection symbol,
# so this is a genuine inverse rather than a guess.

"""
    _parse_ou_line(digits) -> Float64 or nothing

Undo `replace(string(line), "." => "")`. The engine that produced the symbol always
emits a single fractional digit (`0.5`, `2.5`, `10.5`), so the last character is the
tenths place: `"25" → 2.5`, `"105" → 10.5`, `"05" → 0.5`.
"""
function _parse_ou_line(digits::AbstractString)
    length(digits) >= 2 || return nothing
    all(isdigit, digits) || return nothing
    return parse(Float64, digits[1:(end - 1)] * "." * digits[end])
end

"""
    market_for_selection(sel::Symbol) -> AbstractMarket or nothing

The market that owns `sel`, or `nothing` if no market in the three families this
evaluator prices does. `nothing` is not an error here — `selections_to_markets` decides
what to do with it, because a caller filtering on a Double-Chance or Asian-handicap
selection is asking for something these kernels do not price and the message belongs
there.
"""
function market_for_selection(sel::Symbol)
    sel in (:home, :draw, :away) && return Market1X2()
    sel in (:btts_yes, :btts_no) && return MarketBTTS()
    s = String(sel)
    for prefix in ("over_", "under_")
        if startswith(s, prefix)
            line = _parse_ou_line(s[(length(prefix) + 1):end])
            line === nothing || return MarketOverUnder(line)
        end
    end
    return nothing
end

"""
    selections_to_markets(sels) -> Vector{AbstractMarket}

The deduplicated markets a selection filter implies, in first-appearance order.

Throws on a selection this evaluator cannot price, naming it. The alternative — pricing
what it can and silently returning a metric computed over fewer rows than the caller
asked for — is the class of quiet wrong answer this whole path exists to remove.
"""
function selections_to_markets(sels)
    out = AbstractMarket[]
    for s in sels
        m = market_for_selection(Symbol(s))
        m === nothing && error(
            "selection :$(s) belongs to no market this evaluator prices. Supported " *
            "families: 1X2 (:home/:draw/:away), BTTS (:btts_yes/:btts_no) and " *
            "Over/Under (:over_25, :under_25, …). Pass `markets = [...]` explicitly " *
            "for anything else.")
        any(x -> x == m, out) || push!(out, m)
    end
    return out
end

"Every outcome symbol the given markets own, in `market_keys` order."
market_selections(markets) = Symbol[s for m in markets for s in market_keys(m)]


# ==============================================================================
# 3. BUILDING THE INDEXES
# ==============================================================================

"""
    _float_column(df, name) -> (Vector{Float64}, BitVector)

A frame column as values plus a presence mask. An ABSENT column comes back all-missing
rather than raising, so a store built without `odds_close` still supports `LogLoss`.

The value slot of an absent or `missing` entry is `NaN` rather than uninitialised memory:
the mask is what a kernel is supposed to read, but a stray `undef` would make a bug in
one look like noise instead of like a crash.
"""
function _float_column(df::AbstractDataFrame, name::Symbol)
    n = nrow(df)
    vals = fill(NaN, n)
    present = falses(n)
    hasproperty(df, name) || return vals, present
    col = getproperty(df, name)
    @inbounds for i in 1:n
        x = col[i]
        if x === missing
            vals[i] = NaN
        else
            vals[i] = Float64(x)
            present[i] = true
        end
    end
    return vals, present
end

"""
    build_odds_view(odds_df) -> OddsView

Flatten `ds.odds` into the concrete parallel vectors every kernel reads.

Errors — naming the column — if the frame is not the enriched long form
`Data.process_data(::OddsData)` produces, because every kernel downstream would
otherwise fail one at a time with a less useful message.
"""
function build_odds_view(df::AbstractDataFrame)
    n = nrow(df)
    for c in (:match_id, :market_name, :market_line, :selection)
        hasproperty(df, c) || error(
            "build_odds_view: the odds frame has no `$c` column. Expected the enriched " *
            "long form that `Data.process_data(::OddsData)` produces — match_id, " *
            "market_name, market_line, selection, prob_fair_close, is_winner.")
    end

    mid = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        mid[i] = Int(df.match_id[i])
    end

    sel = Vector{Symbol}(undef, n)
    @inbounds for i in 1:n
        s = df.selection[i]
        sel[i] = s isa Symbol ? s : Symbol(s)
    end

    mname = Vector{String}(undef, n)
    @inbounds for i in 1:n
        mname[i] = String(df.market_name[i])
    end

    mline = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        x = df.market_line[i]
        mline[i] = x === missing ? NaN : Float64(x)
    end

    fair, has_fair     = _float_column(df, :prob_fair_close)
    impl, has_impl     = _float_column(df, :prob_implied_close)
    oclose, has_oclose = _float_column(df, :odds_close)

    win = fill(Int8(-1), n)
    if hasproperty(df, :is_winner)
        col = df.is_winner
        @inbounds for i in 1:n
            x = col[i]
            x === missing && continue
            win[i] = Int8(x == true ? 1 : 0)
        end
    end

    return OddsView(n, mid, sel, mname, mline, fair, has_fair,
                    impl, has_impl, oclose, has_oclose, win)
end

OddsView(df::AbstractDataFrame) = build_odds_view(df)

"""
    extract_match_outcomes(matches_df) -> MatchOutcomes

`match_id → (home_score, away_score)` for every fixture whose result is recorded. A
fixture with a missing score is omitted, not sentinelled.
"""
function extract_match_outcomes(df::AbstractDataFrame)
    d = Dict{Int, Tuple{Int, Int}}()
    (hasproperty(df, :match_id) && hasproperty(df, :home_score) &&
     hasproperty(df, :away_score)) || return MatchOutcomes(d)
    sizehint!(d, nrow(df))
    @inbounds for i in 1:nrow(df)
        h = df.home_score[i]
        a = df.away_score[i]
        (h === missing || a === missing) && continue
        d[Int(df.match_id[i])] = (Int(h), Int(a))
    end
    return MatchOutcomes(d)
end

MatchOutcomes(df::AbstractDataFrame) = extract_match_outcomes(df)

"""
    build_evaluation_context(latents, odds_df, matches_df, metrics;
                             markets = nothing, max_goals = 12, threaded = true)
        -> EvaluationContext

Build the shared indexes for `metrics`.

The market list is the UNION of every rule's `scored_markets`, deduplicated, unless
`markets` is given explicitly. Two rules that want overlapping markets price them once
between them; a batch of `CRPS` and `RQR` alone prices NOTHING, because neither reads a
market price.

The odds view is likewise skipped (built over an empty frame) when no market is priced,
because every row of it would be unusable.
"""
function build_evaluation_context(l::AbstractPosteriorLatents,
                                  odds_df::AbstractDataFrame,
                                  matches_df::AbstractDataFrame,
                                  metrics::AbstractVector;
                                  markets = nothing,
                                  max_goals::Integer = Predictions.TPL_MAX_GOALS,
                                  threaded::Bool = true)
    mkts = AbstractMarket[]
    if markets === nothing
        for m in metrics, mk in scored_markets(m)
            any(x -> x == mk, mkts) || push!(mkts, mk)
        end
    else
        for mk in markets
            any(x -> x == mk, mkts) || push!(mkts, mk)
        end
    end

    keep = any(needs_draws, metrics) && !isempty(mkts)
    want_outcomes = any(needs_outcomes, metrics)

    odds = build_odds_view(isempty(mkts) ? similar(odds_df, 0) : odds_df)
    outs = want_outcomes ? extract_match_outcomes(matches_df) :
                           MatchOutcomes(Dict{Int, Tuple{Int, Int}}())
    probs = market_probabilities(l, mkts; keep_draws = keep,
                                 max_goals = max_goals, threaded = threaded)

    return EvaluationContext(l, odds, outs, probs, Int(max_goals))
end

build_evaluation_context(l::AbstractPosteriorLatents, odds_df, matches_df,
                         metric::AbstractScoringRule; kwargs...) =
    build_evaluation_context(l, odds_df, matches_df, [metric]; kwargs...)


# ==============================================================================
# 4. THE JOIN, AS AN OBJECT
# ==============================================================================

"""
    _selection_filter(metric) -> Set{Symbol} or nothing

`nothing` means "no filter" — the empty-`selections` case, which is what the legacy
kernels mean by `isempty(metric.selections)`. A `Set` rather than the vector because the
filter is tested once per odds row and the vectors are short but the frames are not.
"""
function _selection_filter(m::AbstractScoringRule)
    sels = scored_selections(m)
    return isempty(sels) ? nothing : Set{Symbol}(sels)
end

@inline _passes(::Nothing, ::Symbol) = true
@inline _passes(f::Set{Symbol}, s::Symbol) = s in f

"""
    evaluation_rows(ctx; selections = nothing, require_market = true,
                    require_outcome = true) -> Vector{EvaluationRow}

The aligned observations, in `ds.odds` ROW ORDER.

This is the legacy `innerjoin(ds.odds, model_features, on = [4 cols])` followed by
`dropmissing!`, materialised as a typed vector instead of a frame:

  * an odds row whose (match_id, selection) this evaluator did not price is dropped —
    the exact condition under which the inner join would have dropped it;
  * `require_market` reproduces `dropmissing!(df, :prob_fair_close)`;
  * `require_outcome` reproduces `dropmissing!(df, :is_winner)`.

ROW ORDER IS PRESERVED AND LOAD-BEARING. Every aggregate downstream is a `mean` or `sum`
over these rows, floating-point addition is not associative, and the legacy numbers were
accumulated in `ds.odds`' own order. Walking the odds table rather than the fixtures is
what makes the two agree to the last bit.
"""
function evaluation_rows(ctx::EvaluationContext;
                         selections = nothing,
                         require_market::Bool = true,
                         require_outcome::Bool = true)
    o = ctx.odds
    p = ctx.probs
    filt = selections === nothing ? nothing :
           (selections isa Set{Symbol} ? selections : Set{Symbol}(Symbol.(selections)))

    out = Vector{EvaluationRow}()
    sizehint!(out, o.n)

    @inbounds for i in 1:o.n
        require_market && !o.has_fair[i] && continue
        require_outcome && o.is_winner[i] < 0 && continue
        sel = o.selection[i]
        _passes(filt, sel) || continue
        r, c = _eval_locate(p, o.match_id[i], sel)
        (r == 0 || c == 0) && continue
        push!(out, EvaluationRow(o.match_id[i], sel, r, c, p.means[r, c],
                                 o.has_fair[i] ? o.prob_fair_close[i] : NaN,
                                 o.is_winner[i]))
    end
    return out
end


# ==============================================================================
# 5. VERIFYING THE ALIGNMENT
# ==============================================================================

"""
    AlignmentReport

What `verify_alignment` found. `ok` is the conjunction of the three conditions that must
hold for any downstream number to mean anything.

| field                | is                                                        |
|----------------------|-----------------------------------------------------------|
| `n_odds_rows`        | rows in the odds view                                     |
| `n_priced_fixtures`  | fixtures in the latent container                          |
| `n_matched`          | odds rows this evaluator priced                           |
| `n_unpriced`         | odds rows dropped for want of a price                     |
| `n_no_market`        | odds rows with no `prob_fair_close`                       |
| `n_no_outcome`       | odds rows with no settled `is_winner`                     |
| `n_scored`           | rows surviving every filter                               |
| `duplicate_keys`     | (match_id, selection) pairs appearing twice in the odds   |
| `mismatched_ids`     | rows whose model price came from the wrong fixture        |
| `ok`                 | no duplicates, no mismatches, and at least one scored row |
"""
struct AlignmentReport
    n_odds_rows::Int
    n_priced_fixtures::Int
    n_matched::Int
    n_unpriced::Int
    n_no_market::Int
    n_no_outcome::Int
    n_scored::Int
    duplicate_keys::Int
    mismatched_ids::Int
    ok::Bool
end

function Base.show(io::IO, r::AlignmentReport)
    print(io, "AlignmentReport(", r.ok ? "OK" : "PROBLEM", ": ", r.n_scored, "/",
          r.n_odds_rows, " scored, ", r.n_unpriced, " unpriced, ",
          r.duplicate_keys, " duplicate keys, ", r.mismatched_ids, " mismatched)")
end

function Base.show(io::IO, ::MIME"text/plain", r::AlignmentReport)
    println(io, "AlignmentReport — ", r.ok ? "OK" : "PROBLEM")
    println(io, "  odds rows        : ", r.n_odds_rows)
    println(io, "  priced fixtures  : ", r.n_priced_fixtures)
    println(io, "  matched          : ", r.n_matched)
    println(io, "  dropped, no price: ", r.n_unpriced)
    println(io, "  dropped, no line : ", r.n_no_market)
    println(io, "  dropped, unsettl.: ", r.n_no_outcome)
    println(io, "  scored           : ", r.n_scored)
    println(io, "  duplicate keys   : ", r.duplicate_keys)
    print(io,   "  mismatched ids   : ", r.mismatched_ids)
end

"""
    verify_alignment(ctx) -> AlignmentReport

Re-derive the join independently of `evaluation_rows` and check three things that must
hold before any metric computed from this context can be believed:

  1. NO DUPLICATE KEYS. Two odds rows with the same (match_id, selection) both match the
     same model price, so the fixture is silently double-weighted in every mean. The
     legacy `innerjoin` has the same exposure and no check.
  2. EVERY ROW'S PRICE COMES FROM ITS OWN FIXTURE. `probs.match_ids[row.fixture]` must
     equal `row.match_id`. This is the off-by-one-fixture failure, and it is the one that
     produces plausible numbers rather than an error.
  3. SOMETHING WAS SCORED. A context that matched nothing yields `NaN` metrics, which
     rank as "missing" rather than as "wrong" and are easy to miss in a table.

Cheap — one pass and one `Set` — so it is worth running before a long batch rather than
after it.
"""
function verify_alignment(ctx::EvaluationContext)
    o = ctx.odds
    p = ctx.probs

    seen = Set{Tuple{Int, Symbol}}()
    dupes = 0
    matched = 0
    unpriced = 0
    no_market = 0
    no_outcome = 0
    scored = 0
    mismatched = 0

    @inbounds for i in 1:o.n
        key = (o.match_id[i], o.selection[i])
        key in seen ? (dupes += 1) : push!(seen, key)

        r, c = _eval_locate(p, o.match_id[i], o.selection[i])
        if r == 0 || c == 0
            unpriced += 1
            continue
        end
        matched += 1
        p.match_ids[r] == o.match_id[i] || (mismatched += 1)

        o.has_fair[i] || (no_market += 1; continue)
        o.is_winner[i] < 0 && (no_outcome += 1; continue)
        scored += 1
    end

    ok = dupes == 0 && mismatched == 0 && scored > 0
    return AlignmentReport(o.n, length(p.match_ids), matched, unpriced,
                           no_market, no_outcome, scored, dupes, mismatched, ok)
end


# ==============================================================================
# 6. PROBABILITY INTEGRAL TRANSFORM
# ==============================================================================
#
# The alignment check above says the right posterior reached the right fixture. PIT says
# whether that posterior is the right SHAPE — and it is the only diagnostic here that
# looks at the goal distribution as a distribution rather than through a market price.
#
# For a CONTINUOUS predictive distribution F, `F(y)` is Uniform(0,1) when F is correctly
# specified. Goal counts are DISCRETE, so `F(y)` is not — it clusters on the atoms. The
# randomised transform repairs that:
#
#     u ~ Uniform( F(y − 1), F(y) )
#
# which is exactly the construction behind the randomised quantile residuals `RQR`
# reports; `pit_values` returns `u` and `RQR` returns `Φ⁻¹(u)`. Reported here on the
# uniform scale because that is the scale the Kolmogorov-Smirnov test wants and the scale
# a histogram is read on: a U-shape means the predictive is too narrow, a hump means it
# is too wide, and a slope means it is biased.

"""
    PITReport(u_home, u_away, ks_stat, p_value, n_obs)

Randomised PIT values for both sides plus the uniformity test over the pooled set.

`p_value` is the one-sample Kolmogorov-Smirnov test against `Uniform(0, 1)`. A SMALL
p-value says the predictive goal distribution is mis-specified; it does not say how, and
the `u` vectors are carried so a caller can look.

`NaN` for both statistics when fewer than three observations were available — a KS test
on two points is not a number worth reporting.
"""
struct PITReport
    u_home::Vector{Float64}
    u_away::Vector{Float64}
    ks_stat::Float64
    p_value::Float64
    n_obs::Int
end

function Base.show(io::IO, r::PITReport)
    print(io, "PITReport(", r.n_obs, " fixtures, KS = ",
          isnan(r.ks_stat) ? "—" : string(round(r.ks_stat, digits = 4)),
          ", p = ", isnan(r.p_value) ? "—" : string(round(r.p_value, digits = 4)), ")")
end

"""
    pit_values(ctx; rng = Xoshiro(42)) -> PITReport
    pit_values(latents, outcomes; rng = Xoshiro(42)) -> PITReport

Randomised probability integral transform of each side's realised goal count under the
model's plug-in marginal (see [`marginals`](@ref) for how that marginal is chosen).

SEEDED BY DEFAULT, with a private stream. Two calls on the same inputs agree, and
evaluating never perturbs the caller's global RNG — unlike the legacy `RQR` kernel,
whose unseeded `rand` makes its table different on every run.
"""
function pit_values(l::AbstractPosteriorLatents, outs::MatchOutcomes;
                    rng::AbstractRNG = Random.Xoshiro(42))
    ids = latent_match_ids(l)
    uh = Float64[]
    ua = Float64[]
    sizehint!(uh, length(ids))
    sizehint!(ua, length(ids))

    for i in eachindex(ids)
        sc = outcome_of(outs, ids[i])
        sc === nothing && continue
        dh, da = marginals(l, i)
        push!(uh, _pit_draw(sc[1], dh, rng))
        push!(ua, _pit_draw(sc[2], da, rng))
    end

    n = length(uh)
    if n < 3
        return PITReport(uh, ua, NaN, NaN, n)
    end
    pooled = vcat(uh, ua)
    d, p = try
        t = ExactOneSampleKSTest(pooled, Uniform(0.0, 1.0))
        (t.δ, pvalue(t))
    catch
        (NaN, NaN)
    end
    return PITReport(uh, ua, d, p, n)
end

pit_values(ctx::EvaluationContext; kwargs...) =
    pit_values(ctx.latents, ctx.outcomes; kwargs...)

"One randomised PIT value: `u ~ Uniform(F(y−1), F(y))`."
@inline function _pit_draw(y::Integer, dist::UnivariateDistribution, rng::AbstractRNG)
    lo = y > 0 ? cdf(dist, y - 1) : 0.0
    hi = cdf(dist, y)
    hi <= lo && return lo
    return rand(rng, Uniform(lo, hi))
end

"""
    pit_uniformity(report) -> (ks_stat, p_value)
    pit_uniformity(ctx; rng) -> (ks_stat, p_value)

Just the verdict, for a caller that does not want the vectors. A p-value below 0.05 says
the goal-count predictive is mis-specified at that level.
"""
pit_uniformity(r::PITReport) = (r.ks_stat, r.p_value)
pit_uniformity(ctx::EvaluationContext; kwargs...) = pit_uniformity(pit_values(ctx; kwargs...))
