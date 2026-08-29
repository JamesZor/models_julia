# src/evaluation/types.jl

export AbstractScoringRule
export AbstractEvaluationResult, AbstractMetricComponent

# --- The Triggers ---
abstract type AbstractScoringRule end

# --- The Containers ---
abstract type AbstractEvaluationResult end
abstract type AbstractMetricComponent end


# ==============================================================================
# THE TYPED EVALUATOR — DENSE INDEXES
# ==============================================================================
#
# Graduated from `current_development/08_unified_evaluation_framework/`.
#
# WHAT THESE REPLACE, AND WHAT THE OLD PATH COSTS
#
# Every kernel in `metrics_methods/` has the same four-step shape, and each step is
# paid ONCE PER METRIC:
#
#     ppd            = Predictions.model_inference(latents_raw)          # (1)
#     model_features = transform(ppd.df, :distribution => ByRow(mean))   # (2)
#     analysis_df    = innerjoin(ds.odds, model_features, on = [4 cols]) # (3)
#     dropmissing!(analysis_df, [...])                                   # (4)
#
#   (1) prices every market in `DEFAULT_MARKET_CONFIG` — 40-odd of them, including the
#       whole Asian-handicap ladder — to answer a question about three, and stores every
#       posterior draw as a `Vector{Float64}` inside a `Vector{Any}` DataFrame column,
#       which is the fragmentation `src/models/latents/` exists to remove, reintroduced
#       immediately downstream of it.
#   (2) collapses every one of those vectors to a mean, including the ~90% the metric
#       discards three lines later.
#   (3) is a four-column hash join between a ~50,000-row odds frame and an ~80,000-row
#       PPD frame, materialising a third with every column of both.
#   (4) copies it again.
#
# THE REPLACEMENT is three dense typed indexes built ONCE for a whole batch —
# `OddsView`, `MatchOutcomes`, `MarketProbabilities` — after which every kernel is a
# single pass over the odds rows with two integer lookups. No join, no `dropmissing`,
# no intermediate frame, no `Vector{Any}`.
#
# NOTHING BELOW REPLACES ANYTHING ABOVE. The legacy triggers, result containers and
# `compute_metric(metric, exp, ds, latents)` kernels in `metrics_methods/` are untouched
# and behave exactly as they did; the typed path adds `compute_metric(metric, ctx)`
# methods alongside them and `compat.jl` bridges between the two.

export OddsView, MatchOutcomes, MarketProbabilities, EvaluationContext,
       EvaluationRow, EvaluationWorkspace, EvaluationError
export prob_mean, prob_draws, outcome_of, probability_bytes, priced_selections


# ==============================================================================
# 1. THE ODDS VIEW
# ==============================================================================
#
# `ds.odds` is a DataFrame whose numeric columns are `Union{Missing, Float64}` and whose
# `match_id` is `Int32` (`src/Data/fetchers/sql/odds.jl`). A kernel that reads it row by
# row through `eachrow` pays a dynamic dispatch per cell.
#
# This converts the six columns any metric can want into concrete vectors, ONCE. The
# conversion is O(n) over a frame the legacy path hash-joins O(n) times PER METRIC, so
# it is strictly cheaper even before the join is counted.
#
# MISSINGNESS IS A BITVECTOR, NOT A NaN SENTINEL. `dropmissing!(df, :prob_fair_close)`
# drops `missing` and KEEPS a genuine `NaN`; collapsing the two would change which rows
# a metric scores. The `has_*` masks reproduce `dropmissing!` exactly.

"""
    OddsView(odds_df)

`ds.odds`, flattened into the concrete vectors every kernel reads.

`is_winner` is `Int8`: `1` won, `0` lost, `-1` the row's outcome is unknown. The
three-valued encoding is what lets a kernel reproduce `dropmissing!(df, :is_winner)`
with an integer comparison instead of a frame copy.

Build one with [`build_odds_view`](@ref); the constructor is the same thing under the
type's own name.
"""
struct OddsView
    n::Int
    match_id::Vector{Int}
    selection::Vector{Symbol}
    market_name::Vector{String}
    market_line::Vector{Float64}
    prob_fair_close::Vector{Float64}
    has_fair::BitVector
    prob_implied_close::Vector{Float64}
    has_implied::BitVector
    odds_close::Vector{Float64}
    has_odds_close::BitVector
    is_winner::Vector{Int8}
end

Base.length(o::OddsView) = o.n

Base.show(io::IO, o::OddsView) =
    print(io, "OddsView(", o.n, " rows, ", length(unique(o.selection)), " selections)")


# ==============================================================================
# 2. THE OUTCOME INDEX
# ==============================================================================

"""
    MatchOutcomes(scores)

`match_id → (home_score, away_score)` for every fixture whose result is recorded.

A fixture with a missing score is ABSENT rather than present-with-a-sentinel, so a
kernel that asks for it gets `nothing` and skips the row. The legacy CRPS and RQR
kernels `innerjoin` against `ds.matches` and do NOT drop missing scores, so a store
holding a not-yet-played fixture makes `compute_crps(missing, …)` raise a `MethodError`
several minutes into an evaluation.

Build one with [`extract_match_outcomes`](@ref).
"""
struct MatchOutcomes
    scores::Dict{Int, Tuple{Int, Int}}
end

"The realised scoreline for `match_id`, or `nothing` when it is not recorded."
@inline outcome_of(o::MatchOutcomes, mid::Integer) = get(o.scores, Int(mid), nothing)

Base.length(o::MatchOutcomes) = length(o.scores)
Base.show(io::IO, o::MatchOutcomes) =
    print(io, "MatchOutcomes(", length(o.scores), " fixtures)")


# ==============================================================================
# 3. THE PRICED POSTERIOR
# ==============================================================================

"""
    MarketProbabilities

Every posterior market probability the batch needs, in ONE tensor.

| field        | shape                                      |
|--------------|--------------------------------------------|
| `draws`      | `n_draws × n_fixtures × n_selections`      |
| `means`      | `n_fixtures × n_selections`                |
| `match_ids`  | `n_fixtures`, the latent container's order |
| `selections` | `n_selections`, `market_keys` order        |

DRAW-MAJOR ON PURPOSE. `view(draws, :, i, c)` is then CONTIGUOUS, which matters twice:
it is what LPD and MIQ iterate, and a contiguous `SubArray` reduces with the same
pairwise blocking Base uses on a `Vector`, so `mean(view(...))` is bit-identical to the
`mean` of the `Vector{Float64}` the legacy PPD frame stored. Parity with the legacy
kernels depends on that.

`draws` is EMPTY (`0 × 0 × 0`) when every metric in the batch only needs means — LogLoss
and GLMEdge do, LPD and MIQ do not. On a 900-fixture fold at 1,200 draws and eleven
selections that is ~95 MB not allocated.
"""
struct MarketProbabilities
    markets::Vector{AbstractMarket}
    match_ids::Vector{Int}
    selections::Vector{Symbol}
    row_of::Dict{Int, Int}
    col_of::Dict{Symbol, Int}
    draws::Array{Float64, 3}
    means::Matrix{Float64}
    n_draws::Int
    keep_draws::Bool
end

Base.show(io::IO, p::MarketProbabilities) = print(
    io, "MarketProbabilities(", length(p.match_ids), " fixtures × ",
    length(p.selections), " selections × ", p.n_draws, " draws",
    p.keep_draws ? "" : ", means only", ")")

"Bytes the tensor holds. The number to set against the legacy PPD frame's."
probability_bytes(p::MarketProbabilities) = sizeof(p.draws) + sizeof(p.means)

"The selection symbols this container actually priced, in tensor-column order."
priced_selections(p::MarketProbabilities) = copy(p.selections)

@inline function _eval_locate(p::MarketProbabilities, mid::Integer, sel::Symbol)
    r = get(p.row_of, Int(mid), 0)
    r == 0 && return (0, 0)
    c = get(p.col_of, sel, 0)
    return (r, c)
end

"""
    prob_mean(p, match_id, selection) -> Float64 or nothing

The model's mean probability, or `nothing` when this framework did not price that
(fixture, selection) — the exact condition under which `innerjoin` would have dropped
the row.
"""
@inline function prob_mean(p::MarketProbabilities, mid::Integer, sel::Symbol)
    r, c = _eval_locate(p, mid, sel)
    (r == 0 || c == 0) && return nothing
    return @inbounds p.means[r, c]
end

"""
    prob_draws(p, match_id, selection) -> contiguous view or nothing

The full posterior of that probability. `nothing` if unpriced; raises if the tensor was
built means-only, because silently returning the mean would turn LPD into log-loss.
"""
@inline function prob_draws(p::MarketProbabilities, mid::Integer, sel::Symbol)
    p.keep_draws || error(
        "prob_draws: this MarketProbabilities was built means-only. Pass " *
        "`keep_draws = true`, or include a metric whose `needs_draws` is true.")
    r, c = _eval_locate(p, mid, sel)
    (r == 0 || c == 0) && return nothing
    return @inbounds view(p.draws, :, r, c)
end


# ==============================================================================
# 4. THE EVALUATION CONTEXT
# ==============================================================================

"""
    EvaluationContext(latents, odds, outcomes, probs, max_goals)

Everything a batch of metrics reads, built once and shared.

This is the structural fix. `evaluate_experiments` hands every metric the same `latents`
object and lets each one rebuild the PPD, the join and the cleaned frame for itself;
here the three indexes are built once from the union of what the metrics declared
(`scored_markets`, `needs_outcomes`, `needs_draws`) and every kernel reads them.

Build one with [`build_evaluation_context`](@ref).
"""
struct EvaluationContext{L<:AbstractPosteriorLatents}
    latents::L
    odds::OddsView
    outcomes::MatchOutcomes
    probs::MarketProbabilities
    max_goals::Int
end

function Base.show(io::IO, c::EvaluationContext)
    print(io, "EvaluationContext(", n_matches(c.latents), " fixtures, ",
          c.odds.n, " odds rows, ", c.probs, ")")
end


# ==============================================================================
# 5. THE ALIGNED OBSERVATION
# ==============================================================================

"""
    EvaluationRow(match_id, selection, fixture, column, model_prob, market_prob, outcome)

One aligned (fixture, selection) observation: the single row a scoring kernel consumes.

| field         | is                                                            |
|---------------|---------------------------------------------------------------|
| `match_id`    | the fixture                                                   |
| `selection`   | the market outcome, e.g. `:home`, `:over_25`                  |
| `fixture`     | row index into the latent container / probability tensor      |
| `column`      | column index into the probability tensor                      |
| `model_prob`  | the model's posterior MEAN probability                        |
| `market_prob` | `prob_fair_close`, or `NaN` when the book did not quote it    |
| `outcome`     | `1` won, `0` lost, `-1` unknown                               |

`fixture` and `column` are carried rather than re-derived so a kernel that also wants
the full posterior gets it with `view(probs.draws, :, row.fixture, row.column)` and no
second dictionary lookup. `isbits`, so a `Vector{EvaluationRow}` is one dense block.

This is the type that makes the alignment TESTABLE: `evaluation_rows` is the join the
legacy path performs with `innerjoin` + `dropmissing!`, and its output can be compared
against that join row for row.
"""
struct EvaluationRow
    match_id::Int
    selection::Symbol
    fixture::Int
    column::Int
    model_prob::Float64
    market_prob::Float64
    outcome::Int8
end

Base.show(io::IO, r::EvaluationRow) = print(
    io, "EvaluationRow(", r.match_id, " ", r.selection, ": model ",
    round(r.model_prob, digits = 4), ", market ", round(r.market_prob, digits = 4),
    ", y = ", r.outcome, ")")

"Did this row's selection win? Only meaningful when `outcome >= 0`."
@inline row_won(r::EvaluationRow) = r.outcome == 1

"`1.0`/`0.0` for the realised outcome, as the scalar formulae want it."
@inline row_y(r::EvaluationRow) = Float64(r.outcome)


# ==============================================================================
# 6. THE PRICING WORKSPACE
# ==============================================================================

"""
    EvaluationWorkspace

Preallocated scratch for [`price_match_markets!`](@ref): one per WORKER, never one per
fixture and never one per market.

| field     | is                                                              |
|-----------|-----------------------------------------------------------------|
| `grid`    | the `(max_goals × max_goals × n_draws)` score-grid destination   |
| `ws`      | `Predictions.GridWorkspace` — the marginal-PMF scratch           |
| `books`   | one `alloc_market_book` per market, as a `Tuple`                 |
| `markets` | the markets, as a `Tuple` so the pricing loop unrolls            |
| `offsets` | the tensor column each market's outcomes start at                |
| `smile`   | `SmileScoreGrid` for the smile family, `nothing` otherwise       |

`markets` and `books` are `Tuple`s rather than `Vector`s so their length and element
types are in the workspace's type and the per-fixture loop over markets is unrolled
before it runs — the same reason the composable model builder freezes its covariate
vector into a tuple.

`smile` is a type parameter rather than a `Union` field so the smile branch is resolved
at compile time and the ordinary path emits no test for it.
"""
struct EvaluationWorkspace{M<:Tuple, B<:Tuple, S<:Union{Nothing, SmileScoreGrid}}
    grid::Array{Float64, 3}
    ws::GridWorkspace
    books::B
    markets::M
    offsets::Vector{Int}
    smile::S
    max_goals::Int
end

Base.show(io::IO, w::EvaluationWorkspace) = print(
    io, "EvaluationWorkspace(", length(w.markets), " markets, max_goals=", w.max_goals,
    w.smile === nothing ? "" : ", smile", ")")


# ==============================================================================
# 7. A COLLECTED FAILURE
# ==============================================================================

"""
    EvaluationError(model, metric, message)

One metric that raised on one fit. COLLECTED rather than thrown, so a batch of eleven
models does not lose ten results because the eleventh has no odds coverage — and
REPORTED rather than swallowed, which is the half `evaluate_experiments`' `try/catch`
gets wrong: it `@warn`s, drops the whole model's row, and the leaderboard is silently
short.
"""
struct EvaluationError
    model::String
    metric::String
    message::String
end

Base.show(io::IO, e::EvaluationError) =
    print(io, "EvaluationError(", e.model, " / ", e.metric, ": ", e.message, ")")
