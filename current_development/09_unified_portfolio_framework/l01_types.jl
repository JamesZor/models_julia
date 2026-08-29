# ==============================================================================
# 09 — UNIFIED PORTFOLIO & STAKING FRAMEWORK : TYPES
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# ------------------------------------------------------------------------------
# THE ONE DESIGN DECISION THAT SHAPES EVERY OTHER FILE
# ------------------------------------------------------------------------------
#
# `src/Portfolio/` already has a good type hierarchy. `Selection`, `MatchBook`,
# `Slate`, `Trajectory`, `BookSpec`, `PolicySpec`, `ExecutionConfig` and the nine
# abstract seams describe exactly the objects this framework needs, with exactly the
# fields it needs. The briefing's type list differs from them in NAMES ONLY —
# `sels` vs `selections`, `R` vs `payoff_matrix`, `a_kelly` vs `raw_alloc`, `m_id`
# vs `match_id` — and adds no field that `src` does not already carry.
#
# So this framework ALIASES them rather than redeclaring them.
#
#   const MatchBook = BayesianFootball.Portfolio.MatchBook
#
# not
#
#   struct MatchBook ... end     # same fields, different names
#
# THREE REASONS, in descending order of how much they cost to get wrong:
#
#   1. TWO TYPES ANSWERING TO ONE NAME IN ONE SESSION IS THE FAILURE MODE THIS
#      WHOLE PROTOTYPE LINE EXISTS TO REMOVE. `06_typed_posterior_latents` renamed
#      its smile container `SmileScoreGrid` for precisely this reason: a parity
#      harness has to hold both implementations at once, and if
#      `Portfolio.MatchBook` and `UnifiedPortfolio.MatchBook` were different structs
#      with the same name, every `MethodError` and every `show` in this directory
#      would be ambiguous to read.
#
#   2. BACKWARD COMPATIBILITY BECOMES IDENTITY RATHER THAN EMULATION. A book built
#      by this framework IS a `Portfolio.MatchBook`, so `Portfolio.group`,
#      `Portfolio.stake_slate`, `Portfolio.simulate`, `Portfolio.stake_sheet`,
#      `Portfolio.attribution` and every cached `Vector{MatchBook}` in a `.jls` on
#      disk keep working — not because a bridge translates them, but because there
#      is nothing to translate. §3.4 of the briefing asks for "legacy call sites
#      execute completely unmodified"; this is the strongest available form of it.
#
#   3. THE PARITY CLAIM GETS SHARPER. `l05_parity.jl` compares two BUILDERS over one
#      set of types. If the types differed, every comparison would be
#      field-by-field transcription and a field this framework forgot to copy would
#      read as a parity pass.
#
# The briefing's field names are NOT lost. §2.4 below gives one accessor per name,
# so `book_payoff(b)` and `sel_odds_settle(s)` read the way the briefing writes them.
#
# WHAT IS GENUINELY NEW, AND THEREFORE DECLARED HERE RATHER THAN ALIASED:
#
#   OddsIndex       the concretely-typed odds columns + a match → rows index, which
#                   is what removes `odds_df.match_id .== match_id` from the hot loop
#   BookWorkspace   the one score grid, one grid workspace and one destination book
#                   per market that the whole fold reuses (06's kernels, §5)
#   BuildReport     what the builder skipped and why, plus the convergence verdict
#   DailyState      one slate's settlement, as a row rather than five parallel vectors
#   PortfolioSummary / BootstrapCI / PortfolioResult
#                   the metric set the briefing asks for (CAGR, Sharpe, Sortino,
#                   win rate, 1X2 ROI, bootstrap CIs), which `Portfolio.report` has
#                   no fields for
#
# ==============================================================================


# ==============================================================================
# 0. DEPENDENCIES
# ==============================================================================
#
# One include pulls the whole stack: `08` → `07` → `06` → `05`. Same shape as
# `08/l01_types.jl:58`, one level further down.

import BayesianFootball

include(joinpath(@__DIR__, "..", "08_unified_evaluation_framework", "l04_compat_bridge.jl"))

using DataFrames
using Dates
using Distributions
using LinearAlgebra
using Printf
using Random
using Statistics

using .UnifiedEvaluation

# Named imports rather than a blanket `using`, because these are the names the hot
# loop calls and a reader should be able to see where each one comes from.
using .UnifiedEvaluation:
    # 06 — typed posterior containers
    AbstractPosteriorLatents, CountLatents, RecombLatents, SmileLatents,
    n_matches, n_draws, latent_match_ids, match_index,
    to_legacy_dataframe, latents_from_legacy_dataframe,
    # 06 — zero-allocation kernels
    GridWorkspace, alloc_score_grid, compute_score_grid!, compute_score_grid,
    price_market!, price_market, market_keys, market_arity, alloc_market_book,
    SmileScoreGrid, alloc_smile_buffers, fill_smile_buffers!, TPL_MAX_GOALS,
    # 07 — inference outcome
    Fit, FitConfig, FoldFit, ConvergenceSummary, fit_name,
    # 08 — the convergence gate, reused verbatim
    ConvergenceRefusal, convergence_verdict, fit_latents, as_typed_latents,
    # 08 — the deterministic fixtures `l05_parity.jl` §6 assembles a store from
    simulate_scores, synthetic_odds,
    # 06/08 — the parity and cost reporting vocabulary. NOT exported by `08`, and
    # named here rather than reimplemented: a second ULP comparator in this
    # repository would be one too many, and two that disagreed would be worse.
    ParityRow, tpl_compare, tpl_parity_table, ulp_distance,
    CostRow, cost_table, speedup, shrink

const UP_BF   = BayesianFootball
const UP_D    = BayesianFootball.Data
const UP_Pred = BayesianFootball.Predictions
const UP_PF   = BayesianFootball.Portfolio
const UP_BT   = BayesianFootball.BackTesting

using BayesianFootball.Data: AbstractMarket, Market1X2, MarketBTTS, MarketOverUnder,
                             market_group, market_line, outcomes


# ==============================================================================
# 1. THE ABSTRACT SEAMS
# ==============================================================================
#
# Aliased, not redeclared. A component is a struct plus one method on one of these
# functions; redeclaring the abstract type here would mean every `FlatTrust`,
# `SlateDrawdown` and `FixedCap` anyone has ever written no longer subtypes the
# thing this framework dispatches on, and the "add a struct + one method"
# extension story (`src/Portfolio/interfaces.jl`) would silently apply to the wrong
# hierarchy.

const AbstractPricePolicy     = UP_PF.AbstractPricePolicy
const AbstractCommissionModel = UP_PF.AbstractCommissionModel
const AbstractAllocator       = UP_PF.AbstractAllocator
const AbstractShrinkage       = UP_PF.AbstractShrinkage
const AbstractTrustModel      = UP_PF.AbstractTrustModel
const AbstractRiskModel       = UP_PF.AbstractRiskModel
const AbstractExposureCap     = UP_PF.AbstractExposureCap
const AbstractSelectionFilter = UP_PF.AbstractSelectionFilter
const AbstractSlateGrouping   = UP_PF.AbstractSlateGrouping

# The seam METHODS, likewise. `allocate` is Jacot & Mochkovitch's non-mutually-
# exclusive Kelly solve; `shrink_factor` is Baker & McHale; `risk_factor` is
# Busseti-Ryu-Boyd. All three are correct, all three are audited by
# `test/portfolio_tests.jl`, and none of them is on the path this framework speeds
# up — the cost being removed is the per-fixture `(12 × 12 × n_draws)` tensor and
# the `Vector{Any}` unboxing in front of them, not the convex solve itself.
#
# Reimplementing them would put a second Kelly solver in the repository whose only
# job is to agree with the first one to the last bit. That is a liability, not a
# feature.

const settlement_odds = UP_PF.settlement_odds
const net_return      = UP_PF.net_return
const allocate        = UP_PF.allocate
const shrink_factor   = UP_PF.shrink_factor
const trust_for       = UP_PF.trust_for
const trust_vector    = UP_PF.trust_vector
const risk_factor     = UP_PF.risk_factor
const apply_cap       = UP_PF.apply_cap
const keep            = UP_PF.keep
const group           = UP_PF.group

# The payoff morphism (`src/Portfolio/payoff.jl`). Single-sourced through
# `Data.grade_selection`, which is the same grader that writes `is_winner` in the
# odds pipeline — that is what makes "the win mask agrees with settlement" an
# identity rather than a convention, and it must stay single-sourced.
const payoff        = UP_PF.payoff
const payoff_matrix = UP_PF.payoff_matrix
const settle_vector = UP_PF.settle_vector
const grid_index    = UP_PF.grid_index

# Concrete components, re-exported so a caller needs one `using`.
const PerBetCommission  = UP_PF.PerBetCommission
const NetMarketCommission = UP_PF.NetMarketCommission
const NoCommission      = UP_PF.NoCommission
const DeArb             = UP_PF.DeArb
const Normalise         = UP_PF.Normalise
const RawPrice          = UP_PF.RawPrice
const KellyLogUtility   = UP_PF.KellyLogUtility
const NoShrinkage       = UP_PF.NoShrinkage
const FractionalKelly   = UP_PF.FractionalKelly
const BakerMcHale       = UP_PF.BakerMcHale
const FlatTrust         = UP_PF.FlatTrust
const SelectionTrust    = UP_PF.SelectionTrust
const ScheduledTrust    = UP_PF.ScheduledTrust
const NoRisk            = UP_PF.NoRisk
const IsolatedDrawdown  = UP_PF.IsolatedDrawdown
const SlateDrawdown     = UP_PF.SlateDrawdown
const risk_lambda       = UP_PF.risk_lambda
const FixedCap          = UP_PF.FixedCap
const VolTargetCap      = UP_PF.VolTargetCap
const KeepAll           = UP_PF.KeepAll
const MinEdge           = UP_PF.MinEdge
const MarketWhitelist   = UP_PF.MarketWhitelist
const MinOdds           = UP_PF.MinOdds
const FilterChain       = UP_PF.FilterChain
const DailySlate        = UP_PF.DailySlate
const SingleMatchSlate  = UP_PF.SingleMatchSlate


# ==============================================================================
# 2. DOMAIN OBJECTS
# ==============================================================================

const Selection       = UP_PF.Selection
const MatchBook       = UP_PF.MatchBook
const Slate           = UP_PF.Slate
const SlateContext    = UP_PF.SlateContext
const SlateAllocation = UP_PF.SlateAllocation
const Trajectory      = UP_PF.Trajectory
const FixtureInfo     = UP_PF.FixtureInfo

# --- 2.4 the briefing's names, as accessors -----------------------------------
#
# One function per field the briefing spells differently. Functions rather than a
# `getproperty` overload, because `Base.getproperty(::Portfolio.MatchBook, …)` would
# be type piracy — a method on someone else's function for someone else's type,
# visible to every package in the session — and because a `MatchBook` field read
# sits inside `stake_slate`'s inner loop where an added dispatch layer is not free.
#
# The names are prefixed (`book_`, `sel_`) so none of them collides with a legacy
# export: `payoff_matrix` already names the `(sels, max_h, max_a, commission)`
# constructor above, and shadowing it with a one-field getter would be a trap.

"`MatchBook.m_id` — the briefing's `match_id`."
book_match_id(b::MatchBook) = b.m_id
"`MatchBook.date`."
book_date(b::MatchBook) = b.date
"`MatchBook.sels` — the briefing's `selections`."
book_selections(b::MatchBook) = b.sels
"`MatchBook.p_grid` — posterior-mean score grid, normalised to 1."
book_grid(b::MatchBook) = b.p_grid
"`MatchBook.R` — the briefing's `payoff_matrix`. Jacot return matrix, `N × n`."
book_payoff(b::MatchBook) = b.R
"`MatchBook.settle` — the briefing's `settle_vector`. `nothing` for an unplayed fixture."
book_settle(b::MatchBook) = b.settle
"`MatchBook.a_kelly` — the briefing's `raw_alloc`. Full-size allocation on the posterior mean."
book_alloc(b::MatchBook) = b.a_kelly
"`MatchBook.k_shrink` — the briefing's `shrink_k`."
book_shrink(b::MatchBook) = b.k_shrink
"`MatchBook.kkt` — worst first-order-condition violation of `a_kelly`."
book_kkt(b::MatchBook) = b.kkt
"`MatchBook.converged` — did the ALLOCATOR converge. Not the MCMC verdict; see `BuildReport`."
book_converged(b::MatchBook) = b.converged

"`Selection.selection` — the briefing's `sel`."
sel_name(s::Selection) = s.selection
"`Selection.odds_quoted` — the briefing's `odds_close`. The price as traded."
sel_odds_close(s::Selection) = s.odds_quoted
"`Selection.odds_used` — the briefing's `odds_settle`. What we are settled at, post `AbstractPricePolicy`."
sel_odds_settle(s::Selection) = s.odds_used
"`Selection.p_model` — the briefing's `prob_model`."
sel_prob_model(s::Selection) = s.p_model
"`Selection.p_market` — the briefing's `prob_market`. Vig-removed; a BENCHMARK, never a price."
sel_prob_market(s::Selection) = s.p_market
"`p_model - p_market`, in probability points on one de-vigged scale."
sel_edge(s::Selection) = s.p_model - s.p_market


# ==============================================================================
# 3. CONFIGURATION
# ==============================================================================
#
# Aliased for the reason in §1, plus one specific to these three: `BookSpec` is the
# CACHE KEY (`book_cache_key`, `component_hash`). A framework that declared its own
# would compute a different key for a spec that is field-for-field the same, and
# every serialised book cache in `data/` would miss silently on the first run and
# then be rebuilt with the wrong provenance.
#
# THE BRIEFING'S `PolicySpec(allocator, caps, commission, risk, shrinkage, trust)`
# IS NOT ADOPTED, deliberately. `allocator`, `commission` and `shrinkage` change the
# MatchBook and therefore belong to `BookSpec`; `trust`, `risk`, `cap`, `filter` and
# `grouping` are pure post-multipliers on a built book and belong to `PolicySpec`.
# That split is the reason a policy sweep does not rebuild books, which is the
# reason walk-forward evaluation is affordable at all
# (`src/Portfolio/portfolio-module.jl:22-26`). Moving three fields across the line
# would turn every policy sweep back into a full rebuild.

const ExecutionConfig  = UP_PF.ExecutionConfig
const BookSpec         = UP_PF.BookSpec
const PolicySpec       = UP_PF.PolicySpec
const PortfolioSystem  = UP_PF.PortfolioSystem

const component_hash  = UP_PF.component_hash
const book_cache_key  = UP_PF.book_cache_key


# ==============================================================================
# 4. THE ODDS INDEX
# ==============================================================================
#
# WHAT THIS REPLACES. `extract_selections` (book.jl:29) opens with
#
#     rows = view(odds_df, odds_df.match_id .== match_id, :)
#
# and then, once per market,
#
#     sub = view(rows, (rows.market_name .== grp) .& isapprox.(rows.market_line, line), :)
#
# The first line allocates a `BitVector` over the WHOLE odds frame for every
# fixture: on a 500-fixture fold against a 50,000-row frame that is 500 full scans,
# 25 million comparisons to find ~11 rows each time. The second allocates two more
# full-length masks per market per fixture.
#
# `OddsIndex` does one pass, stores the five columns this pipeline reads as
# concretely-typed vectors, and hands `build_book` a `UnitRange` of row positions.
#
# ROW ORDER IS PRESERVED EXACTLY. `view(df, mask, :)` yields rows in ascending
# original-row order, and so does this index: rows are bucketed by match in a stable
# sort. That matters because `extract_selections` builds a `Dict{Symbol,Float64}`
# whose LAST write wins on a duplicate quote, and because the resulting selection
# order fixes the column order of the payoff matrix `R` — which fixes the starting
# point handed to the Kelly solver. A different order is a different answer in the
# last bits.
#
# MISSINGNESS. `odds_close` is stored with `NaN` in place of `missing`, and the
# admission test is `isnan(o) || o <= 1.0` — NOT `o <= 1.0`, because `NaN <= 1.0` is
# `false` and a naive port would admit every missing quote as a valid price.
# `market_line` may NOT be missing: the legacy predicate
# `(rows.market_name .== grp) .& isapprox.(rows.market_line, line)` raises on a
# `missing` line rather than skipping it, so a frame carrying one has never worked
# and this refuses it by name instead of at a `BoundsError` three frames down.

"""
    OddsIndex(odds_df)

The five odds columns this pipeline reads, unboxed once into concrete vectors, plus
`match_id → row positions`.

| field | is |
|---|---|
| `order` | row positions of `odds_df`, grouped by match, ascending within a match |
| `rows`  | `match_id` → the `UnitRange` of `order` holding that match's rows |
| the rest | `odds_df`'s columns permuted into `order`, concretely typed |

Built once per fold and shared by every `build_book` call. `nrow(odds_df)` work,
once, in place of `n_fixtures × nrow(odds_df)`.
"""
struct OddsIndex
    rows::Dict{Int, UnitRange{Int}}
    match_id::Vector{Int}
    market_name::Vector{String}
    market_line::Vector{Float64}
    selection::Vector{Symbol}
    odds_close::Vector{Float64}
    n_source_rows::Int
end

Base.length(oi::OddsIndex) = length(oi.match_id)
Base.haskey(oi::OddsIndex, m_id::Integer) = haskey(oi.rows, Int(m_id))

function Base.show(io::IO, oi::OddsIndex)
    print(io, "OddsIndex(", length(oi.rows), " matches, ", length(oi), " quotes",
          length(oi) == oi.n_source_rows ? "" : " of $(oi.n_source_rows) rows", ")")
end


# ==============================================================================
# 5. THE BOOK WORKSPACE
# ==============================================================================
#
# WHAT THIS REPLACES. `build_book` (book.jl:86-94) does, per fixture:
#
#     score_matrix = Predictions.compute_score_matrix(model, extract_params(model, row))
#     model_probs  = Dict(string(m) => compute_market_probs(score_matrix, m) for m in markets)
#
# `compute_score_matrix` allocates a fresh `(12 × 12 × n_draws)` `Float64` tensor —
# 1.4 MB at 1,200 draws, 3.7 MB at 3,200 — and `extract_params` unboxes a
# `Vector{Any}` DataFrame row to feed it. `compute_market_probs` then allocates one
# `Dict{Symbol,Vector{Float64}}` per market, keyed by a freshly-`string`ed market
# name, and each of its vectors is another `n_draws` heap object.
#
# A 500-fixture fold at 3,200 draws is therefore ~1.8 GB of tensor churn and ~15,000
# dictionary and vector allocations, all of it live for microseconds. The garbage
# collector is the dominant cost of a backtest and none of the memory does any work.
#
# HERE: one tensor, one workspace and one destination vector per market outcome for
# the WHOLE fold, overwritten in place per fixture by `06`'s kernels.
#
# WHY THE SLOTS ARE THREE CONCRETELY-TYPED VECTORS AND NOT ONE `Vector{Any}`.
# `spec.markets.markets` is a `Vector{AbstractMarket}`, so a loop over it dispatches
# dynamically on every iteration. That is survivable for 5 markets — but a dynamic
# call whose return value is a `Union` of tuple types boxes, and a zero-allocation
# claim that depends on Julia's escape analysis eliding that box is not a claim
# worth making (`06/l03_score_grids.jl`, RULE 2). Three homogeneous vectors give
# three statically-dispatched loops and `@allocated == 0` that is true by
# construction rather than by optimisation.
#
# `order` preserves the caller's market ORDER across the three vectors, because
# `extract_selections` must iterate `spec.markets.markets` in the sequence the
# legacy loop does — that sequence is the order selections land in the book, and
# hence the column order of `R`.

"""
    MarketSlot{M, N}

One market's pricing destination: the market itself, its group/line (hoisted out of
the hot loop), its `N` outcome symbols, and the `N` preallocated draw vectors
`price_market!` writes into.
"""
struct MarketSlot{M, N}
    market::M
    group::String
    line::Float64
    keys::NTuple{N, Symbol}
    book::NTuple{N, Vector{Float64}}
end

"""
    FallbackSlot

A market `06`'s kernels have no `price_market!` method for — the Asian-handicap
ladder, correct score, double chance.

These are priced through `Predictions.compute_market_probs` on a `ScoreMatrix` VIEW
of the shared grid, so they still avoid the per-fixture tensor, but they allocate a
`Dict` and `N` vectors per fixture like the legacy path does. Excluded from the
zero-allocation gate by construction, counted in `BuildReport.fallback_markets`, and
warned about once per workspace so the cost is never silent.
"""
struct FallbackSlot
    market::AbstractMarket
    group::String
    line::Float64
    keys::Vector{Symbol}
end

"""
    BookWorkspace(spec, latents; max_goals = 12)

Everything `build_book` needs that does not depend on WHICH fixture it is looking at.

Allocate **one per worker**, not one per fixture — that inversion is the entire
point of the file. Reused across every fixture of a fold:

| field | is | bytes |
|---|---|---|
| `S` | the shared `(max_goals² × n_draws)` score grid | `144 · n_draws · 8` |
| `ws` | `06`'s `GridWorkspace` — the two marginal PMF buffers | `2 · max_goals · 8` |
| `grid` | what `price_market!` reads: `S`, or a `SmileScoreGrid` VIEWING `S` | 0 |
| `slots_*` | one destination vector per market outcome | `n_out · n_draws · 8` |
| `mean_buf` | destination for `mean(S, dims = 3)` | `144 · 8` |

`grid` is parametric so the smile route is a compile-time fact rather than a branch:
a `SmileLatents` builds a `SmileScoreGrid` ONCE, holding `S`, `λ_tot` and `φ` by
reference, and every subsequent `fill_smile_buffers!` writes THROUGH it. The Over/
Under pricer then reaches the smile method with no per-fixture allocation and no
`isa` test — see `06/l03_score_grids.jl` §7.
"""
struct BookWorkspace{G}
    ws::GridWorkspace
    S::Array{Float64, 3}
    grid::G
    λ_tot::Vector{Float64}
    φ::Matrix{Float64}
    slots_1x2::Vector{MarketSlot{Market1X2, 3}}
    slots_btts::Vector{MarketSlot{MarketBTTS, 2}}
    slots_ou::Vector{MarketSlot{MarketOverUnder, 2}}
    slots_fb::Vector{FallbackSlot}
    order::Vector{Tuple{Symbol, Int}}
    mean_buf::Array{Float64, 3}
    max_goals::Int
    n_draws::Int
end

"Total bytes the workspace holds. The number that used to be paid per fixture."
function workspace_bytes(w::BookWorkspace)
    b = sizeof(w.S) + sizeof(w.mean_buf) + 2 * w.max_goals * sizeof(Float64)
    b += sizeof(w.λ_tot) + sizeof(w.φ)
    for s in w.slots_1x2;  b += sum(sizeof, s.book); end
    for s in w.slots_btts; b += sum(sizeof, s.book); end
    for s in w.slots_ou;   b += sum(sizeof, s.book); end
    return b
end

"Markets in this workspace that fall back to `Predictions.compute_market_probs`."
fallback_market_names(w::BookWorkspace) = String[string(s.market) for s in w.slots_fb]

function Base.show(io::IO, w::BookWorkspace)
    n = length(w.slots_1x2) + length(w.slots_btts) + length(w.slots_ou) + length(w.slots_fb)
    @printf(io, "BookWorkspace(%d markets, %d draws, %.1f KiB",
            n, w.n_draws, workspace_bytes(w) / 1024)
    isempty(w.slots_fb) || print(io, ", ", length(w.slots_fb), " fallback")
    print(io, ")")
end


# ==============================================================================
# 6. THE BUILD REPORT
# ==============================================================================
#
# `build_books` returns `nothing` for a fixture it cannot stake and drops it
# (book.jl:151). Five different causes collapse into one absence, and the difference
# between them is the difference between "the odds feed is fine" and "the odds feed
# lost a market last Tuesday". On a real fold that silence is how a data outage gets
# mistaken for a modelling result.
#
# `BuildReport` names each cause and lists the fixtures. It also carries the MCMC
# verdict, which the legacy pipeline has nowhere to put at all.

"""
    BuildReport

Why the builder produced `n_books` books from `n_fixtures` fixtures, and whether the
posterior underneath them converged.

| field | is |
|---|---|
| `skipped_no_fixture` | in the latents, absent from the fixture table |
| `skipped_unplayed` | no result and `require_result = true` |
| `skipped_no_quotes` | no row in the odds frame |
| `skipped_no_selections` | quoted, but no complete market group survived |
| `errored` | `match_id => message`; the legacy builder swallows these silently |
| `fallback_markets` | markets priced through `Predictions`, not `06`'s kernels |
| `converged` | `fit.diagnostics.passed`, or `nothing` when no `Fit` was involved |
| `gated` | did `require_converged` actually admit this build |
"""
struct BuildReport
    n_fixtures::Int
    n_books::Int
    skipped_no_fixture::Vector{Int}
    skipped_unplayed::Vector{Int}
    skipped_no_quotes::Vector{Int}
    skipped_no_selections::Vector{Int}
    errored::Vector{Pair{Int, String}}
    fallback_markets::Vector{String}
    converged::Union{Nothing, Bool}
    failed_gates::Vector{String}
    gated::Bool
    elapsed::Float64
end

BuildReport(n_fixtures::Integer) = BuildReport(
    Int(n_fixtures), 0, Int[], Int[], Int[], Int[], Pair{Int,String}[], String[],
    nothing, String[], false, 0.0)

"Total fixtures the builder declined to stake."
n_skipped(r::BuildReport) = length(r.skipped_no_fixture) + length(r.skipped_unplayed) +
                            length(r.skipped_no_quotes) + length(r.skipped_no_selections) +
                            length(r.errored)

function Base.show(io::IO, r::BuildReport)
    print(io, "BuildReport(", r.n_books, "/", r.n_fixtures, " books")
    r.converged === nothing || print(io, ", MCMC ", r.converged ? "PASS" : "FAIL")
    n_skipped(r) == 0 || print(io, ", ", n_skipped(r), " skipped")
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::BuildReport)
    println(io, "BuildReport")
    @printf(io, "  built            : %d of %d fixtures  (%.2f s)\n",
            r.n_books, r.n_fixtures, r.elapsed)
    if r.converged !== nothing
        @printf(io, "  MCMC convergence : %s%s\n", r.converged ? "PASS" : "FAIL",
                isempty(r.failed_gates) ? "" : "  (failed: " * join(r.failed_gates, ", ") * ")")
        @printf(io, "  gate enforced    : %s\n", r.gated ? "yes" : "no")
    end
    for (label, v) in ("no fixture row" => r.skipped_no_fixture,
                       "unplayed"       => r.skipped_unplayed,
                       "no quotes"      => r.skipped_no_quotes,
                       "no selections"  => r.skipped_no_selections)
        isempty(v) && continue
        @printf(io, "  skipped, %-15s : %d %s\n", label, length(v),
                length(v) <= 6 ? string(v) : string(v[1:6])[1:end-1] * ", …]")
    end
    isempty(r.errored) || @printf(io, "  errored          : %d  (first: %s)\n",
                                  length(r.errored), r.errored[1])
    isempty(r.fallback_markets) ||
        @printf(io, "  fallback pricing : %s\n", join(r.fallback_markets, ", "))
end


# ==============================================================================
# 7. SIMULATION RESULTS
# ==============================================================================
#
# `Portfolio.Trajectory` stores a simulation as six parallel vectors plus a bet
# frame. Everything a path metric needs is in there, but "slate 14" is five indexed
# reads into five different vectors that nothing enforces are the same length, and
# there is no place to put the bankroll BEFORE the slate settled — which is the
# denominator of every per-slate return.
#
# `DailyState` is that row. `Trajectory` is still produced and still returned, because
# `BackTesting.generate_tearsheet` and every existing runner read it.

"""
    DailyState

One settlement window, after it settled.

`bankroll_open` is the bankroll the stakes were sized against and `bankroll_close`
is `bankroll_open * (1 + pnl_frac)`. Both are in the simulation's currency;
`pnl_frac` and `exposure` are fractions of `bankroll_open`, which is what makes a
run at `initial_bankroll = 1_000` and one at `1.0` the same trajectory scaled.
"""
struct DailyState
    idx::Int
    date::Date
    n_fixtures::Int
    n_bets::Int
    bankroll_open::Float64
    bankroll_close::Float64
    stake_frac::Float64
    pnl_frac::Float64
    exposure::Float64
    k_risk::Float64
    capped::Bool
end

"Log growth realised in this window. The quantity Kelly maximises."
log_growth(d::DailyState) = log(1.0 + d.pnl_frac)

function Base.show(io::IO, d::DailyState)
    @printf(io, "DailyState(%s, %d bets, expo %.3f, pl %+.4f, bank %.2f)",
            d.date, d.n_bets, d.exposure, d.pnl_frac, d.bankroll_close)
end

"""
    PortfolioSummary

The metric set the briefing asks for, plus the path statistics `Portfolio.path_metrics`
already produced so nothing is lost by using this instead.

THREE OF THESE NEED THEIR CONVENTION STATED, because a number that means something
slightly different from what a reader assumes is worse than a missing one:

* `cagr` annualises `final/initial` over the CALENDAR span from the first slate to
  the last, `(final/initial)^(365.25/days) - 1`. It is `NaN` when the span is zero —
  a single-slate backtest has no annual rate, and reporting one would be inventing
  eleven months of evidence.
* `sharpe` and `sortino` are computed on per-slate LOG returns, not on the flat
  per-bet P/L, because a slate is the compounding unit. `sharpe_ann` scales by
  `sqrt(slates_per_year)` inferred from the same calendar span; on an eight-fixture
  weekly programme that is `sqrt(52)`, and it is `NaN` when `cagr` is.
* `mdd` follows `Portfolio.path_metrics`: a NEGATIVE PERCENT (`-18.4` is an 18.4%
  drawdown), so the two agree when placed side by side.

`win_rate` is over STAKED SELECTIONS, counting a push (`payoff == 0`) as neither a
win nor a loss but keeping it in the denominator — the stake was committed.
"""
struct PortfolioSummary
    initial_bankroll::Float64
    final_bankroll::Float64
    total_return_pct::Float64
    cagr::Float64
    growth_per_slate::Float64
    roi::Float64
    roi_1x2::Float64
    mdd::Float64
    ulcer::Float64
    calmar::Float64
    martin::Float64
    sharpe::Float64
    sharpe_ann::Float64
    sortino::Float64
    win_rate::Float64
    n_slates::Int
    n_fixtures::Int
    n_bets::Int
    total_stake::Float64
    total_pnl::Float64
    mean_exposure::Float64
    max_exposure::Float64
    worst_slate::Float64
    mean_k_risk::Float64
    n_capped::Int
    span_days::Int
end

"`PortfolioSummary` as a `NamedTuple`, for `DataFrame(rows)` and for `merge`."
as_namedtuple(s::PortfolioSummary) =
    NamedTuple{fieldnames(PortfolioSummary)}(getfield(s, f) for f in fieldnames(PortfolioSummary))

function Base.show(io::IO, s::PortfolioSummary)
    @printf(io, "PortfolioSummary(%.2f → %.2f, %+.2f%%, ROI %+.2f%%, MDD %.2f%%, %d bets)",
            s.initial_bankroll, s.final_bankroll, s.total_return_pct, s.roi, s.mdd, s.n_bets)
end

"""
    BootstrapCI

Percentile intervals from `B` resamples, **clustered by match**.

Resampling individual BETS would understate every interval badly: eleven selections
on one fixture share one scoreline and are strongly dependent, so treating them as
eleven independent observations divides the standard error by roughly `sqrt(11)`.
On the reference ScottishLower book that is the difference between an interval that
excludes zero and one that does not (`src/Portfolio/metrics.jl:47`).

`roi_*` is bit-identical to `Portfolio.bootstrap_roi` at the same `B` and `seed`:
the resampling draws the same indices from the same `MersenneTwister` in the same
order, and the growth statistics are accumulated from those same indices rather than
from a second pass.
"""
struct BootstrapCI
    roi_lo::Float64
    roi_hi::Float64
    roi_sd::Float64
    growth_lo::Float64
    growth_hi::Float64
    growth_sd::Float64
    p_roi_positive::Float64
    B::Int
    seed::Int
end

function Base.show(io::IO, c::BootstrapCI)
    @printf(io, "BootstrapCI(ROI 95%% [%+.2f, %+.2f]%%, P(ROI>0) = %.3f, B = %d)",
            c.roi_lo, c.roi_hi, c.p_roi_positive, c.B)
end

"""
    PortfolioResult

Everything `simulate_portfolio` produced.

| field | is |
|---|---|
| `daily_states` | `Vector{DailyState}`, chronological |
| `summary` | `PortfolioSummary` |
| `metrics` | `NamedTuple` of any `BackTesting.AbstractWealthMetric`s requested |
| `bootstrap_ci` | `BootstrapCI`, or `nothing` when `bootstrap = false` |
| `trajectory` | the legacy `Portfolio.Trajectory`, unchanged, for `generate_tearsheet` |
| `attribution` | per-family stake / P&L / ROI / hit rate |
| `converged` | the MCMC verdict carried through from the build, or `nothing` |

`converged` is carried, not recomputed. A `PortfolioResult` pulled off disk six
months from now answers "should this number be believed" without a `DataStore`,
without the chains and without a re-audit — the same property `07` gives a `Fit`.
"""
struct PortfolioResult{M<:NamedTuple}
    daily_states::Vector{DailyState}
    summary::PortfolioSummary
    metrics::M
    bootstrap_ci::Union{Nothing, BootstrapCI}
    trajectory::Trajectory
    attribution::DataFrame
    converged::Union{Nothing, Bool}
    failed_gates::Vector{String}
end

Base.length(r::PortfolioResult) = length(r.daily_states)
Base.getindex(r::PortfolioResult, i::Integer) = r.daily_states[i]
Base.iterate(r::PortfolioResult, s::Int = 1) =
    s > length(r) ? nothing : (r.daily_states[s], s + 1)

"The daily states as a `DataFrame`, one row per settlement window."
function states_frame(r::PortfolioResult)
    isempty(r.daily_states) && return DataFrame(
        idx = Int[], date = Date[], n_fixtures = Int[], n_bets = Int[],
        bankroll_open = Float64[], bankroll_close = Float64[], stake_frac = Float64[],
        pnl_frac = Float64[], exposure = Float64[], k_risk = Float64[], capped = Bool[])
    return DataFrame(NamedTuple{fieldnames(DailyState)}(getfield(d, f)
                     for f in fieldnames(DailyState)) for d in r.daily_states)
end

function Base.show(io::IO, r::PortfolioResult)
    print(io, "PortfolioResult(", length(r), " slates, ")
    @printf(io, "%.2f → %.2f", r.summary.initial_bankroll, r.summary.final_bankroll)
    r.converged === nothing || print(io, r.converged ? ", converged" : ", UNCONVERGED")
    print(io, ")")
end
