# src/Portfolio/types.jl
#
# Abstract types first (the swappable seams), then the domain objects that flow through the
# pipeline, then the configuration structs.

# ===================================================================
# 1. Abstract types -- the swappable seams
# ===================================================================

"""
    AbstractPricePolicy

How a traded price becomes the price we are settled at.
Contract: `settlement_odds(policy, decimal_odds, group_overround) -> Float64`.
"""
abstract type AbstractPricePolicy end

"""
    AbstractCommissionModel

How exchange commission is charged.
Contract: `net_return(model, decimal_odds) -> Float64` (profit per unit staked on a win).
"""
abstract type AbstractCommissionModel end

"""
    AbstractAllocator

Turns a belief over outcomes plus a payoff matrix into stakes.
Contract: `allocate(alloc, p, R, exec) -> (a, kkt, converged)`.
"""
abstract type AbstractAllocator end

"""
    AbstractShrinkage

Parameter-uncertainty correction applied to the point-estimate allocation.
Contract: `shrink_factor(s, score_matrix, R, p, alloc, exec) -> Float64` in `[0, 1]`.
"""
abstract type AbstractShrinkage end

"""
    AbstractTrustModel

How much we believe the model over the market, per selection.
Contract: `trust_for(model, sel, ctx) -> Float64` in `[0, 1]`.
"""
abstract type AbstractTrustModel end

"""
    AbstractRiskModel

The drawdown budget.
Contract: `risk_factor(model, probs, rets) -> Float64` (one factor for the whole slate) or
`Vector{Float64}` (one per match).
"""
abstract type AbstractRiskModel end

"""
    AbstractExposureCap

Hard bound on stake settled simultaneously. There is deliberately no `NoCap`: a cap `< 1` is
what makes a non-positive bankroll unrepresentable rather than merely asserted against.
Contract: `apply_cap(cap, stakes) -> (stakes, capped::Bool)`.
"""
abstract type AbstractExposureCap end

"""
    AbstractSelectionFilter

Curation. Contract: `keep(filter, sel, stake, ctx) -> Bool`.
"""
abstract type AbstractSelectionFilter end

"""
    AbstractSlateGrouping

How matches are partitioned into simultaneous settlement windows.
Contract: `group(grouping, books) -> Vector{Slate}`.
"""
abstract type AbstractSlateGrouping end

# ===================================================================
# 2. Domain objects
# ===================================================================

"""
    Selection

One priced leg of one match.

`odds_quoted` is the price as traded; `odds_used` is what we are settled at after the
`AbstractPricePolicy` has been applied and is never above `odds_quoted` under `DeArb`.
`p_market` is the vig-removed market probability and is a *forecast benchmark only* -- it is
never used as a price.
"""
struct Selection
    family::String        # "1X2_home", "O/U 2.5_over_25" -- the trust key
    group::String         # "1X2" | "BTTS" | "OverUnder" | ...
    line::Float64
    selection::Symbol
    odds_quoted::Float64
    odds_used::Float64
    p_model::Float64
    p_market::Float64
end

"""
    MatchBook

Everything needed to stake and settle one match, and nothing that depends on a `PolicySpec`.
This is the cache boundary: a `MatchBook` is a pure function of the data and the `BookSpec`.

* `p_grid`  -- posterior-mean score grid, length `N = max_h * max_a`, normalised to 1.
* `R`       -- Jacot return matrix, `N x n`; wealth after the bets is `1 .+ R * a`.
* `settle`  -- realised per-unit payoff of each selection (win / push / lose), or `nothing`
              for a fixture that has not been played. An unsettled book can be STAKED but not
              SIMULATED; `simulate` refuses them.
* `a_kelly` -- allocation on the posterior mean, at full size.
* `k_shrink`-- parameter-uncertainty factor from the `AbstractShrinkage`.
* `kkt`     -- KKT residual of `a_kelly`; should be ~1e-6.
"""
struct MatchBook
    m_id::Int
    date::Date
    sels::Vector{Selection}
    p_grid::Vector{Float64}
    R::Matrix{Float64}
    settle::Union{Nothing,Vector{Float64}}
    a_kelly::Vector{Float64}
    k_shrink::Float64
    kkt::Float64
    converged::Bool
end

function Base.:(==)(a::Selection, b::Selection)
    a.family == b.family && a.group == b.group && a.line == b.line &&
    a.selection == b.selection && a.odds_quoted == b.odds_quoted &&
    a.odds_used == b.odds_used && a.p_model == b.p_model && a.p_market == b.p_market
end

function Base.:(==)(a::MatchBook, b::MatchBook)
    a.m_id == b.m_id && a.date == b.date && a.sels == b.sels &&
    a.p_grid == b.p_grid && a.R == b.R && a.settle == b.settle &&
    a.a_kelly == b.a_kelly && a.k_shrink == b.k_shrink && a.kkt == b.kkt &&
    a.converged == b.converged
end

"A set of matches that settle together and therefore share one bankroll."
struct Slate
    window::Date
    books::Vector{MatchBook}
end

function Base.:(==)(a::Slate, b::Slate)
    a.window == b.window && a.books == b.books
end

"Context handed to trust / risk / filter so bankroll- or time-dependent policies are possible."
struct SlateContext
    idx::Int
    date::Date
    bankroll::Float64
end

"Stakes for one slate plus the risk diagnostics that produced them."
struct SlateAllocation
    stakes::Vector{Vector{Float64}}
    k_risk::Float64
    exposure::Float64
    capped::Bool
end

"""
    Trajectory

The result of a simulation. `bets` carries `:match_id, :date, :family, :stake, :odds, :pnl`
so it can be handed straight to `BackTesting.generate_tearsheet`.
"""
struct Trajectory
    bankroll::Vector{Float64}
    dates::Vector{Date}
    slate_pl::Vector{Float64}
    k_risk::Vector{Float64}
    exposure::Vector{Float64}
    n_capped::Int
    total_stake::Float64
    total_pl::Float64
    bets::DataFrame
end

# ===================================================================
# 3. Configuration
# ===================================================================

"""
    ExecutionConfig

Exchange and sizing constraints. `budget` bounds `sum(a)` within a single match; the
simultaneous bound across a slate is the `AbstractExposureCap` in the `PolicySpec`.
"""
Base.@kwdef struct ExecutionConfig{C<:AbstractCommissionModel}
    commission::C            = PerBetCommission(0.02)
    max_selection_stake::Float64 = 0.50
    min_selection_stake::Float64 = 1e-4
    budget::Float64          = 0.99
    barrier_mu::Float64      = 1e-6
    require_complete_markets::Bool = true
end

"""
    BookSpec

Everything that changes a `MatchBook`. **This is the cache key** -- hash it. Changing any field
here invalidates built books; changing a `PolicySpec` does not.
"""
Base.@kwdef struct BookSpec{P<:AbstractPricePolicy,
                            A<:AbstractAllocator,
                            S<:AbstractShrinkage,
                            E<:ExecutionConfig}
    markets::Data.MarketConfig
    price::P     = DeArb()
    allocator::A = KellyLogUtility()
    shrink::S    = BakerMcHale()
    exec::E      = ExecutionConfig()
end

"""
    PolicySpec

Everything that is a pure post-multiplier on an already-built book. Free to sweep against a
cached `Vector{MatchBook}`.

Note: `risk` is homogeneous of degree 0 in the stakes it is handed, so once the drawdown
constraint binds, `trust` can only reshape the book -- it cannot rescale it. See `stake_slate`.
"""
Base.@kwdef struct PolicySpec{T<:AbstractTrustModel,
                              R<:AbstractRiskModel,
                              C<:AbstractExposureCap,
                              F<:AbstractSelectionFilter,
                              G<:AbstractSlateGrouping}
    trust::T    = FlatTrust(0.25)
    risk::R     = SlateDrawdown(23.0)
    cap::C      = FixedCap(0.25)
    filter::F   = KeepAll()
    grouping::G = DailySlate()
end

"A complete staking system: the book it builds and the policy it stakes with."
struct PortfolioSystem{B<:BookSpec, P<:PolicySpec}
    book::B
    policy::P
end
PortfolioSystem(b::BookSpec) = PortfolioSystem(b, PolicySpec())

# ===================================================================
# 4. The odds index  (graduated from 09_unified_portfolio_framework)
# ===================================================================
#
# WHAT THIS REPLACES. `extract_selections` (book.jl) opens with
#
#     rows = view(odds_df, odds_df.match_id .== match_id, :)
#
# and then, once per market,
#
#     sub = view(rows, (rows.market_name .== grp) .& isapprox.(rows.market_line, line), :)
#
# The first line allocates a `BitVector` over the WHOLE odds frame for every fixture: on a
# 500-fixture fold against a 50,000-row frame that is 500 full scans, 25 million comparisons to
# find ~11 rows each time. The second allocates two more full-length masks per market per fixture.
#
# `OddsIndex` does one pass, stores the five columns this pipeline reads as concretely-typed
# vectors, and hands the builder a `UnitRange` of row positions.
#
# ROW ORDER IS PRESERVED EXACTLY. `view(df, mask, :)` yields rows in ascending original-row order,
# and so does this index: rows are bucketed by match in a stable single pass. That matters because
# `extract_selections` builds a `Dict{Symbol,Float64}` whose LAST write wins on a duplicate quote,
# and because the resulting selection order fixes the column order of the payoff matrix `R` --
# which fixes the starting point handed to the Kelly solver. A different order is a different
# answer in the last bits.

"""
    OddsIndex

The five odds columns this pipeline reads, unboxed once into concrete vectors, plus
`match_id -> row positions`.

| field | is |
|---|---|
| `rows` | `match_id` -> the `UnitRange` of stored positions holding that match's rows |
| the rest | `odds_df`'s columns permuted into match-grouped order, concretely typed |

Built once per fold by [`build_odds_index`](@ref) and shared by every `build_book` call.
`nrow(odds_df)` work, once, in place of `n_fixtures x nrow(odds_df)`.

`odds_close` stores `NaN` in place of `missing`, and the admission test downstream is
`isnan(o) || o <= 1.0` -- NOT `o <= 1.0`, because `NaN <= 1.0` is `false` and a naive port would
admit every missing quote as a valid price.
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

# ===================================================================
# 5. The book workspace
# ===================================================================
#
# WHAT THIS REPLACES. `build_book` does, per fixture:
#
#     score_matrix = Predictions.compute_score_matrix(model, extract_params(model, row))
#     model_probs  = Dict(string(m) => compute_market_probs(score_matrix, m) for m in markets)
#
# `compute_score_matrix` allocates a fresh `(12 x 12 x n_draws)` tensor -- 1.4 MB at 1,200 draws,
# 3.7 MB at 3,200 -- and `extract_params` unboxes a `Vector{Any}` DataFrame row to feed it. A
# 500-fixture fold at 3,200 draws is ~1.8 GB of tensor churn plus ~15,000 dictionary and vector
# allocations, all of it live for microseconds. The garbage collector is the dominant cost of a
# backtest and none of that memory does any work.
#
# HERE: one tensor, one workspace and one destination vector per market outcome for the WHOLE
# fold, overwritten in place per fixture by the score-grid kernels in
# `src/predictions/score_grids/`.
#
# WHY THE SLOTS ARE THREE CONCRETELY-TYPED VECTORS AND NOT ONE `Vector{Any}`.
# `spec.markets.markets` is a `Vector{AbstractMarket}`, so a loop over it dispatches dynamically
# on every iteration. A dynamic call whose return value is a `Union` of tuple types boxes, and a
# zero-allocation claim that depends on escape analysis eliding that box is not a claim worth
# making. Three homogeneous vectors give three statically-dispatched loops and `@allocated == 0`
# that is true by construction rather than by optimisation.

"""
    MarketSlot{M, N}

One market's pricing destination: the market itself, its group/line (hoisted out of the hot
loop), its `N` outcome symbols, and the `N` preallocated draw vectors `price_market!` writes into.
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

A market the score-grid kernels have no `price_market!` method for -- the Asian-handicap ladder,
correct score, double chance, draw-no-bet.

These are priced through `Predictions.compute_market_probs` on a `ScoreMatrix` VIEW of the shared
grid, so they still avoid the per-fixture tensor, but they allocate a `Dict` and `N` vectors per
fixture like the legacy path does. Counted in `BuildReport.fallback_markets` and warned about once
per workspace so the cost is never silent.
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

Allocate **one per worker**, not one per fixture -- that inversion is the entire point.
Reused across every fixture of a fold:

| field | is | bytes |
|---|---|---|
| `S` | the shared `(max_goals^2 x n_draws)` score grid | `144 * n_draws * 8` |
| `ws` | the `GridWorkspace` -- two marginal PMF buffers | `2 * max_goals * 8` |
| `grid` | what `price_market!` reads: `S`, or a `SmileScoreGrid` VIEWING `S` | 0 |
| `slots_*` | one destination vector per market outcome | `n_out * n_draws * 8` |
| `mean_buf` | destination for `mean(S, dims = 3)` | `144 * 8` |

`grid` is parametric so the smile route is a compile-time fact rather than a branch: a
`SmileLatents` builds a `SmileScoreGrid` ONCE, holding `S`, `lambda_tot` and `phi` by reference,
and every subsequent `fill_smile_buffers!` writes THROUGH it. The Over/Under pricer then reaches
the smile method with no per-fixture allocation and no `isa` test.
"""
struct BookWorkspace{G}
    ws::Predictions.GridWorkspace
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

# ===================================================================
# 6. The build report
# ===================================================================
#
# `build_books` returns `nothing` for a fixture it cannot stake and drops it. Five different
# causes collapse into one absence, and the difference between them is the difference between
# "the odds feed is fine" and "the odds feed lost a market last Tuesday". On a real fold that
# silence is how a data outage gets mistaken for a modelling result.

"""
    BuildReport

Why the builder produced `n_books` books from `n_fixtures` fixtures, and whether the posterior
underneath them converged.

| field | is |
|---|---|
| `skipped_no_fixture` | in the latents, absent from the fixture table |
| `skipped_unplayed` | no result and `require_result = true` |
| `skipped_no_quotes` | no row in the odds frame |
| `skipped_no_selections` | quoted, but no complete market group survived |
| `errored` | `match_id => message`; the legacy builder swallows these silently |
| `fallback_markets` | markets priced through `Predictions`, not the typed kernels |
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
                length(v) <= 6 ? string(v) : string(v[1:6])[1:end-1] * ", ...]")
    end
    isempty(r.errored) || @printf(io, "  errored          : %d  (first: %s)\n",
                                  length(r.errored), r.errored[1])
    isempty(r.fallback_markets) ||
        @printf(io, "  fallback pricing : %s\n", join(r.fallback_markets, ", "))
end

# ===================================================================
# 7. Simulation results
# ===================================================================
#
# `Trajectory` stores a simulation as six parallel vectors plus a bet frame. Everything a path
# metric needs is in there, but "slate 14" is five indexed reads into five different vectors that
# nothing enforces are the same length, and there is no place to put the bankroll BEFORE the slate
# settled -- which is the denominator of every per-slate return.
#
# `DailyState` is that row. `Trajectory` is still produced and still returned, because
# `BackTesting.generate_tearsheet` and every existing runner read it.

"""
    DailyState

One settlement window, after it settled.

`bankroll_open` is the bankroll the stakes were sized against and `bankroll_close` is
`bankroll_open * (1 + pnl_frac)`. Both are in the simulation's currency; `pnl_frac` and `exposure`
are fractions of `bankroll_open`, which is what makes a run at `initial_bankroll = 1_000` and one
at `1.0` the same trajectory scaled.
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

The headline metric set, plus every path statistic `path_metrics` already produced so nothing is
lost by using this instead.

THREE OF THESE NEED THEIR CONVENTION STATED, because a number that means something slightly
different from what a reader assumes is worse than a missing one:

* `cagr` annualises `final/initial` over the CALENDAR span from the first slate to the last,
  `(final/initial)^(365.25/days) - 1`. It is `NaN` when the span is zero -- a single-slate
  backtest has no annual rate, and reporting one would be inventing eleven months of evidence.
* `sharpe` and `sortino` are computed on per-slate LOG returns, not on flat per-bet P/L, because
  a slate is the compounding unit. `sharpe_ann` scales by `sqrt(slates_per_year)` inferred from
  the same calendar span, and is `NaN` when `cagr` is.
* `mdd` follows `path_metrics`: a NEGATIVE PERCENT (`-18.4` is an 18.4% drawdown), so the two
  agree when placed side by side.

`win_rate` is over STAKED SELECTIONS, counting a push (`payoff == 0`) as neither a win nor a loss
but keeping it in the denominator -- the stake was committed.
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
    @printf(io, "PortfolioSummary(%.2f -> %.2f, %+.2f%%, ROI %+.2f%%, MDD %.2f%%, %d bets)",
            s.initial_bankroll, s.final_bankroll, s.total_return_pct, s.roi, s.mdd, s.n_bets)
end

"""
    BootstrapCI

Percentile intervals from `B` resamples, **clustered by match**.

Resampling individual BETS would understate every interval badly: eleven selections on one fixture
share one scoreline and are strongly dependent, so treating them as eleven independent
observations divides the standard error by roughly `sqrt(11)`. On the reference ScottishLower book
that is the difference between an interval that excludes zero and one that does not.

`roi_*` is bit-identical to `bootstrap_roi` at the same `B` and `seed`: the resampling draws the
same indices from the same `MersenneTwister` in the same order.
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
| `trajectory` | the legacy `Trajectory`, unchanged, for `generate_tearsheet` |
| `attribution` | per-family stake / P&L / ROI / hit rate |
| `converged` | the MCMC verdict carried through from the build, or `nothing` |

`converged` is carried, not recomputed. A `PortfolioResult` pulled off disk six months from now
answers "should this number be believed" without a `DataStore`, without the chains and without a
re-audit -- the same property a `Fit` has.
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

function Base.show(io::IO, r::PortfolioResult)
    print(io, "PortfolioResult(", length(r), " slates, ")
    @printf(io, "%.2f -> %.2f", r.summary.initial_bankroll, r.summary.final_bankroll)
    r.converged === nothing || print(io, r.converged ? ", converged" : ", UNCONVERGED")
    print(io, ")")
end
