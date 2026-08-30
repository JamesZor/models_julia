# ==============================================================================
# 08 — UNIFIED EVALUATION FRAMEWORK : THE TYPE HIERARCHY
# ==============================================================================
#
# The bottom of the include chain. Loads `07_unified_inference_framework` (and,
# through it, `06_typed_posterior_latents` and `05_composable_count_builder`), then
# declares every scoring rule, every result container and the scorecard.
#
# NOTHING IN THIS FILE COMPUTES A NUMBER. The kernels are `l02_scoring_rules.jl`.
#
# ------------------------------------------------------------------------------
# WHY THE TRIGGERS CARRY *MARKETS* AND NOT ONLY *SELECTIONS*
# ------------------------------------------------------------------------------
#
# `src/evaluation/` names a metric's scope with a `Vector{Symbol}` of SELECTIONS —
# `LogLoss([:over_25])`. That is a POST-HOC FILTER: the pipeline prices every market
# in `DEFAULT_MARKET_CONFIG` (five scalar markets, eleven Over/Under lines and the
# whole Asian-handicap ladder — 40-odd markets), materialises a PPD row per fixture
# per selection, joins the lot against `ds.odds`, and only then throws away the
# 39 markets the metric never wanted.
#
# On a 900-fixture fold at 1,200 draws that is ~90 selections × 900 × 1,200 × 8 bytes
# ≈ 780 MB of posterior vectors built to answer a question about three of them.
#
# So the triggers here carry `markets::Vector{<:AbstractMarket}` — what to PRICE — and
# keep `selections::Vector{Symbol}` — what to SCORE. The default for `selections` is
# empty, meaning "every outcome of every market in `markets`", which is exactly the
# legacy meaning of an empty filter. A legacy construction `LogLoss([:over_25])` still
# works and now additionally tells the pricer that `MarketOverUnder(2.5)` is the only
# market it needs to build, because a selection symbol determines its market uniquely
# (§2.1).
#
# ------------------------------------------------------------------------------
# WHY THE RESULT STRUCTS ARE THE LEGACY ONES, FIELD FOR FIELD
# ------------------------------------------------------------------------------
#
# `to_dataframe_row` flattens a result by walking `propertynames` and joining the path
# with underscores (`src/evaluation/translator.jl:41-56`). The COLUMN NAMES of every
# leaderboard, every saved evaluation CSV and every `display_summary_metric` column
# list are therefore a direct function of these structs' field names and order.
#
# Renaming `LogLossComponent.diff_ll`, or reordering `LPDComponent`, silently renames
# or reorders columns in files people have already written. So the containers below are
# the legacy ones — same names, same fields, same order — and everything new is either
# an added TRIGGER field (which does not reach the column names) or a suffix that only
# appears when a non-default option is used (§4.2).
#
# ==============================================================================

# `import`, not `using`: this framework must be loadable into a module that also binds
# the name `Evaluation` (§ the header of `l04_compat_bridge.jl`), and `using
# BayesianFootball` would import that name here and make the `Legacy` submodule's
# binding an error. Every `src` reference below is explicitly qualified for that reason.
import BayesianFootball

# The inference framework, and through it the typed latents and the count builder.
# One include loads all four prototypes.
include(joinpath(@__DIR__, "..", "07_unified_inference_framework", "l06_compat_bridge.jl"))

using DataFrames
using Dates
using Distributions
using MCMCChains
using Printf
using Statistics
using StatsBase: skewness, kurtosis

using .UnifiedInference

# Everything this framework reads out of `07` / `06` / `05`. Named explicitly rather
# than pulled in with a bare `using`, because a typo in one of these is a `MethodError`
# on the first fixture of a real fold and an `UndefVarError` here, at load time.
using .UnifiedInference:
    # 07 — the run container and its audit
    Fit, FoldFit, FitConfig, FitMetadata, ConvergenceSummary, ConvergenceGates,
    audit_convergence, fit_name, chains, total_draws, format_elapsed,
    upgrade_to_fit, LatentStates, as_latent_states,
    # 06 — the typed posterior containers
    AbstractPosteriorLatents, CountLatents, RecombLatents, SmileLatents,
    n_matches, n_draws, latent_match_ids, latent_matrices, match_index,
    observation_family, latent_bytes, latent_allocations,
    recomb_total_home, recomb_total_away, n_strikes, smile_intensity,
    # 06 — the zero-allocation kernels
    GridWorkspace, alloc_score_grid, compute_score_grid!, compute_score_grid,
    price_market!, price_market, alloc_market_book, market_keys, market_arity,
    SmileScoreGrid, alloc_smile_buffers, fill_smile_buffers!, TPL_MAX_GOALS,
    # 06 — extraction and the legacy bridges
    extract_latents, latent_family, to_legacy_dataframe,
    latents_from_legacy_dataframe, tpl_ordered_ids,
    # 06 — the parity scaffolding and the deterministic fixtures l05 reuses
    ulp_distance, ParityRow, tpl_compare, tpl_parity_table, tpl_dataframe_bytes,
    tpl_synthetic_site, tpl_synthetic_chain, tpl_synthetic_fixtures, tpl_team_map,
    tpl_feature_set, tpl_legacy_latents_df,
    # 05 — the composable count builder the runner assembles its two engines from
    CountModelBuilder, add!, build, cb_chain_columns, WealthCovariate,
    PoissonObservation, NegativeBinomialObservation,
    # 07 — the lifecycle entry points the runner drives
    ReplaySampler, fit_model, run_folds, save_fit, load_fit,
    save_latents, load_latents,
    SequentialExecution, ThreadedExecution, QueuedExecution, AutoExecution

const UE_BF   = BayesianFootball
const UE_D    = BayesianFootball.Data
const UE_TI   = BayesianFootball.TypesInterfaces
const UE_PG   = BayesianFootball.Models.PreGame
const UE_Pred = BayesianFootball.Predictions
const UE_Eval = BayesianFootball.Evaluation

using BayesianFootball.Data: AbstractMarket, Market1X2, MarketBTTS, MarketOverUnder,
                             market_group, market_line, outcomes


# ==============================================================================
# 1. THE ABSTRACT CONTRACT
# ==============================================================================

"""
    AbstractScoringRule

A metric TRIGGER: what to score, over which markets, with which options. Carries no
data and no result — it is the dispatch key `compute_metric` resolves on.

Every concrete rule must supply:

  * `compute_metric(rule, latents::AbstractPosteriorLatents, odds, matches)` in
    `l02_scoring_rules.jl` — the kernel;
  * `get_metric_method_name(rule) -> String` — the leaderboard column prefix;
  * `scored_markets(rule) -> Vector{<:AbstractMarket}` — what the pricer must build,
    or `AbstractMarket[]` for a rule that reads goal counts rather than prices.

The name is the repository's own (`src/evaluation/types.jl:7`) and the hierarchy is
deliberately re-declared rather than extended: this prototype must be loadable
alongside `BayesianFootball.Evaluation` without either shadowing the other, and
`l04_compat_bridge.jl` §6 is where the two are reconciled.
"""
abstract type AbstractScoringRule end

"""
    AbstractEvaluationResult

One metric's answer for one fit. Flattened into a DataFrame row by `to_dataframe_row`,
which walks `propertynames` recursively — so every field is either a `Real` or an
`AbstractMetricComponent`, and nothing else.
"""
abstract type AbstractEvaluationResult end

"""
    AbstractMetricComponent

A nested group of scalars inside a result (`LogLossComponent`, `DistributionStats`, …).
The recursion step of the flattener.
"""
abstract type AbstractMetricComponent end

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
scored_selections(::AbstractScoringRule) = Symbol[]

"""
    needs_outcomes(rule) -> Bool

Whether the kernel reads realised `(home_score, away_score)` from `ds.matches`. Lets
`evaluate_fits` skip building the outcome index for a batch that does not want it.
"""
needs_outcomes(::AbstractScoringRule) = false

"Whether the kernel needs the full posterior draw vector, or only its mean."
needs_draws(::AbstractScoringRule) = true


# ==============================================================================
# 2. SELECTION ↔ MARKET
# ==============================================================================
#
# The bridge that lets a legacy `LogLoss([:over_25])` tell the pricer it needs exactly
# `MarketOverUnder(2.5)` and nothing else.
#
# This is well-defined because `Data.outcomes` mints a DISTINCT symbol per market per
# line: `Market1X2()` owns `:home/:draw/:away`, `MarketBTTS()` owns
# `:btts_yes/:btts_no`, and `MarketOverUnder(L)` owns `:over_<L>/:under_<L>` with the
# decimal point deleted (`over_under.jl:12-13`). No two markets in the standard config
# share a selection symbol, so the map is a genuine inverse rather than a guess.

"""
    _ue_parse_ou_line(digits) -> Float64

Undo `replace(string(line), "." => "")`. The engine that produced the symbol always
emits a single fractional digit (`0.5`, `2.5`, `10.5`), so the last character is the
tenths place: `"25" → 2.5`, `"105" → 10.5`, `"05" → 0.5`.
"""
function _ue_parse_ou_line(digits::AbstractString)
    length(digits) >= 2 || return nothing
    all(isdigit, digits) || return nothing
    whole = digits[1:(end - 1)]
    frac  = digits[end]
    return parse(Float64, whole * "." * frac)
end

"""
    market_for_selection(sel::Symbol) -> AbstractMarket or nothing

The market that owns `sel`, or `nothing` if no market in the standard three families
does. `nothing` is not an error here — `selections_to_markets` decides what to do with
it, because a caller filtering on a Double-Chance or Asian-handicap selection is asking
for something this framework's kernels do not price, and the message belongs there.
"""
function market_for_selection(sel::Symbol)
    sel in (:home, :draw, :away)     && return Market1X2()
    sel in (:btts_yes, :btts_no)     && return MarketBTTS()
    s = String(sel)
    for prefix in ("over_", "under_")
        if startswith(s, prefix)
            line = _ue_parse_ou_line(s[(length(prefix) + 1):end])
            line === nothing || return MarketOverUnder(line)
        end
    end
    return nothing
end

"""
    selections_to_markets(sels) -> Vector{AbstractMarket}

The deduplicated markets a selection filter implies, in first-appearance order.

Throws on a selection this framework cannot price, naming it. The alternative — pricing
what it can and silently returning a metric computed over fewer rows than the caller
asked for — is the exact class of quiet-wrong-answer this prototype exists to remove.
"""
function selections_to_markets(sels)
    out = AbstractMarket[]
    for s in sels
        m = market_for_selection(Symbol(s))
        m === nothing && error(
            "selection :$(s) belongs to no market this framework prices. Supported " *
            "families: 1X2 (:home/:draw/:away), BTTS (:btts_yes/:btts_no) and " *
            "Over/Under (:over_25, :under_25, …). Pass `markets = [...]` explicitly " *
            "for anything else.")
        any(x -> x == m, out) || push!(out, m)
    end
    return out
end

"""
    market_selections(markets) -> Vector{Symbol}

Every outcome symbol the given markets own, in `market_keys` order.
"""
market_selections(markets) = Symbol[s for m in markets for s in market_keys(m)]

"""
    DEFAULT_SCORED_MARKETS

The three markets the briefing names: 1X2, Over/Under 2.5 and BTTS.

Not `Data.DEFAULT_MARKET_CONFIG`, which is 40-odd markets. A default that prices the
Asian-handicap ladder to report a 1X2 log-loss is how `src/evaluation/` came to spend
most of its time in `model_inference`.
"""
const DEFAULT_SCORED_MARKETS = AbstractMarket[Market1X2(), MarketOverUnder(2.5), MarketBTTS()]

"""
    _ue_resolve_scope(markets, selections) -> (Vector{AbstractMarket}, Vector{Symbol})

The one place the two ways of naming a scope are reconciled, so every trigger's
constructor agrees:

  * both given            → used as-is (the caller is being explicit);
  * selections only       → markets derived from them (the LEGACY construction);
  * markets only          → selections left empty, meaning "all outcomes of these";
  * neither               → `DEFAULT_SCORED_MARKETS`, no filter.
"""
function _ue_resolve_scope(markets, selections)
    sels = Symbol[Symbol(s) for s in selections]
    if markets === nothing
        mkts = isempty(sels) ? copy(DEFAULT_SCORED_MARKETS) : selections_to_markets(sels)
    else
        mkts = AbstractMarket[m for m in markets]
        isempty(mkts) && error("markets must not be empty — nothing would be priced.")
    end
    return mkts, sels
end


# ==============================================================================
# 3. THE TRIGGERS
# ==============================================================================
#
# Each rule accepts BOTH shapes: the briefing's (`markets = …`) and the legacy
# positional one (`LogLoss(:over_25)`, `LogLoss([:over_25, :under_25])`). The legacy
# forms are what every runner in this repository currently writes, and a prototype that
# needed them rewritten would not be a compatibility bridge.

# --- 3.1 LogLoss ---------------------------------------------------------------

"""
    LogLoss(; markets = DEFAULT_SCORED_MARKETS, selections = Symbol[])
    LogLoss(selections::Vector{Symbol})      # legacy
    LogLoss(selection::Symbol)               # legacy
    LogLoss(markets::Vector{<:AbstractMarket})

Binary cross-entropy of the model's mean market probability against the realised
outcome, alongside the same quantity for the market's own vig-free closing price.

    LL = − ( y·log p̂ + (1−y)·log(1−p̂) ),   p̂ clamped to [1e-15, 1−1e-15]

reported as `model_ll`, `market_ll` and `diff_ll = model − market`. NEGATIVE `diff_ll`
means the model beat the closing line.

THE BRIEFING SAYS "MULTI-CLASS CROSS-ENTROPY", AND THIS IS NOT THAT — deliberately.
`−Σᵢ yᵢ log pᵢ` over the three 1X2 outcomes is a different number from the mean of the
three binary terms, and `src/evaluation/metrics_methods/logloss.jl:49` computes the
binary form. Three reasons the binary form stays:

  1. It is what every leaderboard in `data/` was written with. Switching would make
     new numbers silently incomparable with old ones.
  2. It generalises. Over/Under and BTTS are two-outcome markets and Asian handicaps
     are priced per side; a multi-class form needs a partition of outcomes that the
     odds table does not carry.
  3. It keeps the market baseline meaningful — `market_ll` is the same functional
     applied to `prob_fair_close`, so `diff_ll` is a like-for-like comparison.
"""
struct LogLoss <: AbstractScoringRule
    markets::Vector{AbstractMarket}
    selections::Vector{Symbol}
end

function LogLoss(; markets = nothing, selections = Symbol[])
    m, s = _ue_resolve_scope(markets, selections)
    return LogLoss(m, s)
end
LogLoss(selections::AbstractVector{Symbol}) = LogLoss(; selections = selections)
LogLoss(selection::Symbol) = LogLoss(; selections = [selection])
LogLoss(markets::AbstractVector{<:AbstractMarket}) = LogLoss(; markets = markets)

scored_markets(m::LogLoss)    = m.markets
scored_selections(m::LogLoss) = m.selections
needs_draws(::LogLoss)        = false

# --- 3.2 LPD -------------------------------------------------------------------

"""
    LPD(; markets = DEFAULT_SCORED_MARKETS, selections = Symbol[], target = :market)
    LPD(selections::Vector{Symbol})      # legacy
    LPD(selection::Symbol)               # legacy

Log posterior predictive density — the log of the POSTERIOR MEAN likelihood, not the
likelihood at the posterior mean:

    LPD_i = log( (1/S) Σ_s p(y_i | θ^(s)) )

evaluated by log-sum-exp so a fold with tiny probabilities does not underflow.

TWO TARGETS, because the briefing and `src` mean different things by "LPD":

| `target`  | `y_i` is                        | baseline                                |
|-----------|---------------------------------|-----------------------------------------|
| `:market` | a binary market outcome         | `log p_fair_close` — the closing line   |
| `:score`  | the realised `(g_h, g_a)` pair  | none; `market_lpd`/`diff_lpd` are `NaN` |

`:market` is `src/evaluation/metrics_methods/lpd.jl` and is the default, so every
existing caller gets the number it already has. `:score` is the briefing's
`log((1/S) Σ_s P(G_h = g_h, G_a = g_a | θ^s))`, read straight off the score grid — a
strictly sharper test of the joint, and the one that can distinguish two models that
price 1X2 identically out of different score distributions.

There is no market baseline for `:score` and none is invented: the odds table carries
no full correct-score distribution for these leagues, so `diff_lpd` would be a
comparison against a number nobody quoted.
"""
struct LPD <: AbstractScoringRule
    markets::Vector{AbstractMarket}
    selections::Vector{Symbol}
    target::Symbol
end

function LPD(; markets = nothing, selections = Symbol[], target::Symbol = :market)
    target in (:market, :score) ||
        error("LPD: target must be :market or :score, got :$target.")
    m, s = _ue_resolve_scope(markets, selections)
    return LPD(m, s, target)
end
LPD(selections::AbstractVector{Symbol}) = LPD(; selections = selections)
LPD(selection::Symbol) = LPD(; selections = [selection])
LPD(markets::AbstractVector{<:AbstractMarket}) = LPD(; markets = markets)

scored_markets(m::LPD)    = m.target === :market ? m.markets : AbstractMarket[]
scored_selections(m::LPD) = m.selections
needs_outcomes(m::LPD)    = m.target === :score

# --- 3.3 CRPS ------------------------------------------------------------------

"""
    CRPS(; max_goals = 30)

Continuous Ranked Probability Score of each side's goal count against its realised
value, on the discrete form

    CRPS = Σ_{x=0}^{max_goals} ( F(x) − 1{x ≥ y} )²

reported for home, away, and their per-match average.

PLUG-IN, NOT POSTERIOR-PREDICTIVE, and that is inherited rather than chosen. `F` is the
CDF of ONE marginal built from the POSTERIOR MEAN λ (and mean r), exactly as
`src/evaluation/metrics_methods/crps.jl:88` does — not the posterior-averaged CDF
`(1/S) Σ_s F_s(x)`, which is the quantity a Bayesian CRPS would use and which is
strictly better calibrated. Changing it here would make every CRPS number in this
repository's history incomparable with every new one for a reason unrelated to the
container swap this prototype is about. Recorded in `README.md` as a live defect in
`src`, not silently fixed.

`max_goals = 30` is the `src` default. The tail beyond 30 goals contributes
`(1 − 1)² = 0` for any realised score, so the truncation is exact for real football and
the parameter exists only for other sports.
"""
Base.@kwdef struct CRPS <: AbstractScoringRule
    max_goals::Int = 30
end

needs_outcomes(::CRPS) = true
needs_draws(::CRPS)    = false

# --- 3.4 RQR -------------------------------------------------------------------

"""
    RQR(; n_sims = 1000, seed = 42)

Randomized Quantile Residuals — Dunn & Smyth's device for giving a DISCRETE
distribution a continuous residual:

    u ~ Uniform( F(y−1), F(y) ),        r = Φ⁻¹(u)

If the marginal is correctly specified, `r` is standard normal. The reported
`DistributionStats` — mean, sd, skewness, excess kurtosis, and the Shapiro-Wilk `W`
and `p` — are the test of that.

THE MARGINAL IS CHOSEN BY DISPATCH ON THE LATENT CONTAINER, not by a `hasproperty`
probe of a DataFrame:

| container                           | marginal                                    |
|-------------------------------------|---------------------------------------------|
| `CountLatents{T, Nothing}`          | `Poisson(λ̄)`                                |
| `CountLatents{T, <:NamedTuple}`     | `NegativeBinomial(r̄, r̄/(r̄+λ̄))`              |
| `RecombLatents{T}`                  | `Poisson(λ̄_open + λ̄_pen + λ̄_og)`            |
| `SmileLatents{T, Obs}`              | as the count case, on the grid intensities  |

`src` does the same selection with `hasproperty(df, :r)` / `hasproperty(df, :r_h)` and
an `Inf` sentinel meaning "Poisson" (`rqr.jl:58-68`); a container that carries neither
column reaches the Poisson branch by falling off the end of an `if`. Here a container
with no dispersion cannot reach the negative-binomial method at all.

`n_sims` IS RANDOMISATION REPLICATES, AND IT AVERAGES SUMMARIES, NOT RESIDUALS. One
draw of `u` per observation gives one valid RQR sample and a noisy Shapiro-Wilk `p`.
Averaging the RESIDUALS across replicates would be wrong — it shrinks them toward the
mid-quantile normal score and manufactures normality — so each replicate is summarised
in full and the SUMMARIES are averaged. `n_sims = 1` reproduces `src` draw for draw,
which is what `l05_parity.jl` uses.

`seed` makes the whole thing reproducible. `src` calls `rand` on the global RNG with no
seed, so its RQR table is different on every run — a diagnostic that cannot be
re-checked. See `README.md`.
"""
Base.@kwdef struct RQR <: AbstractScoringRule
    n_sims::Int = 1000
    seed::Int = 42
end

needs_outcomes(::RQR) = true
needs_draws(::RQR)    = false

# --- 3.5 GLMEdge ---------------------------------------------------------------

"""
    GLMEdge(; target_selection = :all, min_edge = 0.0, markets = …, selections = …)
    GLMEdge(selections::Vector{Symbol})      # legacy
    GLMEdge(selection::Symbol)               # legacy

Logistic regression of the realised outcome on the market's fair probability and the
model's EDGE over it:

    logit P(Y = 1) = β₀ + β₁·p_fair_close + β₂·(p̂_model − p_fair_close)

`β₂` — reported as `spread_fair` — is the question: does the model's disagreement with
the closing line predict which way the result goes? `β₂ > 0` with a small p-value is
the signature of a model that knows something the market does not. `β₁ ≈ 1` and
`β₀ ≈ 0` say the closing line itself is calibrated, which is the null this is measured
against.

`target_selection` is the briefing's spelling of the legacy `selections` filter:
`:all` means no filter, anything else is a one-selection filter. `min_edge` drops rows
with `|p̂ − p_fair| < min_edge` — a way to ask whether the model's *confident*
disagreements are the informative ones. `0.0` keeps every row, which is the legacy
behaviour and the parity setting.
"""
struct GLMEdge <: AbstractScoringRule
    markets::Vector{AbstractMarket}
    selections::Vector{Symbol}
    min_edge::Float64
end

function GLMEdge(; target_selection::Symbol = :all, min_edge::Real = 0.0,
                   markets = nothing, selections = Symbol[])
    sels = Symbol[Symbol(s) for s in selections]
    if target_selection !== :all && isempty(sels)
        sels = [target_selection]
    end
    m, s = _ue_resolve_scope(markets, sels)
    min_edge >= 0 || error("GLMEdge: min_edge must be non-negative, got $min_edge.")
    return GLMEdge(m, s, Float64(min_edge))
end
GLMEdge(selections::AbstractVector{Symbol}) = GLMEdge(; selections = selections)
GLMEdge(selection::Symbol) = GLMEdge(; selections = [selection])
GLMEdge(markets::AbstractVector{<:AbstractMarket}) = GLMEdge(; markets = markets)

scored_markets(m::GLMEdge)    = m.markets
scored_selections(m::GLMEdge) = m.selections
needs_draws(::GLMEdge)        = false

# --- 3.6 MIQ -------------------------------------------------------------------

"""
    MIQ(; markets = MIQ_DEFAULT_MARKETS)

Market-Implied Quantile. For each priced selection, where does the market's fair
probability sit inside the model's POSTERIOR distribution of that probability?

    q_i = (1/S) · #{ s : p̂ᵢ^(s) ≤ p_fair,i }

A model that is systematically more bullish than the market puts `q` near 1. The
DIAGNOSTIC is the gap between the `q` distribution of the selections that WON and of
those that LOST: if the model has an edge, the market underprices winners, so winners
carry a lower `q` than losers and `mean_gap = mean(q | lose) − mean(q | win)` is
positive. The two-sample Kolmogorov-Smirnov statistic tests that the two `q`
distributions differ at all.

Unlike every other rule here, MIQ reads the FULL posterior of the price rather than its
mean — collapsing to a point probability would destroy the quantity being measured.

`MIQ_DEFAULT_MARKETS` covers the twelve selections `MIQResult` reports: 1X2, BTTS and
the three central Over/Under lines.
"""
const MIQ_DEFAULT_MARKETS = AbstractMarket[
    Market1X2(), MarketBTTS(),
    MarketOverUnder(1.5), MarketOverUnder(2.5), MarketOverUnder(3.5),
]

struct MIQ <: AbstractScoringRule
    markets::Vector{AbstractMarket}
end

MIQ(; markets = nothing) =
    MIQ(markets === nothing ? copy(MIQ_DEFAULT_MARKETS) : AbstractMarket[m for m in markets])
MIQ(markets::AbstractVector{<:AbstractMarket}) = MIQ(; markets = markets)

scored_markets(m::MIQ) = m.markets


# ==============================================================================
# 4. THE RESULT CONTAINERS
# ==============================================================================
#
# Legacy field names and legacy field ORDER. See the file header for why.

# --- 4.1 the containers --------------------------------------------------------

"""
    LogLossComponent(model_ll, market_ll, diff_ll, n_obs)

`diff_ll = model_ll − market_ll`. NEGATIVE means the model beat the closing line.
"""
struct LogLossComponent <: AbstractMetricComponent
    model_ll::Float64
    market_ll::Float64
    diff_ll::Float64
    n_obs::Int
end

struct LogLossResult <: AbstractEvaluationResult
    overall::LogLossComponent
end

"""
    LPDComponent(model_lpd, model_std, model_skewness, model_kurtosis,
                 market_lpd, diff_lpd, elpd, n_obs)

`diff_lpd = model − market`; POSITIVE means the model beat the line (LPD is a
utility, log-loss is a cost — the signs are opposite and both match `src`).
`elpd` is the SUM, the quantity that goes into a model-comparison table.

For `LPD(target = :score)` the market fields are `NaN`: there is no quoted
correct-score distribution to compare against (§3.2).
"""
struct LPDComponent <: AbstractMetricComponent
    model_lpd::Float64
    model_std::Float64
    model_skewness::Float64
    model_kurtosis::Float64
    market_lpd::Float64
    diff_lpd::Float64
    elpd::Float64
    n_obs::Int
end

struct LPDResult <: AbstractEvaluationResult
    overall::LPDComponent
end

struct CRPSComponent <: AbstractMetricComponent
    mean::Float64
end

"""
    CRPSResults(home, away, all)

`all` is the per-match AVERAGE of the home and away scores, not their pool — the
convention `src/evaluation/metrics_methods/crps.jl:92` uses, so a match contributes one
number rather than two.
"""
struct CRPSResults <: AbstractEvaluationResult
    home::CRPSComponent
    away::CRPSComponent
    all::CRPSComponent
end

"The briefing's spelling. `src` names this type `CRPSResults`; both work."
const CRPSResult = CRPSResults

"""
    DistributionStats(mean, std, skewness, kurtosis, shapiro_w, shapiro_p)

`kurtosis` is EXCESS kurtosis (`StatsBase` convention): 0 for a normal.
"""
struct DistributionStats <: AbstractMetricComponent
    mean::Float64
    std::Float64
    skewness::Float64
    kurtosis::Float64
    shapiro_w::Float64
    shapiro_p::Float64
end

"""
    RQRResult(home, away, all)

`all` POOLS the home and away residuals (`vcat`), unlike `CRPSResults.all` which
averages them. Both conventions are `src`'s and they differ; preserved rather than
harmonised, because harmonising would change published numbers.
"""
struct RQRResult <: AbstractEvaluationResult
    home::DistributionStats
    away::DistributionStats
    all::DistributionStats
end

struct GLMCoefComponent <: AbstractMetricComponent
    coef::Float64
    std_error::Float64
    z_score::Float64
    p_value::Float64
end

struct GLMEdgeResult <: AbstractEvaluationResult
    intercept::GLMCoefComponent
    prob_fair::GLMCoefComponent
    spread_fair::GLMCoefComponent
    n_obs::Int
end

"""
    MIQStats(mean, std, mean_gap, ks_d_stat, p_value, n_winners, n_losers)

`missing` throughout when a group has fewer than two winners or two losers — a
two-sample test on one observation is not a number worth reporting.
"""
struct MIQStats <: AbstractMetricComponent
    mean::Union{Missing, Float64}
    std::Union{Missing, Float64}
    mean_gap::Union{Missing, Float64}
    ks_d_stat::Union{Missing, Float64}
    p_value::Union{Missing, Float64}
    n_winners::Int
    n_losers::Int
end

struct MIQResult <: AbstractEvaluationResult
    all::MIQStats
    home::MIQStats
    draw::MIQStats
    away::MIQStats
    over_15::MIQStats
    under_15::MIQStats
    over_25::MIQStats
    under_25::MIQStats
    over_35::MIQStats
    under_35::MIQStats
    btts_yes::MIQStats
    btts_no::MIQStats
end

"The selections `MIQResult` reports, in field order. `:all` is the pool, not a selection."
const MIQ_FIELD_SELECTIONS =
    (:home, :draw, :away, :over_15, :under_15, :over_25, :under_25,
     :over_35, :under_35, :btts_yes, :btts_no)

# --- 4.2 names and suffixes ----------------------------------------------------

"""
    get_metric_method_name(x) -> String

The column prefix. Defined on both the TRIGGER and the RESULT because `src` calls it on
both and means slightly different things: the result's name is the bare family
(`"logloss"`), the trigger's name additionally records a selection filter
(`"logloss_over_25"`). `to_dataframe_row` uses the RESULT's, plus a suffix from the
trigger (§4.3) — matching `src/evaluation/translator.jl:46` exactly.
"""
function get_metric_method_name end

get_metric_method_name(::LogLossResult)  = "logloss"
get_metric_method_name(::LPDResult)      = "lpd"
get_metric_method_name(::CRPSResults)    = "crps"
get_metric_method_name(::RQRResult)      = "rqr"
get_metric_method_name(::GLMEdgeResult)  = "glmedge"
get_metric_method_name(::MIQResult)      = "miq"

_ue_family_name(::LogLoss) = "logloss"
_ue_family_name(::LPD)     = "lpd"
_ue_family_name(::CRPS)    = "crps"
_ue_family_name(::RQR)     = "rqr"
_ue_family_name(::GLMEdge) = "glmedge"
_ue_family_name(::MIQ)     = "miq"

function get_metric_method_name(m::AbstractScoringRule)
    sels = scored_selections(m)
    base = _ue_family_name(m)
    return isempty(sels) ? base * "_all" : base * "_" * join(String.(sels), "_")
end

get_metric_method_name(m::CRPS) = "crps"
get_metric_method_name(m::RQR)  = "rqr"
get_metric_method_name(m::MIQ)  = "miq"

"""
    metric_column_suffix(metric) -> String

Appended to the result's column prefix so two rules of the same family in one batch do
not overwrite each other's columns.

`src/evaluation/translator.jl:27` derives this from `selections` alone. This adds the
NON-DEFAULT options too — `LPD(target = :score)` gets `_score`, `GLMEdge(min_edge=0.02)`
gets `_edge002` — and adds NOTHING when every option is at its default. That last
clause is the compatibility guarantee: a legacy construction produces a legacy column
name, character for character.
"""
function metric_column_suffix(m::AbstractScoringRule)
    sels = scored_selections(m)
    return isempty(sels) ? "" : "_" * join(String.(sels), "_")
end

function metric_column_suffix(m::LPD)
    s = isempty(m.selections) ? "" : "_" * join(String.(m.selections), "_")
    return m.target === :score ? s * "_score" : s
end

function metric_column_suffix(m::GLMEdge)
    s = isempty(m.selections) ? "" : "_" * join(String.(m.selections), "_")
    return m.min_edge == 0.0 ? s :
           s * "_edge" * replace(@sprintf("%.3f", m.min_edge), "." => "")
end

metric_column_suffix(m::RQR) = m.n_sims == 1 ? "" : ""
metric_column_suffix(::CRPS) = ""
metric_column_suffix(::MIQ)  = ""


# ==============================================================================
# 5. THE SCORECARD
# ==============================================================================

"""
    EvaluationError(model, metric, message)

One metric that raised on one fit. Collected rather than thrown, so a batch of eleven
models does not lose ten results because the eleventh has no odds coverage — and
REPORTED rather than swallowed, which is the half `src`'s `try/catch`
(`batch_runner.jl:38-41`) gets wrong: it `@warn`s, drops the whole model's row, and the
resulting leaderboard is silently short.
"""
struct EvaluationError
    model::String
    metric::String
    message::String
end

Base.show(io::IO, e::EvaluationError) =
    print(io, "EvaluationError(", e.model, " / ", e.metric, ": ", e.message, ")")

"""
    MetricScorecard

The result of `evaluate_fits`: one wide row per fit, plus the convergence verdict that
decided whether that row exists at all.

| field               | is                                                            |
|---------------------|---------------------------------------------------------------|
| `rows`              | `:model` + every flattened metric column, sorted by `:model`   |
| `convergence`       | one row per fit: verdict, R-hat, ESS, divergences, failed gates|
| `metrics`           | the rules that were run                                        |
| `excluded`          | fits filtered out by the convergence gate, by name             |
| `errors`            | every `(fit, metric)` that raised, with the message            |
| `require_converged` | whether the gate was filtering or only flagging                |
| `elapsed`           | wall-clock seconds                                             |

`convergence` HAS A ROW FOR EVERY FIT, INCLUDING THE EXCLUDED ONES. That is the point:
a scorecard that simply omitted the unconverged runs would look identical to one where
they were never submitted, and "eleven models, three of which did not converge" is a
different message from "eight models".
"""
struct MetricScorecard
    rows::DataFrame
    convergence::DataFrame
    metrics::Vector{AbstractScoringRule}
    excluded::Vector{String}
    errors::Vector{EvaluationError}
    require_converged::Bool
    elapsed::Float64
end

DataFrames.DataFrame(sc::MetricScorecard) = sc.rows
DataFrames.nrow(sc::MetricScorecard) = nrow(sc.rows)
Base.length(sc::MetricScorecard) = nrow(sc.rows)

"The fits that produced a scored row, in scorecard order."
scored_models(sc::MetricScorecard) =
    nrow(sc.rows) == 0 ? String[] : String.(sc.rows.model)

function Base.show(io::IO, sc::MetricScorecard)
    print(io, "MetricScorecard(", nrow(sc.rows), " scored, ",
          length(sc.excluded), " excluded, ", length(sc.errors), " errors)")
end

function Base.show(io::IO, ::MIME"text/plain", sc::MetricScorecard)
    println(io, "MetricScorecard")
    println(io, "  metrics    : ", join(get_metric_method_name.(sc.metrics), ", "))
    println(io, "  scored     : ", nrow(sc.rows), " fit(s)")
    println(io, "  gate       : ", sc.require_converged ?
                "require_converged = true  (unconverged fits EXCLUDED)" :
                "require_converged = false (unconverged fits FLAGGED)")
    if !isempty(sc.excluded)
        println(io, "  excluded   : ", join(sc.excluded, ", "))
    end
    if !isempty(sc.errors)
        println(io, "  errors     : ", length(sc.errors))
        for e in sc.errors
            println(io, "      ", e.model, " / ", e.metric, " — ", e.message)
        end
    end
    print(io,   "  elapsed    : ", format_elapsed(sc.elapsed))
end
