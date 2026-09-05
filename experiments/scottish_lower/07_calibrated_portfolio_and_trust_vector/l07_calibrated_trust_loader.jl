# ==============================================================================
# l07 — Calibrated portfolio forensics, line pruning and trust-vector loader
# ==============================================================================
#
# Definitions only. Every runner in this suite reads its data, its containers, its
# specs, its ledger arithmetic and its grid from here, so that two runners cannot
# quietly disagree about what "the T-25 book" or "capital efficiency" means.
#
# ------------------------------------------------------------------------------
# WHAT THIS SUITE IS ASKING, AND WHY THE ANSWER MOVED
# ------------------------------------------------------------------------------
#
# `experiments/scottish_lower/MARKET_LINE_EDA_REPORT.md` pruned the book down to
# `1X2 + O/U 2.5` and `eda/MULTITIER_TRUST_REPORT.md` fixed the conviction ratio at
# 1.4:1 (`CanonicalScottishLowerTrust`). Both were measured on RAW posterior latents
# priced at the Betfair CLOSE. Two things have changed underneath those verdicts:
#
#   1. `src/Calibration/` graduated. Posterior rate draws are pooled with the
#      tradeable T-25 book, so 1X2 / totals / BTTS are three partitions of one
#      12x12 tensor and edges are shrunk toward the market by construction.
#   2. The price instant moved from the close to T-25, which the calibration stream
#      measured as worth +22 to +38 points of return on the SAME strategy
#      (`current_development/calibration_generative_eda/README.md` §7.2).
#
# A verdict fitted under (raw, close) is therefore not evidence about (calibrated,
# T-25), and this suite re-derives all three: which lines to stake, how much trust
# to put on them, and how much drawdown budget to spend.
#
# ------------------------------------------------------------------------------
# THE ONE STRUCTURAL FACT THAT ORGANISES THE WHOLE TRUST SWEEP
# ------------------------------------------------------------------------------
#
# `stake_slate` composes the stake as
#
#     a_kelly  ->  x trust  ->  x k_shrink  ->  x k_risk  ->  cap  ->  filter
#
# and `FractionalKelly(f)` returns a CONSTANT `k_shrink = f` (`shrinkage.jl:18`).
# So flat trust `tau` at `FractionalKelly(0.30)` is the same pre-risk vector as
# trust `1.0` at `FractionalKelly(0.30 * tau)`: **the trust sweep and the Kelly
# sweep are the same one-dimensional knob**, and this suite sweeps it once, as
# trust, with the Kelly fraction pinned at the production 0.30.
#
# That matters because of the two regimes `README §8.8` identified:
#
#   * while `SlateDrawdown`'s bisected `k_risk < 1`, the constraint absorbs any
#     uniform rescale of the book exactly, so absolute trust is INERT and only the
#     tier RATIO survives (`eda/MULTITIER_TRUST_REPORT.md` §2.1);
#   * once `k_risk` pins at 1 the risk model is doing nothing, lambda is inert, and
#     absolute trust becomes the ONLY live knob.
#
# Calibrated containers stake far less and therefore sit much closer to the second
# regime than raw ones do. Every sweep row in this suite carries `mean_k_risk` and
# `frac_k_pinned`, because a "trust optimum" read off a table where `k_risk < 1`
# everywhere is an artefact and a reader has no way to see that from the return.
#
# ------------------------------------------------------------------------------
# DATABASE BOUNDARY
# ------------------------------------------------------------------------------
#
# READS `mcmc_experiments` (canonical fits) and `betdb` (odds, results). WRITES
# NEITHER. `betdb.paper_runbook` and `betdb.paper_replay` are never opened; the
# live console on 8085 and the replay console on 8086 are not this suite's business.
# ==============================================================================

using BayesianFootball
using CSV
using DataFrames
using Dates
using Printf
using Statistics

const L07_PF = BayesianFootball.Portfolio
const L07_CAL = BayesianFootball.Calibration

const L07_DIR = @__DIR__
const L07_OUT = joinpath(L07_DIR, "results")
isdir(L07_OUT) || mkpath(L07_OUT)


# ==============================================================================
# 1. Configuration — the constants every runner shares
# ==============================================================================

const L07_EXPERIMENT = "scottish_lower_joint_player_2426"

"""
    L07_MODELS

The two canonical runs the work package names. `m12` is the production hybrid
(RAPM teamsheet + pxG); `m05` is the team-state control (wealth covariate + pxG).
Carrying both is what makes Hypothesis 4 — cross-model consistency — answerable at
all: a verdict that holds on one model and not the other is a model quirk.
"""
const L07_MODELS = [
    (key = "m12", name = "m12_joint_hybrid_synergy",
     run_id = "132df5c2-c742-4e95-8693-3aeb2b2cbaef"),
    (key = "m05", name = "m05_joint_production_wealth",
     run_id = "ed541a7c-01e2-447e-a771-783517728d47"),
]

"""
    L07_GATE_SEASONS

The 40-fold 24/25 + 25/26 study. The canonical runs were extended to 43 folds with
the 26/27 August programme on 2026-09-04, so every container is cut back with
`restrict_latents` before anything is scored — otherwise this suite would be
comparing its own numbers to published ones measured on a different fixture set.
"""
const L07_GATE_SEASONS = ["24/25", "25/26"]

"T-25 — the start of MatchDay's execution band, the earliest instant a slate commits."
const L07_AS_OF = -25.0
const L07_MAX_STALENESS = 90.0

"""
    L07_SPLIT_DATE

Slates up to and including this date are the SELECTION window; everything after is
the EVALUATION window. Every pruning verdict and every tuned parameter in this
suite is fitted on the first and reported on the second.

It is the same boundary `MARKET_LINE_EDA_REPORT.md` §0 and the calibration stream's
§6.3 / §7.6 used, deliberately: reusing it keeps this suite's out-of-sample numbers
comparable with theirs rather than merely honest on their own terms.
"""
const L07_SPLIT_DATE = Date(2025, 5, 3)
const L07_PERIOD_START = Date(2024, 8, 1)
const L07_PERIOD_END = Date(2026, 4, 25)

"Production risk settings, held fixed everywhere except where a runner sweeps them."
const L07_KELLY_FRACTION = 0.30
const L07_LAMBDA = 23.0
const L07_CAP = 0.25
const L07_COMMISSION = 0.02

"Bet-count floor below which a per-line verdict is not asserted. See `l07_classify`."
const L07_MIN_BETS = 60


# ==============================================================================
# 2. Containers — the calibrators under test
# ==============================================================================
#
# Four, and each one is here for a stated reason rather than for coverage:
#
#   raw       the in-grid IDENTITY control. `StaticGeometricLaw(w = 1.0)` with the
#             pool map is `is_identity_calibrator`, and `calibrate_latents`
#             short-circuits to a copy on it, so this row must reproduce the
#             uncalibrated model bit for bit. A sweep whose control does not
#             reproduce its own baseline is not a sweep.
#   inv       the T-25 LogLoss optimum on both models (README §7.3), and the arm
#             that reaches +192% at matched risk once both risk knobs move (§8.7).
#   std       the T-25 ECE optimum, and the arm that wins §7.4's lambda-only
#             risk-matched panel. Kept because the two optima disagree and the
#             disagreement is the finding.
#   inv_anch  `inv` with `PreservedDispersion` + `:pool_mean` — README §8.6's
#             `inv_B_anch`, the best scheme measured anywhere in that stream. The
#             anchor, not the preserved width, is what earns it (§8.11 item 4).

l07_calibrators() = [
    ("raw", GenerativeRateCalibrator(name = "identity_control",
                                     law = StaticGeometricLaw(w = 1.0),
                                     book_as_of_minutes = L07_AS_OF)),
    ("inv", GenerativeRateCalibrator(name = "scot_lower_t25_inv",
                                     law = InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
                                     book_as_of_minutes = L07_AS_OF)),
    ("std", GenerativeRateCalibrator(name = "scot_lower_t25_std",
                                     law = StandardGaussianLaw(w_base = 0.40, sigma = 0.15),
                                     book_as_of_minutes = L07_AS_OF)),
    ("inv_anch", GenerativeRateCalibrator(name = "scot_lower_t25_inv_anch",
                                          law = InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
                                          dispersion = PreservedDispersion(),
                                          anchor = :pool_mean,
                                          book_as_of_minutes = L07_AS_OF)),
]


# ==============================================================================
# 3. Specs — books and policies
# ==============================================================================

"""
    l07_book_spec(markets; kelly = L07_KELLY_FRACTION)

The production `BookSpec`: `DeArb` pricing, Kelly log-utility, fractional-Kelly
shrinkage, 2% commission, 0.001 minimum selection stake.

`markets` is the only thing runners vary. `l2_full_direction_markets()` (13
directions, including O/U 0.5) is the FORENSIC scope — it lets the allocator stake
every ladder so each one's economics can be measured. `l2_tradeable_markets()` (11)
is the STAKING scope, and the O/U 0.5 exclusion it encodes is one of the verdicts
this suite re-tests rather than assumes.
"""
l07_book_spec(markets; kelly::Real = L07_KELLY_FRACTION) = L07_PF.BookSpec(
    markets = Data.MarketConfig(markets),
    price = DeArb(),
    allocator = KellyLogUtility(),
    shrink = L07_PF.FractionalKelly(Float64(kelly)),
    exec = ExecutionConfig(commission = PerBetCommission(L07_COMMISSION), budget = 0.99,
                           min_selection_stake = 0.001))

"""
    l07_policy(trust; lambda = L07_LAMBDA, cap = L07_CAP)

`SlateDrawdown(lambda)` + `FixedCap(cap)` + `DailySlate()`. The slate is the
execution atom, so the risk budget is solved once per settlement window over every
fixture in it, not per fixture.
"""
l07_policy(trust; lambda::Real = L07_LAMBDA, cap::Real = L07_CAP) =
    L07_PF.PolicySpec(trust = trust, risk = SlateDrawdown(Float64(lambda)),
                      cap = FixedCap(Float64(cap)), grouping = DailySlate())


# ==============================================================================
# 4. Trust vectors — flat, tiered, and the geometric conviction ladder
# ==============================================================================

"""
    L07_TIER_KEYS

The direction keys `TieredTrust` normalises to: `(market_group, line, direction)`
with totals collapsed to `:over` / `:under`. Spelled once here so a typo in a tier
table is a `KeyError` at construction rather than a silently unstaked market.
"""
const L07_KEY_HOME     = ("1x2", 0.0, :home)
const L07_KEY_DRAW     = ("1x2", 0.0, :draw)
const L07_KEY_AWAY     = ("1x2", 0.0, :away)
const L07_KEY_OVER_05  = ("over_under", 0.5, :over)
const L07_KEY_UNDER_05 = ("over_under", 0.5, :under)
const L07_KEY_OVER_15  = ("over_under", 1.5, :over)
const L07_KEY_UNDER_15 = ("over_under", 1.5, :under)
const L07_KEY_OVER_25  = ("over_under", 2.5, :over)
const L07_KEY_UNDER_25 = ("over_under", 2.5, :under)
const L07_KEY_OVER_35  = ("over_under", 3.5, :over)
const L07_KEY_UNDER_35 = ("over_under", 3.5, :under)
const L07_KEY_BTTS_Y   = ("btts", 0.0, :btts_yes)
const L07_KEY_BTTS_N   = ("btts", 0.0, :btts_no)

const L07_ALL_KEYS = [L07_KEY_HOME, L07_KEY_DRAW, L07_KEY_AWAY,
                      L07_KEY_OVER_05, L07_KEY_UNDER_05,
                      L07_KEY_OVER_15, L07_KEY_UNDER_15,
                      L07_KEY_OVER_25, L07_KEY_UNDER_25,
                      L07_KEY_OVER_35, L07_KEY_UNDER_35,
                      L07_KEY_BTTS_Y, L07_KEY_BTTS_N]

"""
    L07_CANONICAL_TIERS

`CanonicalScottishLowerTrust`'s assignment, as tier INDICES rather than weights:
Home and Under 2.5 in tier 1, Draw and Away in tier 2, everything else gated.

Separating the assignment from the ladder is the whole point. `MULTITIER_TRUST_REPORT`
§2.1 showed the allocator is scale-invariant in absolute trust while the risk
constraint binds, so `(0.35, 0.25)` and `(0.70, 0.50)` are the same portfolio — the
ratio is the parameter and the level is not, *in that regime*. This suite tests
whether calibration moves the container out of that regime, which requires holding
the assignment fixed while the level and the ratio both move.
"""
const L07_CANONICAL_TIERS = Dict(
    L07_KEY_HOME => 1, L07_KEY_UNDER_25 => 1,
    L07_KEY_DRAW => 2, L07_KEY_AWAY => 2,
)

"""
    l07_tiered(tiers, t1, ratio) -> TieredTrust

A geometric conviction ladder: tier `k` receives `t1 / ratio^(k-1)`, every direction
absent from `tiers` receives zero.

`ratio = 1.4` at `t1 = 0.35` reproduces `CanonicalScottishLowerTrust()` exactly (tier
2 lands on 0.25), which `l07_assert_canonical_reproduced` checks rather than trusts.
"""
function l07_tiered(tiers::AbstractDict, t1::Real, ratio::Real)
    ratio > 0 || throw(ArgumentError("conviction ratio must be positive: $ratio"))
    table = Dict{Tuple{String,Float64,Symbol},Float64}()
    for (key, tier) in tiers
        tier >= 1 || throw(ArgumentError("tier index must be >= 1 for $key: $tier"))
        w = Float64(t1) / Float64(ratio)^(tier - 1)
        w <= 1.0 || throw(ArgumentError(
            "tier $tier weight $w exceeds 1.0 at t1 = $t1, ratio = $ratio"))
        table[key] = w
    end
    return TieredTrust(table; default = 0.0)
end

"The canonical policy, rebuilt through `l07_tiered` so the ladder is one code path."
l07_canonical_trust() = l07_tiered(L07_CANONICAL_TIERS, 0.35, 1.4)

"""
    l07_assert_canonical_reproduced()

`l07_tiered(L07_CANONICAL_TIERS, 0.35, 1.4)` must equal `CanonicalScottishLowerTrust()`
to 1e-12 on every one of the 13 directions, including the nine that are zero.

Without this the suite's "canonical" benchmark could drift from production's by a
rounding step in the geometric ladder and every comparison against it would be
against something nobody deploys.
"""
function l07_assert_canonical_reproduced()
    ours = l07_tiered(L07_CANONICAL_TIERS, 0.35, 1.4)
    prod = CanonicalScottishLowerTrust()
    worst = 0.0
    for key in L07_ALL_KEYS
        a = get(ours.table, key, ours.default)
        b = get(prod.table, key, prod.default)
        worst = max(worst, abs(a - b))
    end
    worst <= 1e-12 || error(
        "l07_tiered(canonical, 0.35, 1.4) disagrees with CanonicalScottishLowerTrust() " *
        "by $worst; the geometric ladder is not reproducing the production table.")
    return worst
end


# ==============================================================================
# 5. Data — store, book, inversion, fits
# ==============================================================================

"""
    l07_load_context(; models = L07_MODELS, quiet = false) -> NamedTuple

Everything the runners need, built ONCE:

  `ds`        the Scottish Lower datastore
  `book`      the T-25 point-in-time book, staleness-bounded at 90 minutes
  `refusals`  why each market was refused, by reason
  `coverage`  rows / fixtures / staleness / overround
  `rates`     the inverted `(lambda_mkt_h, lambda_mkt_a)`, computed once
  `fits`      season-restricted canonical fits, keyed by model key

The inversion depends on the BOOK alone — never on the model or the law — so
computing it per calibrator would pay four times for one answer and risk getting
four slightly different ones. It is computed here, once, over the union of both
models' fixture sets, and every `calibrate_fit` call is handed it.
"""
function l07_load_context(; models = L07_MODELS, quiet::Bool = false)
    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
    gate_ids = Set(Int.(ds.matches.match_id[in.(ds.matches.season, Ref(L07_GATE_SEASONS))]))

    book, refusals = L07_CAL.point_in_time_book(ds;
        config = L07_CAL.PointInTimeBookConfig(as_of_minutes = L07_AS_OF,
                                               max_staleness_minutes = L07_MAX_STALENESS))
    coverage = L07_CAL.book_coverage(book, refusals)

    db = PostgresStorage(L07_EXPERIMENT)
    fits = Dict{String,Any}()
    raw_counts = Dict{String,Int}()
    for m in models
        raw = load_fit(db, m.name)
        l = L07_CAL.restrict_latents(raw.latents, gate_ids)
        raw_counts[m.key] = Models.n_matches(raw.latents)
        fits[m.key] = Fit(raw.config, raw.folds, l, raw.diagnostics, raw.metadata,
                          raw.save_path)
    end

    all_ids = unique(vcat([collect(Models.latent_match_ids(fits[m.key].latents))
                           for m in models]...))
    rates = L07_CAL.invert_market_rates(first(l07_calibrators())[2], book;
                                        match_ids = all_ids)
    icov = L07_CAL.inversion_coverage(rates, all_ids)

    if !quiet
        @printf("datastore   : %d matches, %d in gate seasons\n",
                nrow(ds.matches), length(gate_ids))
        @printf("book        : %d rows | %d fixtures | %d markets | staleness med %.0f p90 %.0f | overround %.4f\n",
                coverage.n_rows, coverage.n_fixtures, coverage.n_markets,
                coverage.median_staleness, coverage.p90_staleness, coverage.median_overround)
        for (reason, n) in L07_CAL.book_refusal_summary(refusals)
            @printf("              refused %5d  %s\n", n, reason)
        end
        for m in models
            @printf("fit %-4s    : %d -> %d fixtures after gate-season restriction | converged=%s\n",
                    m.key, raw_counts[m.key], Models.n_matches(fits[m.key].latents),
                    fits[m.key].diagnostics.passed)
        end
        @printf("inversion   : %d/%d accepted (%.1f%% of all, %.1f%% of quoted)\n",
                icov.n_accepted, icov.n_fixtures, 100 * icov.coverage,
                100 * icov.coverage_quoted)
        for (reason, n) in L07_CAL.inversion_refusals(rates)
            @printf("              refused %5d  %s\n", n, reason)
        end
    end

    return (; ds, gate_ids, book, refusals, coverage, rates, icov, fits, db)
end

"""
    l07_container(ctx, model_key, cal) -> CalibratedFit

`calibrate_fit` with the shared inversion. `quiet = true` because a sweep that
prints per-fixture diagnostics 32 times is unreadable, and the diagnostics are on
the returned object for anything that wants them.
"""
l07_container(ctx, model_key::AbstractString, cal) =
    calibrate_fit(cal, ctx.fits[model_key], ctx.book; rates = ctx.rates, quiet = true)


# ==============================================================================
# 6. The ledger — one row per struck bet, in comparable units
# ==============================================================================

"""
    l07_family_catalog(books) -> Dict{String,NamedTuple}

`family` -> `(group, line, selection)`, read off the `Selection` objects the
simulation actually priced. Never hard-coded: the family encoding belongs to the
odds feed and is free to change, and a stale hard-coded map would mislabel a line
rather than fail.
"""
function l07_family_catalog(books)
    catalog = Dict{String,NamedTuple}()
    for book in books, sel in book.sels
        get!(catalog, sel.family) do
            (; group = sel.group, line = sel.line, selection = sel.selection)
        end
    end
    return catalog
end

"Readable market key (`\"OU2.5\"`) for a priced selection."
function l07_market_key(group::AbstractString, line::Real)
    group == "1X2" && return "1X2"
    group == "BTTS" && return "BTTS"
    group == "OverUnder" && return "OU" * string(line)
    return String(group)
end

"""
    l07_ledger(result, books; model, container, trust) -> DataFrame

The struck bets, rescaled out of bankroll fractions into currency.

`stake` and `pnl` are fractions of the bankroll at their OWN slate. In a compounding
backtest that makes them incommensurable across time — a 1% stake in the first week
and a 1% stake in the last are different amounts of money — so both are multiplied
by the slate's opening bankroll before anything is summed. Every currency figure in
this suite is post-rescale, and `l07_gate_ledger_accounting` asserts the rescale
reproduces the simulation's own totals.
"""
function l07_ledger(result, books; model::AbstractString, container::AbstractString,
                    trust::AbstractString)
    bets = copy(result.trajectory.bets)
    nrow(bets) == 0 && return DataFrame()

    opening = Dict{Date,Float64}()
    for state in result.daily_states
        haskey(opening, state.date) && error(
            "Two slates share the date $(state.date); the bankroll rescale would be ambiguous.")
        opening[state.date] = state.bankroll_open
    end

    catalog = l07_family_catalog(books)
    missing_families = setdiff(Set(String.(bets.family)), keys(catalog))
    isempty(missing_families) || error(
        "Ledger holds families absent from the priced books: $(collect(missing_families)).")

    bets.model = fill(String(model), nrow(bets))
    bets.container = fill(String(container), nrow(bets))
    bets.trust = fill(String(trust), nrow(bets))
    bets.group = [catalog[f].group for f in bets.family]
    bets.line = [catalog[f].line for f in bets.family]
    bets.market_key = [l07_market_key(catalog[f].group, catalog[f].line) for f in bets.family]
    bets.direction = [String(catalog[f].selection) for f in bets.family]
    bets.bank_open = [opening[d] for d in bets.date]
    bets.abs_stake = bets.stake .* bets.bank_open
    bets.abs_pnl = bets.pnl .* bets.bank_open
    bets.won = bets.payoff .> 0
    bets.pushed = bets.payoff .== 0
    bets.edge = bets.p_model .- bets.p_market
    bets.window = [d <= L07_SPLIT_DATE ? "selection" : "evaluation" for d in bets.date]
    return bets
end

"""
    l07_gate_ledger_accounting(result, ledger) -> NamedTuple

GATE 1 of the work package, on one simulation.

Three invariants, and each one catches a different class of bug:

  * every `stake`, `pnl`, `odds`, `p_model` and `p_market` is finite — no NaN, no
    Inf, no missing price survived pricing;
  * the ledger's bankroll-fraction stake and P&L sum to `trajectory.total_stake`
    and `total_pl`, so the rescale in `l07_ledger` did not lose or invent capital;
  * `pnl == stake * payoff` bet by bet, which is the identity the whole ROI column
    rests on.

The third is the load-bearing one: if it holds, "Kelly ROI" is realised P&L over
realised turnover and cannot disagree with `PortfolioSummary.roi` except through a
weighting choice that is stated.
"""
function l07_gate_ledger_accounting(result, ledger::DataFrame)
    nrow(ledger) == 0 && return (; ok = false, reason = "empty ledger",
                                 n_nonfinite = 0, stake_err = NaN, pnl_err = NaN,
                                 identity_err = NaN)
    cols = [:stake, :pnl, :payoff, :odds, :p_model, :p_market, :abs_stake, :abs_pnl]
    n_nonfinite = sum(sum(.!isfinite.(ledger[!, c])) for c in cols)

    stake_err = abs(sum(ledger.stake) - result.trajectory.total_stake)
    pnl_err = abs(sum(ledger.pnl) - result.trajectory.total_pl)
    identity_err = maximum(abs.(ledger.pnl .- ledger.stake .* ledger.payoff))

    ok = n_nonfinite == 0 && stake_err <= 1e-9 && pnl_err <= 1e-9 && identity_err <= 1e-12
    return (; ok, reason = ok ? "pass" : "see fields", n_nonfinite, stake_err, pnl_err,
            identity_err)
end


# ==============================================================================
# 7. Per-line metrics
# ==============================================================================

"Longest run of `true` in `x`."
function l07_max_streak(x::AbstractVector{Bool})
    best = 0; run = 0
    for v in x
        run = v ? run + 1 : 0
        run > best && (best = run)
    end
    return best
end

"""
    l07_standalone_drawdown(frame) -> (units, pct_of_turnover)

Deepest peak-to-trough excursion of this slice's OWN cumulative P&L.

NOT the drawdown the line caused the portfolio. Kelly allocates jointly, so removing
a line changes every other stake in the slate; the portfolio counterfactual is a
re-simulation, not a subtraction. What this measures is how LUMPY the contribution
was, which is what separates a line that bleeds steadily from one that is flat until
it is not.
"""
function l07_standalone_drawdown(frame::AbstractDataFrame)
    nrow(frame) == 0 && return (0.0, NaN)
    ordered = sort(frame, [:date, :match_id])
    equity = cumsum(ordered.abs_pnl)
    peak = -Inf; worst = 0.0
    for v in equity
        v > peak && (peak = v)
        worst = min(worst, v - peak)
    end
    turnover = sum(ordered.abs_stake)
    return (worst, turnover > 0 ? 100 * worst / turnover : NaN)
end

"Book-wide denominators for the share and efficiency columns, within one window."
function l07_totals(frame::AbstractDataFrame)
    turnover = sum(frame.abs_stake)
    pnl = sum(frame.abs_pnl)
    return (; turnover, pnl, kelly_roi = turnover > 0 ? 100 * pnl / turnover : NaN)
end

"""
    l07_line_metrics(frame, totals) -> NamedTuple

Everything the forensic tables read for one slice of the ledger.

`flat_roi_pct` is `mean(payoff)`: at a flat one-unit stake the net per-unit payoff IS
the return, so no separate staking simulation is needed for it. `kelly_roi_pct` is
realised P&L over realised turnover — what the compounding backtest actually earned
on the capital it committed here. The two disagree exactly when Kelly's sizing
disagrees with the line's average edge, and that gap is the signal `l07_classify`
reads.

`capital_efficiency` is this slice's Kelly ROI over the WHOLE BOOK's, in the SAME
window. Above 1.00 the line returns more than its share of the capital it consumed;
below it, the line is being carried. It is `NaN` when the book's own ROI is not
positive, because a ratio to a negative denominator flips sign and would report a
losing line as efficient.

`capital_efficiency_anchored` is the same ratio against a FIXED denominator supplied
by the caller — in this suite, the SELECTION window's book ROI for the same container
and scope. It exists because the same-window ratio turns out to be unusable out of
sample: over the 50 evaluation slates the book's own ROI collapses toward zero, so
the ratio either explodes (a +45% line divided by a +0.7% book reads as efficiency
63) or is undefined (a book that lost money). Neither can be compared against a 0.25
threshold. The anchored version divides by a number that was KNOWN at the split date,
so it is both stable and legitimately out-of-sample, and it is what
[`l07_oos_gate`](@ref) tests. The same-window column is kept beside it because the
work package names it and because a reader should be able to see the instability
rather than take this note on trust.
"""
function l07_line_metrics(frame::AbstractDataFrame, totals::NamedTuple;
                         anchor_roi::Real = NaN)
    n = nrow(frame)
    n == 0 && return nothing
    ordered = sort(frame, [:date, :match_id])
    turnover = sum(frame.abs_stake)
    pnl = sum(frame.abs_pnl)
    win_rate = mean(frame.won)
    mean_p_model = mean(frame.p_model)
    dd_units, dd_pct = l07_standalone_drawdown(frame)

    kelly_roi = turnover > 0 ? 100 * pnl / turnover : NaN
    efficiency = (isfinite(kelly_roi) && isfinite(totals.kelly_roi) && totals.kelly_roi > 0) ?
                 kelly_roi / totals.kelly_roi : NaN
    efficiency_anchored = (isfinite(kelly_roi) && isfinite(anchor_roi) && anchor_roi > 0) ?
                          kelly_roi / anchor_roi : NaN

    return (;
        n_bets = n,
        n_matches = length(unique(frame.match_id)),
        win_rate_pct = 100 * win_rate,
        push_rate_pct = 100 * mean(frame.pushed),
        mean_odds = mean(frame.odds),
        mean_p_model = mean_p_model,
        mean_p_market = mean(frame.p_market),
        mean_edge = mean(frame.edge),
        calib_bias = win_rate - mean_p_model,
        flat_roi_pct = 100 * mean(frame.payoff),
        kelly_roi_pct = kelly_roi,
        total_pnl_units = pnl,
        turnover_units = turnover,
        capital_share_pct = totals.turnover > 0 ? 100 * turnover / totals.turnover : NaN,
        capital_efficiency = efficiency,
        capital_efficiency_anchored = efficiency_anchored,
        mean_stake_frac = mean(frame.stake),
        max_stake_frac = maximum(frame.stake),
        standalone_dd_units = dd_units,
        standalone_dd_pct_turnover = dd_pct,
        max_win_streak = l07_max_streak(ordered.won),
        max_loss_streak = l07_max_streak(.!ordered.won .& .!ordered.pushed),
    )
end

"""
    l07_breakdown(ledger; scope) -> DataFrame

Line-level (`direction = "ALL"`) and direction-level rows, for each of the three
windows (`full`, `selection`, `evaluation`), in one long frame.

Both granularities live together because the Over/Under asymmetry question is a
comparison BETWEEN the two levels, and splitting them into separate files would make
that join the reader's problem. The window is a column for the same reason: the
in-sample / out-of-sample stability question is a comparison between windows.
"""
function l07_breakdown(ledger::DataFrame; scope::AbstractString = "")
    nrow(ledger) == 0 && return DataFrame()
    # The anchor: the SELECTION window's book ROI. Known at the split date, so using it
    # as the evaluation window's efficiency denominator leaks nothing forward.
    sel = filter(:window => ==("selection"), ledger)
    anchor = nrow(sel) == 0 ? NaN : l07_totals(sel).kelly_roi

    rows = NamedTuple[]
    for window in ("full", "selection", "evaluation")
        w = window == "full" ? ledger : filter(:window => ==(window), ledger)
        nrow(w) == 0 && continue
        totals = l07_totals(w)
        base = (; scope, window,
                model = first(w.model), container = first(w.container),
                trust = first(w.trust),
                book_kelly_roi_pct = totals.kelly_roi,
                book_turnover_units = totals.turnover,
                anchor_book_roi_pct = anchor)

        for sub in groupby(sort(w, :market_key), :market_key)
            m = l07_line_metrics(sub, totals; anchor_roi = anchor)
            m === nothing && continue
            push!(rows, merge(base, (; market_key = first(sub.market_key),
                                     direction = "ALL"), m))
            for dsub in groupby(sort(sub, :direction), :direction)
                dm = l07_line_metrics(dsub, totals; anchor_roi = anchor)
                dm === nothing && continue
                push!(rows, merge(base, (; market_key = first(sub.market_key),
                                         direction = first(dsub.direction)), dm))
            end
        end
    end
    return isempty(rows) ? DataFrame() : DataFrame(rows)
end


# ==============================================================================
# 8. The pruning rule
# ==============================================================================

"""
    l07_classify(kelly_roi, efficiency, n_bets; min_bets = L07_MIN_BETS) -> (verdict, reason)

    KEEP        kelly_roi > 0  AND  efficiency >= 0.50  AND  n_bets >= min_bets
    PRUNE       kelly_roi <= 0  OR  (efficiency < 0.25 AND n_bets >= min_bets)
    CONDITIONAL otherwise

Deliberately the SAME rule as `MARKET_LINE_EDA_REPORT.md` §2, at a lower bet floor
(60 rather than 100) because calibration strikes 20-40% fewer bets and the old floor
would refuse to judge lines that are now simply smaller.

A line can clear the ROI test and fail the efficiency one. That is the dilution case
precisely: profitable, but at a rate low enough to drag the book's average down while
occupying slate budget the exposure cap then denies to better selections.

**§5.1 of that report recorded this rule being REFUTED out of sample** — it selected
a basket that finished last of every multi-market configuration tested. It is kept
because it explains WHY a line pays; the adjudicator is the out-of-sample
re-simulation in `r07_optimal_portfolio_comparison.jl`, not this function.
"""
function l07_classify(kelly_roi::Real, efficiency::Real, n_bets::Integer;
                      min_bets::Integer = L07_MIN_BETS)
    if n_bets < min_bets
        return ("CONDITIONAL", @sprintf("only %d bets, below the %d floor", n_bets, min_bets))
    end
    if !isfinite(kelly_roi)
        return ("CONDITIONAL", "no turnover")
    end
    if kelly_roi <= 0
        return ("PRUNE", @sprintf("Kelly ROI %.2f%% <= 0 on %d bets", kelly_roi, n_bets))
    end
    if !isfinite(efficiency)
        return ("CONDITIONAL",
                @sprintf("Kelly ROI %+.2f%% but the book's own ROI is not positive, so efficiency is undefined",
                         kelly_roi))
    end
    if efficiency < 0.25
        return ("PRUNE", @sprintf("capital efficiency %.2f < 0.25 on %d bets",
                                  efficiency, n_bets))
    end
    if efficiency >= 0.50
        return ("KEEP", @sprintf("Kelly ROI %+.2f%%, efficiency %.2f", kelly_roi, efficiency))
    end
    return ("CONDITIONAL", @sprintf("Kelly ROI %+.2f%%, efficiency %.2f — profitable but dilutive",
                                    kelly_roi, efficiency))
end

"""
    l07_oos_gate(sel, eva) -> (pass, reason)

GATE 2 of the work package, stated as the work package states it: a direction may be
recommended for a non-zero staking tier only if `Kelly ROI > 0` AND
`capital_efficiency >= 0.25` in BOTH the selection and the evaluation window.

This is strictly harder than `l07_classify`, and on purpose. `l07_classify` reads one
window and can be fitted; this reads both and cannot.

Efficiency here is `capital_efficiency_anchored` — the ratio against the SELECTION
window's book ROI — not the same-window ratio the work package names. The same-window
version is reported beside it and is unusable in the evaluation window for the reason
given in `l07_line_metrics`: with the book's own out-of-sample ROI near zero the ratio
either explodes or is undefined, and a threshold of 0.25 then admits or refuses on the
denominator rather than on the line.
"""
function l07_oos_gate(sel::Union{Nothing,NamedTuple}, eva::Union{Nothing,NamedTuple})
    sel === nothing && return (false, "no bets in the selection window")
    eva === nothing && return (false, "no bets in the evaluation window")
    checks = [
        (sel.kelly_roi_pct > 0, @sprintf("IS Kelly ROI %+.2f%%", sel.kelly_roi_pct)),
        (eva.kelly_roi_pct > 0, @sprintf("OOS Kelly ROI %+.2f%%", eva.kelly_roi_pct)),
        (isfinite(sel.capital_efficiency_anchored) && sel.capital_efficiency_anchored >= 0.25,
         @sprintf("IS efficiency %.2f", sel.capital_efficiency_anchored)),
        (isfinite(eva.capital_efficiency_anchored) && eva.capital_efficiency_anchored >= 0.25,
         @sprintf("OOS efficiency %.2f", eva.capital_efficiency_anchored)),
    ]
    failed = [msg for (ok, msg) in checks if !ok]
    isempty(failed) && return (true, "all four conditions hold")
    return (false, join(failed, "; "))
end


# ==============================================================================
# 9. Window metrics — a compounding path restricted to a date range
# ==============================================================================

"""
    l07_window_metrics(states, from, to) -> NamedTuple

Return, Sharpe, drawdown and Calmar for the slates in `[from, to]`, recompounded
from a bankroll of 1.0 at the window's first slate.

It is NOT a slice of the full-period equity curve: an out-of-sample return has to be
what a bettor starting at that date would have made, so the window is re-based. The
BETS are still the full-period simulation's bets, though, so this measures the
out-of-sample performance of a strategy whose stakes were sized against the
in-sample bankroll path — which is the honest reading and is stated rather than
hidden, because re-simulating from the split date would also re-solve every
subsequent slate's risk budget and answer a different question.

`sharpe_ann` scales the per-slate log-return Sharpe by `sqrt(slates_per_year)`
inferred from the window's own calendar span, matching `PortfolioSummary`'s
convention exactly so the two can sit side by side.
"""
function l07_window_metrics(states, from::Date, to::Date)
    window = [s for s in states if from <= s.date <= to]
    length(window) < 2 && return (; n_slates = length(window), return_pct = NaN,
                                  cagr_pct = NaN, sharpe_ann = NaN, mdd_pct = NaN,
                                  calmar = NaN, turnover = NaN, n_bets = 0,
                                  mean_exposure = NaN, mean_k_risk = NaN,
                                  frac_k_pinned = NaN)

    pnl = [s.pnl_frac for s in window]
    bank = cumprod(1.0 .+ pnl)
    peak = accumulate(max, bank)
    drawdown = 100 .* (bank .- peak) ./ peak

    logret = log.(1.0 .+ pnl)
    days = Dates.value(window[end].date - window[1].date)
    slates_per_year = days > 0 ? length(window) * 365.25 / days : NaN
    sharpe = std(logret) > 0 ? mean(logret) / std(logret) : NaN
    sharpe_ann = (isnan(sharpe) || isnan(slates_per_year)) ? NaN : sharpe * sqrt(slates_per_year)

    ret_pct = 100 * (bank[end] - 1.0)
    cagr_pct = days > 0 ? 100 * (bank[end]^(365.25 / days) - 1.0) : NaN
    mdd = minimum(drawdown)
    calmar = mdd < 0 ? cagr_pct / abs(mdd) : NaN

    return (;
        n_slates = length(window),
        return_pct = ret_pct,
        cagr_pct,
        sharpe_ann,
        mdd_pct = mdd,
        calmar,
        turnover = sum(s.stake_frac for s in window),
        n_bets = sum(s.n_bets for s in window),
        mean_exposure = mean(s.exposure for s in window),
        mean_k_risk = mean(s.k_risk for s in window),
        frac_k_pinned = mean(s.k_risk .>= 1.0 - 1e-9 for s in window),
    )
end

"""
    l07_summary_row(result; kw...) -> NamedTuple

`PortfolioSummary`'s headline fields plus the two risk-regime diagnostics
(`mean_k_risk`, `frac_k_pinned`) and the in-sample / out-of-sample window metrics,
flattened into one row so a sweep is one `DataFrame`.

`frac_k_pinned` is the fraction of slates on which `SlateDrawdown`'s bisected `k`
hit 1.0 — i.e. the constraint stopped binding and lambda stopped doing anything. A
lambda sweep without it is uninterpretable (README §8.8), and a trust sweep without
it is worse: while `k < 1` a uniform trust rescale is absorbed EXACTLY and the
optimum a table appears to show is arithmetic noise.
"""
function l07_summary_row(result; kw...)
    s = result.summary
    states = result.daily_states
    k = [st.k_risk for st in states]
    isel = l07_window_metrics(states, L07_PERIOD_START, L07_SPLIT_DATE)
    oos = l07_window_metrics(states, L07_SPLIT_DATE + Day(1), L07_PERIOD_END)
    return merge((; kw...), (;
        n_slates = s.n_slates, n_bets = s.n_bets,
        return_pct = s.total_return_pct, cagr_pct = 100 * s.cagr,
        flat_roi_pct = s.roi, sharpe_ann = s.sharpe_ann, sortino = s.sortino,
        mdd_pct = s.mdd, calmar = s.calmar, ulcer = s.ulcer,
        turnover = s.total_stake, mean_exposure = s.mean_exposure,
        max_exposure = s.max_exposure, worst_slate = s.worst_slate,
        mean_k_risk = isempty(k) ? NaN : mean(k),
        frac_k_pinned = isempty(k) ? NaN : mean(k .>= 1.0 - 1e-9),
        n_capped = s.n_capped,
        is_return_pct = isel.return_pct, is_sharpe = isel.sharpe_ann,
        is_mdd_pct = isel.mdd_pct, is_calmar = isel.calmar, is_slates = isel.n_slates,
        oos_return_pct = oos.return_pct, oos_cagr_pct = oos.cagr_pct,
        oos_sharpe = oos.sharpe_ann, oos_mdd_pct = oos.mdd_pct,
        oos_calmar = oos.calmar, oos_slates = oos.n_slates,
        oos_k_pinned = oos.frac_k_pinned,
    ))
end


# ==============================================================================
# 10. The sweep grid
# ==============================================================================

"§3.D.1 — flat trust levels."
const L07_FLAT_TAUS = [0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]

"""
    L07_TIER1_TAUS, L07_RATIOS

§3.D.1's tiered levels and conviction ratios, plus two grid points the work package
did not ask for and the analysis cannot do without:

  `0.35`  the CANONICAL tier-1 level. Without it the production policy is not a cell
          in its own sweep and every table would have to interpolate to say where it
          sits.
  `1.0`   the ratio at which the tiers collapse to a uniform weight over the
          canonical BASKET. It separates the two things "tiered trust" bundles —
          which directions are staked at all, and how the staked ones are weighted
          against each other — so a gain can be attributed to one or the other.
"""
const L07_TIER1_TAUS = [0.30, 0.35, 0.50, 0.70, 1.00]
const L07_RATIOS = [1.0, 1.2, 1.4, 1.6, 2.0]

"§3.D.2 — the slate drawdown risk parameter."
const L07_LAMBDAS = [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 23.0, 28.0]

"§3.D.3 — exposure caps."
const L07_CAPS = [0.20, 0.25]

"""
    l07_trust_grid(; tiers = L07_CANONICAL_TIERS) -> Vector{NamedTuple}

23 trust vectors: 7 flat levels and 16 (tier-1 level x conviction ratio) pairs.

The flat arm is not redundant with `ratio = 1.0`: flat trust stakes all 11 tradeable
directions while the tiered arm gates 9 of 13 to zero, so the two differ in the BET
SET and not only in the weights. That difference is Hypothesis 1's territory and the
grid has to span it.
"""
function l07_trust_grid(; tiers::AbstractDict = L07_CANONICAL_TIERS)
    rows = NamedTuple[]
    for tau in L07_FLAT_TAUS
        push!(rows, (; kind = "flat", label = @sprintf("flat_%.2f", tau),
                     t1 = tau, ratio = 1.0, trust = FlatTrust(tau)))
    end
    for t1 in L07_TIER1_TAUS, r in L07_RATIOS
        push!(rows, (; kind = "tiered", label = @sprintf("tier_%.2f_r%.1f", t1, r),
                     t1 = t1, ratio = r, trust = l07_tiered(tiers, t1, r)))
    end
    return rows
end

"""
    l07_sweep(slates, grid; br, lambdas, caps, quiet) -> DataFrame

The cartesian sweep, run against PRE-BUILT, PRE-GROUPED slates.

Book construction — pricing every fixture's 12x12 tensor, the de-arb, the allocator
warm-up — is by far the dominant cost and depends only on `(BookSpec, container)`,
never on the policy. Building it once and re-solving only `stake_slate` per grid
point is what makes a 368-cell grid per container affordable at all. It is also what
makes the grid a controlled comparison: every cell prices bit-identical books.

`bootstrap = false` throughout: a 4,000-resample confidence interval per cell would
dominate the runtime and no cell is quoted with one.
"""
function l07_sweep(slates, grid; br, lambdas = L07_LAMBDAS, caps = L07_CAPS,
                   quiet::Bool = true, kw...)
    rows = NamedTuple[]
    for g in grid, lam in lambdas, cap in caps
        policy = l07_policy(g.trust; lambda = lam, cap = cap)
        result = L07_PF.simulate_portfolio(policy, slates; converged = br.converged,
                                           failed_gates = br.failed_gates,
                                           bootstrap = false)
        push!(rows, l07_summary_row(result; kw..., trust_kind = g.kind,
                                    trust_label = g.label, t1 = g.t1, ratio = g.ratio,
                                    lambda = lam, cap = cap))
        quiet || @printf("  %-16s lam %5.1f cap %.2f -> %+8.2f%%  Sharpe %5.3f  MDD %7.2f%%  k %.3f\n",
                         g.label, lam, cap, rows[end].return_pct, rows[end].sharpe_ann,
                         rows[end].mdd_pct, rows[end].mean_k_risk)
    end
    return DataFrame(rows)
end


# ==============================================================================
# 11. Reporting helpers
# ==============================================================================

"Markdown table from a header and pre-formatted string rows."
function l07_md_table(header::Vector{String}, rows::Vector{Vector{String}})
    io = IOBuffer()
    println(io, "| ", join(header, " | "), " |")
    println(io, "| ", join(fill(":---", length(header)), " | "), " |")
    for r in rows
        println(io, "| ", join(r, " | "), " |")
    end
    return String(take!(io))
end

l07_fmt(x::Real; d::Int = 2) = isfinite(x) ? string(round(x; digits = d)) : "—"
l07_pct(x::Real) = isfinite(x) ? @sprintf("%+.2f", x) : "—"
l07_pct3(x::Real) = isfinite(x) ? @sprintf("%+.3f", x) : "—"

"Write `df` to `results/<name>` and say so."
function l07_write(df::DataFrame, name::AbstractString)
    path = joinpath(L07_OUT, name)
    CSV.write(path, df)
    @printf("  wrote %-46s %7d rows\n", name, nrow(df))
    return path
end

println("l07 loader ready — ", length(l07_calibrators()), " containers, ",
        length(L07_MODELS), " models, ",
        length(l07_trust_grid()) * length(L07_LAMBDAS) * length(L07_CAPS),
        " grid cells per container")
