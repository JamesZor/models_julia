# ==============================================================================
# r67 — Fractional-Kelly portfolio backtest for the hierarchical team-kappa grid
# ==============================================================================
#
# WHAT THIS IS. The economic reading of r66. Proper scores say whether a posterior
# is sharper; they do not say whether a Kelly stake vector built from it compounds.
# The two are not the same question — Experiment 06 already recorded a case where
# the LogLoss winner was not the bankroll winner — so both candidates and both of
# their shared-κ controls are simulated over the identical 99 Scottish Lower slates.
#
# TWO CONFIGURATIONS, AND WHY BOTH.
#
#   A · `l60_book` + `l60_policy` — the Experiment 06 research baseline.
#       FractionalKelly(0.30), 2% exchange commission, FlatTrust(1.0),
#       SlateDrawdown(23.0), FixedCap(0.20), DailySlate.
#       Every selection the book prices is stakeable. This is the configuration the
#       shared-κ leaderboard was measured under, so it is the only one under which a
#       hierarchical number can be compared to a recorded one.
#
#   B · `l60_book` + `MatchDay.canonical_scottish_lower_policy()` — production.
#       `CanonicalScottishLowerTrust()` (`P1_conservative_tilt`: Home and Under 2.5 at
#       0.35, Draw and Away at 0.25, everything else at exactly 0.00), SlateDrawdown(23.0),
#       FixedCap(0.25), DailySlate.
#       This is what the live MatchDay ledger stakes with. A component that helps under
#       A and not under B does not help the account that exists.
#
#   The BOOK is held identical across A and B on purpose. Changing the market set and
#   the policy together would leave a difference unattributable; the tiered trust already
#   gates the market set down to what it trusts.
#
# PERSISTENCE. Every simulated run, its bet ledger and its metadata go to PostgreSQL
# `mcmc_experiments` through `save_portfolio_db`, keyed to the model run UUID, and every
# one is read back with `load_portfolio_db` and checked bet-for-bet before the runner
# claims it was stored.
#
# HOW TO RUN
#   julia --project -t 16 experiments/scottish_lower/06_joint_player_lineup_fusion/r67_portfolio_hierarchical_kappa.jl
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and shared experiment state
# ==============================================================================
using BayesianFootball
using CSV
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Statistics
using ThreadPinning
using UUIDs

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

# Brings `ds`, `db`, the book and policy objects, the run manifest, the Betfair
# closing-line frame, and the §0 artefact compatibility shim without which the two
# shared-κ controls cannot be deserialized at all.
include(joinpath(@__DIR__, "l66_hierarchical_kappa_eval_loader.jl"))

const R67_MD = BayesianFootball.MatchDay
const R67_PORTFOLIO = BayesianFootball.Portfolio
const R67_INF = BayesianFootball.Training.Inference

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================
const R67_NAMES = L66_NAMES
const R67_OUTPUT_DIR = L66_OUTPUT_DIR

const R67_CONFIGS = [
    ("A_experiment06_baseline", l60_book, l60_policy),
    ("B_canonical_production", l60_book, R67_MD.canonical_scottish_lower_policy()),
]

# %%
# ==============================================================================
# 3. Metrics
# ==============================================================================
"""
    r67_turnover(summary) -> Float64

How many times the bankroll was cycled through the book over the whole backtest.

Stated because "turnover" has no single convention. `total_stake` is ALREADY accumulated
in bankroll-FRACTION units — each bet contributes its stake as a fraction of the bankroll
open on its slate — so it is the turnover multiple directly, and dividing it by
`initial_bankroll` (a currency amount) would be a unit error. The identity that checks
this: `total_stake ≈ mean_exposure × n_slates`, and `flat_roi = total_pnl / total_stake`.

It is a gross-exposure number, not a return.
"""
r67_turnover(s) = s.total_stake

"""
    r67_metric_row(...) -> NamedTuple

Everything the work package asks for, plus the MATCH-CLUSTERED bootstrap interval on flat ROI.

The interval is the only number here that says whether a return difference is a result. Eleven
selections on one fixture share one scoreline, so `bootstrap_portfolio` resamples FIXTURES, not
bets; resampling bets would divide the standard error by roughly √11 and make every arm look
significantly profitable. `p_roi_positive` is the resampled probability that flat ROI exceeds zero.
"""
function r67_metric_row(name, config, run_id, portfolio_id, n_books, n_skip, s, ci)
    return (
        model = name, config = config,
        model_run_id = string(run_id), portfolio_run_id = string(portfolio_id),
        n_books = n_books, n_skipped = n_skip,
        n_slates = s.n_slates, n_fixtures = s.n_fixtures, n_bets = s.n_bets,
        total_return_pct = s.total_return_pct,
        cagr_pct = 100 * s.cagr,
        flat_roi_pct = s.roi,
        roi_1x2_pct = s.roi_1x2,
        sharpe = s.sharpe,
        sharpe_ann = s.sharpe_ann,
        sortino = s.sortino,
        calmar = s.calmar,
        martin = s.martin,
        ulcer = s.ulcer,
        max_drawdown_pct = s.mdd,
        worst_slate_pct = 100 * s.worst_slate,
        win_rate_pct = 100 * s.win_rate,
        mean_exposure = s.mean_exposure,
        max_exposure = s.max_exposure,
        turnover = r67_turnover(s),
        total_stake = s.total_stake,
        total_pnl = s.total_pnl,
        mean_k_risk = s.mean_k_risk,
        n_capped = s.n_capped,
        span_days = s.span_days,
        roi_lo = ci === nothing ? NaN : ci.roi_lo,
        roi_hi = ci === nothing ? NaN : ci.roi_hi,
        roi_sd = ci === nothing ? NaN : ci.roi_sd,
        growth_lo = ci === nothing ? NaN : ci.growth_lo,
        growth_hi = ci === nothing ? NaN : ci.growth_hi,
        p_roi_positive = ci === nothing ? NaN : ci.p_roi_positive,
        bootstrap_B = ci === nothing ? 0 : ci.B,
    )
end

"""
    r67_season_pnl(bets, season_of) -> DataFrame

Flat ROI and staked P/L split by season. A backtest ROI is one number over two seasons;
this is what says whether it came from both of them.

`roi` here is FLAT: realised P/L over stake, in bankroll-fraction units, not a compounded
return. The compounded number is path-dependent and cannot be decomposed by season.
"""
function r67_season_pnl(bets::AbstractDataFrame, season_of::Dict{Int,String})
    nrow(bets) == 0 && return DataFrame()
    frame = copy(bets)
    frame.season = [get(season_of, Int(m), "unknown") for m in frame.match_id]
    return combine(groupby(frame, :season),
        nrow => :n_bets,
        :stake => sum => :stake,
        :pnl => sum => :pnl,
        [:pnl, :stake] => ((p, s) -> 100 * sum(p) / sum(s)) => :flat_roi_pct,
        :pnl => (p -> 100 * mean(p .> 0)) => :win_rate_pct)
end

# %%
# ==============================================================================
# 4. Execution
# ==============================================================================
println("\n" * "="^118)
println(" EXPERIMENT 06 · HIERARCHICAL TEAM KAPPA · FRACTIONAL-KELLY PORTFOLIO BACKTEST")
println("="^118)
println("  database  : ", db)
println("  book      : FractionalKelly(0.30) · 2% commission · 1X2 + O/U 2.5 + BTTS · DeArb pricing")
println("  config A  : FlatTrust(1.0), SlateDrawdown(23.0), FixedCap(0.20), DailySlate")
println("  config B  : CanonicalScottishLowerTrust(), SlateDrawdown(23.0), FixedCap(0.25), DailySlate")
println("  started   : ", Dates.now())

bf_odds = l66_betfair_closing_odds(ds)
@printf("  Betfair TWA closes: %d rows across %d matches\n",
        nrow(bf_odds), length(unique(bf_odds.match_id)))

season_of = l66_season_of(ds)

fits = Dict{String,Any}()
run_ids = Dict{String,UUID}()
for name in R67_NAMES
    fit, resolved = l66_load_fit(db, name)
    fits[name] = fit
    run_ids[name] = resolved
    @printf("  loaded %-30s run %s  div %d  %s\n", name, string(resolved),
            fit.diagnostics.n_divergent,
            fit.diagnostics.passed ? "CONVERGED" : "NOT CONVERGED")
end

# THE BOOK IS BUILT ONCE PER MODEL, NOT ONCE PER (MODEL, CONFIGURATION). Both configurations
# share one `BookSpec`, so pricing every fixture twice would be waste — and it would leave the two
# configurations priced by two separately constructed objects, which is exactly the kind of seam a
# paired comparison must not have. What differs between A and B is the policy, and only the policy.
rows = NamedTuple[]
ledger_frames = DataFrame[]
season_frames = DataFrame[]

books_by_model = Dict{String,Any}()
reports_by_model = Dict{String,Any}()
for name in R67_NAMES
    fit = fits[name]
    converged = fit.diagnostics.passed
    converged || @warn "$name did not pass strict convergence gating; pricing with require_converged=false" rhat=fit.diagnostics.max_rhat div=fit.diagnostics.n_divergent
    books, report = R67_PORTFOLIO.build_books_reported(
        l60_book, fit, bf_odds, ds; require_converged = converged, quiet = true)
    books_by_model[name] = books
    reports_by_model[name] = report
    @printf("  priced %-30s %d books | %d skipped\n", name, length(books), n_skipped(report))
end

for (config_name, book, policy) in R67_CONFIGS
    println("\n" * "="^118)
    println(" CONFIGURATION $config_name")
    println("="^118)
    for name in R67_NAMES
        fit = fits[name]
        converged = fit.diagnostics.passed
        books = books_by_model[name]
        report = reports_by_model[name]

        result = R67_PORTFOLIO.simulate_portfolio(policy, books, report; bootstrap = true)

        portfolio_id = save_portfolio_db(
            result, run_ids[name], db;
            book_spec = book, policy_spec = policy,
            metadata = (; candidate = name,
                          config = config_name,
                          kappa = occursin("hierarchical", name) ? "hierarchical" : "shared",
                          arm = startswith(name, "m05") ? "m05" : "m12",
                          runner = "r67_portfolio_hierarchical_kappa",
                          odds_source = "betfair_twa_minus20_to_close",
                          converged = converged))

        # PERSISTENCE IS NOT CLAIMED UNTIL IT IS READ BACK. A ledger that round-trips
        # unequal means the CSV below and the database disagree about what was bet.
        reloaded = load_portfolio_db(portfolio_id, db)
        reloaded.summary.total_return_pct == result.summary.total_return_pct || error(
            "$name/$config_name: reloaded return differs from the simulated one")
        isequal(reloaded.trajectory.bets, result.trajectory.bets) || error(
            "$name/$config_name: reloaded bet ledger differs from the simulated one")

        s = result.summary
        push!(rows, r67_metric_row(name, config_name, run_ids[name], portfolio_id,
                                   length(books), n_skipped(report), s, result.bootstrap_ci))

        bets = copy(result.trajectory.bets)
        if nrow(bets) > 0
            insertcols!(bets, 1, :model => fill(name, nrow(bets)))
            insertcols!(bets, 2, :config => fill(config_name, nrow(bets)))
            insertcols!(bets, 3, :model_run_id => fill(string(run_ids[name]), nrow(bets)))
            insertcols!(bets, 4, :portfolio_run_id => fill(string(portfolio_id), nrow(bets)))
            push!(ledger_frames, bets)

            season = r67_season_pnl(result.trajectory.bets, season_of)
            insertcols!(season, 1, :model => fill(name, nrow(season)))
            insertcols!(season, 2, :config => fill(config_name, nrow(season)))
            push!(season_frames, season)
        end

        @printf("  %-30s bets %5d | return %+8.2f%% | ROI %+6.2f%% | MDD %7.2f%% | Sharpe %5.3f | portfolio %s\n",
                name, s.n_bets, s.total_return_pct, s.roi, s.mdd, s.sharpe_ann,
                string(portfolio_id))
    end
end

# %%
# ==============================================================================
# 5. Output
# ==============================================================================
summary = DataFrame(rows)
ledger = vcat(ledger_frames...; cols = :union)
seasons = vcat(season_frames...; cols = :union)

CSV.write(joinpath(R67_OUTPUT_DIR, "r67_portfolio_summary.csv"), summary)
CSV.write(joinpath(R67_OUTPUT_DIR, "r67_trade_ledger.csv"), ledger)
CSV.write(joinpath(R67_OUTPUT_DIR, "r67_portfolio_by_season.csv"), seasons)

for (config_name, _, _) in R67_CONFIGS
    sub = sort(filter(:config => ==(config_name), summary), :total_return_pct; rev = true)
    println("\n" * "="^168)
    println(" CONFIGURATION $config_name")
    println("="^168)
    @printf(" %-30s | %5s | %8s | %8s | %-20s | %8s | %8s | %7s | %7s | %7s | %7s | %8s | %8s\n",
            "Model", "Bets", "Return%", "FlatROI", "ROI 95% CI (clustered)", "P(ROI>0)",
            "1X2 ROI", "MaxDD%", "Sharpe", "Sortino", "Calmar", "Win%", "Turnover")
    println("-"^168)
    for row in eachrow(sub)
        @printf(" %-30s | %5d | %+8.2f | %+8.2f | [%+7.2f, %+7.2f] | %8.3f | %8.2f | %7.2f | %7.3f | %7.3f | %7.3f | %7.2f | %8.2f\n",
                row.model, row.n_bets, row.total_return_pct, row.flat_roi_pct,
                row.roi_lo, row.roi_hi, row.p_roi_positive,
                row.roi_1x2_pct, row.max_drawdown_pct, row.sharpe_ann, row.sortino,
                row.calmar, row.win_rate_pct, row.turnover)
    end
    println("="^168)
end

println("\n PAIRED CONTRAST — hierarchical minus shared, same configuration")
@printf(" %-4s | %-24s | %10s | %10s | %10s | %9s\n",
        "Arm", "Config", "Δ Return%", "Δ ROI%", "Δ MaxDD%", "Δ Sharpe")
println("-"^90)
for (config_name, _, _) in R67_CONFIGS, arm in ("m05", "m12")
    control = L66_CONTROL_NAME[arm]
    candidate = L66_CANDIDATE_NAME[arm]
    c = only(filter(r -> r.model == control && r.config == config_name, eachrow(summary)))
    h = only(filter(r -> r.model == candidate && r.config == config_name, eachrow(summary)))
    @printf(" %-4s | %-24s | %+10.2f | %+10.2f | %+10.2f | %+9.3f\n",
            arm, config_name,
            h.total_return_pct - c.total_return_pct,
            h.flat_roi_pct - c.flat_roi_pct,
            h.max_drawdown_pct - c.max_drawdown_pct,
            h.sharpe_ann - c.sharpe_ann)
end

println("\n BY SEASON (flat ROI on staked P/L)")
@printf(" %-30s | %-24s | %-7s | %6s | %10s | %9s\n",
        "Model", "Config", "Season", "Bets", "Flat ROI%", "Win%")
println("-"^102)
for row in eachrow(sort(seasons, [:config, :model, :season]))
    @printf(" %-30s | %-24s | %-7s | %6d | %+10.2f | %9.2f\n",
            row.model, row.config, row.season, row.n_bets, row.flat_roi_pct, row.win_rate_pct)
end

@printf("\nPostgreSQL now holds %d portfolio runs, their ledgers and their artefacts for this study.\n",
        nrow(summary))
println("CSVs written under $R67_OUTPUT_DIR")
println("Finished: ", Dates.now())
