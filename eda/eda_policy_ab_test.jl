# ==============================================================================
# eda_policy_ab_test.jl — policy A/B on the canonical Gen-4 40-fold fits
# ==============================================================================
#
# Question: when six correlated market expressions compete for one daily-slate risk budget,
# does line-specific trust improve realised log-growth relative to FlatTrust(0.30)?
#
# Held fixed: immutable PostgreSQL fit UUID, Betfair TWA close [-20, 0], full six-market book,
# DeArb settlement, KellyLogUtility, no second shrinkage multiplier, 2% commission,
# SlateDrawdown(23), FixedCap(0.20), DailySlate, and the historical fixture set.
#
# Policies:
#   P1 status quo       FlatTrust(0.30)
#   P2 hard pruning     0.30 on 1X2/O-U 2.5; 0.00 elsewhere
#   P3 damped tails     0.30 on 1X2/O-U 2.5; 0.05 elsewhere
#   P4 drawdown adaptive uniform 0.30 × {1.00, 0.75, 0.50, 0.25}
#
# P4 is deliberately included as a falsifiable control. Portfolio's drawdown solver is
# homogeneous of degree zero, so uniform trust scaling should be neutralised whenever
# SlateDrawdown binds. The daily output records whether that expected no-op occurs.
#
# This runner launches no MCMC. It loads exact completed fits from mcmc_experiments.
#
# Usage:
#   julia --project -t 8 eda/eda_policy_ab_test.jl
#
# Outputs:
#   eda/results/stochastic_control_capacity/policy_ab_summary.csv
#   eda/results/stochastic_control_capacity/policy_ab_daily.csv
#   eda/results/stochastic_control_capacity/policy_ab_ledger.csv
#   eda/results/stochastic_control_capacity/policy_ab_build_report.csv
#   eda/results/stochastic_control_capacity/policy_definitions.csv

using BayesianFootball
using CSV
using DataFrames
using Dates
using LibPQ
using LinearAlgebra
using Printf
using Statistics
using ThreadPinning
using UUIDs

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "stochastic_control_common.jl"))

const SC_POLICY_BUILD_PATH = joinpath(SC_OUTPUT_DIR, "policy_ab_build_report.csv")
const SC_POLICY_SUMMARY_PATH = joinpath(SC_OUTPUT_DIR, "policy_ab_summary.csv")
const SC_POLICY_DAILY_PATH = joinpath(SC_OUTPUT_DIR, "policy_ab_daily.csv")
const SC_POLICY_LEDGER_PATH = joinpath(SC_OUTPUT_DIR, "policy_ab_ledger.csv")
const SC_POLICY_DEFINITIONS_PATH = joinpath(SC_OUTPUT_DIR, "policy_definitions.csv")

function sc_policy_definitions()
    return DataFrame(
        policy = ["P1_status_quo", "P2_hard_pruning", "P3_damped_tail", "P4_drawdown_adaptive"],
        core_trust = [0.30, 0.30, 0.30, 0.30],
        fringe_trust = [0.30, 0.00, 0.05, 0.30],
        state_dependent = [false, false, false, true],
        description = [
            "FlatTrust(0.30) on every selection",
            "SelectionTrust: 0.30 on 1X2/O-U 2.5 and 0.00 on fringe lines",
            "SelectionTrust: 0.30 on 1X2/O-U 2.5 and 0.05 on fringe lines",
            "Flat 0.30 trust multiplied by 1.00/0.75/0.50/0.25 at 0/5/10/15% opening drawdown",
        ],
    )
end

function sc_policy_factories()
    return [
        ("P1_status_quo", sc_status_quo_policy),
        ("P2_hard_pruning", sc_hard_pruning_policy),
        ("P3_damped_tail", sc_damped_tail_policy),
        ("P4_drawdown_adaptive", sc_drawdown_adaptive_policy),
    ]
end

function sc_build_report_row(ref::SCCanonicalRun, report, fit)
    diagnostics = fit.diagnostics
    return (
        experiment = ref.experiment,
        model = ref.name,
        run_uuid = string(ref.run_uuid),
        config_hash = ref.config_hash,
        n_folds = length(fit),
        strict_convergence_passed = diagnostics.passed,
        max_rhat = diagnostics.max_rhat,
        min_ess_bulk = diagnostics.min_ess_bulk,
        divergences = diagnostics.n_divergent,
        n_fixtures = report.n_fixtures,
        n_books = report.n_books,
        n_skipped = length(report.skipped_no_fixture) + length(report.skipped_unplayed) +
                    length(report.skipped_no_quotes) + length(report.skipped_no_selections) +
                    length(report.errored),
        build_converged = report.converged,
        failed_gates = join(report.failed_gates, "; "),
    )
end

function eda_policy_ab_test()
    println("\n", "="^118)
    println(" STOCHASTIC CONTROL — SIX-MARKET POLICY A/B")
    println("="^118)

    mkpath(SC_OUTPUT_DIR)
    inventory = sc_verify_run_inventory()
    println("  Run inventory verified: $(nrow(inventory)) immutable 40-fold addresses")

    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
    betfair = sc_betfair_closing_odds(ds)
    book_spec = sc_book_spec()
    println("  DataStore : $(nrow(ds.matches)) matches")
    println("  Betfair   : $(nrow(betfair)) rows, $(length(unique(betfair.match_id))) matches")
    println("  Book      : 1X2, O/U 0.5, 1.5, 2.5, 3.5, BTTS")
    println("  Risk      : SlateDrawdown(23), FixedCap(0.20), 2% commission")

    summary_rows = NamedTuple[]
    daily_frames = DataFrame[]
    ledger_frames = DataFrame[]
    build_rows = NamedTuple[]

    for ref in sc_policy_runs()
        println("\n", "-"^118)
        println(" $(ref.name) — $(ref.run_uuid)")
        println("-"^118)
        fit = sc_load_fit(ref)

        # Build the score grids and MatchBooks once. PolicySpec is explicitly the cheap sweep seam.
        status_policy = sc_status_quo_policy()
        status_result, books, report = run_portfolio_simulation(
            book_spec,
            status_policy,
            fit,
            betfair,
            ds;
            initial_bankroll = SC_INITIAL_BANKROLL,
            bootstrap = false,
            require_converged = false,
            quiet = true,
        )
        push!(build_rows, sc_build_report_row(ref, report, fit))

        for (policy_name, factory) in sc_policy_factories()
            result = policy_name == "P1_status_quo" ? status_result :
                     SC_PF.simulate_portfolio(
                         factory(),
                         books,
                         report;
                         initial_bankroll = SC_INITIAL_BANKROLL,
                         bootstrap = false,
                     )
            push!(summary_rows, sc_policy_summary(result, ref.name, policy_name))
            push!(daily_frames, sc_policy_daily_frame(result, ref.name, policy_name))
            push!(ledger_frames, sc_simulation_ledger(result, books, ref.name, policy_name))

            summary = result.summary
            @printf("  %-22s bank=%8.2f ret=%+8.2f%% Sharpe=%6.3f MDD=%7.2f%% bets=%4d turn=%6.2f cap=%3d\n",
                    policy_name, summary.final_bankroll, summary.total_return_pct,
                    summary.sharpe_ann, summary.mdd, summary.n_bets, summary.total_stake,
                    summary.n_capped)
        end
    end

    summary = DataFrame(summary_rows)
    daily = vcat(daily_frames...; cols = :union)
    ledger = vcat(ledger_frames...; cols = :union)
    builds = DataFrame(build_rows)
    definitions = sc_policy_definitions()

    # Internal consistency gates: the ledger and daily states must reduce to the engine summary.
    for row in eachrow(summary)
        selected_daily = daily[(daily.model .== row.model) .& (daily.policy .== row.policy), :]
        selected_ledger = ledger[(ledger.model .== row.model) .& (ledger.policy .== row.policy), :]
        nrow(selected_ledger) == row.n_bets || error(
            "$(row.model)/$(row.policy) ledger has $(nrow(selected_ledger)) bets; summary has $(row.n_bets).")
        isapprox(sum(selected_daily.stake_frac), row.total_turnover; atol = 1e-12, rtol = 1e-12) ||
            error("$(row.model)/$(row.policy) turnover does not reproduce the summary.")
        isapprox(selected_daily.bankroll_close[end], row.final_bankroll; atol = 1e-10, rtol = 1e-12) ||
            error("$(row.model)/$(row.policy) final bankroll does not reproduce the summary.")
    end

    CSV.write(SC_POLICY_SUMMARY_PATH, summary)
    CSV.write(SC_POLICY_DAILY_PATH, daily)
    CSV.write(SC_POLICY_LEDGER_PATH, ledger)
    CSV.write(SC_POLICY_BUILD_PATH, builds)
    CSV.write(SC_POLICY_DEFINITIONS_PATH, definitions)

    println("\nWrote:")
    for path in (SC_POLICY_SUMMARY_PATH, SC_POLICY_DAILY_PATH, SC_POLICY_LEDGER_PATH,
                 SC_POLICY_BUILD_PATH, SC_POLICY_DEFINITIONS_PATH)
        println("  ", path)
    end
    return (; summary, daily, ledger, builds, definitions)
end

if abspath(PROGRAM_FILE) == @__FILE__
    eda_policy_ab_test()
end
