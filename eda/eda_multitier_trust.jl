# ==============================================================================
# eda_multitier_trust.jl — categorical multi-tier SelectionTrust across six markets
# ==============================================================================
#
# The asymmetric-selection-trust audit (eda/eda_asymmetric_selection_trust.jl,
# eda/ASYMMETRIC_SELECTION_TRUST_REPORT.md) established a strict binary trust gate: Home, Draw,
# Away, and Under 2.5 at flat tau=0.30 ("asymmetric core"), everything else at zero. That gate
# treats a super-alpha line (Under 2.5, pooled ROI +18.67%, IR 1.242) identically to a modest
# diversifier (Away, pooled ROI +5.63%, IR 0.654). This script asks whether tilting the scarce
# 20%-of-bankroll slate cap toward the highest-margin directions with a categorical multi-tier
# trust table improves risk-adjusted growth, or whether it only concentrates tail risk.
#
# Six candidate policies are tested, re-solving the full joint Kelly allocation from scratch for
# every tier assignment (surviving stakes are never obtained by rescaling an old ledger):
#
#   P0_flat_benchmark    — Tier 1 (Under 2.5, Home) and Tier 2 (Draw, Away) both at tau=0.30.
#                          This is exactly asymmetric core; it reproduces the standing champion.
#   P1_conservative_tilt — Tier 1 @ 0.35, Tier 2 @ 0.25.
#   P2_conviction_tilt   — Tier 1 @ 0.40, Tier 2 @ 0.20.
#   P3_aggressive_tilt   — Tier 1 @ 0.50, Tier 2 @ 0.25.
#   P4_four_tier_probe   — Tier 1 @ 0.40, Tier 2 @ 0.25, Tier 3 (Under 1.5, Over 3.5) @ 0.05.
#   P5_grid_sweep        — every (tau1, tau2) pair on a 6x6 grid, tau1 in [0.25,0.50] step 0.05,
#                          tau2 in [0.10,0.35] step 0.05, tau3 fixed at 0.00. Traces the Return
#                          vs. Max Drawdown Pareto frontier as conviction concentrates capital.
#
# Every other fringe direction (O/U 0.5, O/U 3.5 under, O/U 1.5 over, BTTS both sides, Under 0.5,
# Over 2.5) stays pinned at zero trust in every policy tested here; that pruning is the finding of
# the prior audit, not something this script re-litigates.
#
# This runner launches no MCMC. Saved-fit compatibility requires execution at the
# artifact-compatible worktree pinned to commit 784c8ea81328760e75498b19d13c2dab762bde8e
# (`/home/james/bet_project/.worktrees/BayesianFootball-stochastic-eda-runtime`), because the
# current `JointGammaPoissonObservation` type has changed since serialization.
#
# Usage:
#   julia --project -t 8 eda/eda_multitier_trust.jl
#
# Outputs:
#   eda/results/multitier_trust/multitier_policy_summary.csv
#   eda/results/multitier_trust/multitier_policy_windows.csv
#   eda/results/multitier_trust/multitier_policy_daily.csv
#   eda/results/multitier_trust/multitier_policy_ledger.csv
#   eda/results/multitier_trust/multitier_selection_summary.csv
#   eda/results/multitier_trust/multitier_grid_sweep.csv
#   eda/results/multitier_trust/multitier_policy_definitions.csv
#   eda/results/multitier_trust/multitier_build_report.csv

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

const MT_OUTPUT_DIR = joinpath(@__DIR__, "results", "multitier_trust")

const MT_SELECTION_CATALOG = [
    (family = "1X2_home", group = "1X2", line = 0.0, selection = :home,
     market_key = "1X2", direction = "home"),
    (family = "1X2_draw", group = "1X2", line = 0.0, selection = :draw,
     market_key = "1X2", direction = "draw"),
    (family = "1X2_away", group = "1X2", line = 0.0, selection = :away,
     market_key = "1X2", direction = "away"),
    (family = "O/U 0.5_over_05", group = "OverUnder", line = 0.5, selection = :over_05,
     market_key = "OU0.5", direction = "over"),
    (family = "O/U 0.5_under_05", group = "OverUnder", line = 0.5, selection = :under_05,
     market_key = "OU0.5", direction = "under"),
    (family = "O/U 1.5_over_15", group = "OverUnder", line = 1.5, selection = :over_15,
     market_key = "OU1.5", direction = "over"),
    (family = "O/U 1.5_under_15", group = "OverUnder", line = 1.5, selection = :under_15,
     market_key = "OU1.5", direction = "under"),
    (family = "O/U 2.5_over_25", group = "OverUnder", line = 2.5, selection = :over_25,
     market_key = "OU2.5", direction = "over"),
    (family = "O/U 2.5_under_25", group = "OverUnder", line = 2.5, selection = :under_25,
     market_key = "OU2.5", direction = "under"),
    (family = "O/U 3.5_over_35", group = "OverUnder", line = 3.5, selection = :over_35,
     market_key = "OU3.5", direction = "over"),
    (family = "O/U 3.5_under_35", group = "OverUnder", line = 3.5, selection = :under_35,
     market_key = "OU3.5", direction = "under"),
    (family = "BTTS_btts_yes", group = "BTTS", line = 0.0, selection = :btts_yes,
     market_key = "BTTS", direction = "yes"),
    (family = "BTTS_btts_no", group = "BTTS", line = 0.0, selection = :btts_no,
     market_key = "BTTS", direction = "no"),
]

const MT_TIER1_FAMILIES = Set(["1X2_home", "O/U 2.5_under_25"])
const MT_TIER2_FAMILIES = Set(["1X2_draw", "1X2_away"])
const MT_TIER3_FAMILIES = Set(["O/U 1.5_under_15", "O/U 3.5_over_35"])

const MT_POLICY_SPECS = [
    (name = "P0_flat_benchmark", tau1 = 0.30, tau2 = 0.30, tau3 = 0.00,
     description = "Flat asymmetric core: Home, Draw, Away, Under 2.5 all at tau=0.30; every " *
                   "other direction at 0.00. Reproduces the standing champion " *
                   "(m12 +143.91%, Sharpe 1.516)."),
    (name = "P1_conservative_tilt", tau1 = 0.35, tau2 = 0.25, tau3 = 0.00,
     description = "Tier 1 (Under 2.5, Home) at 0.35; Tier 2 (Draw, Away) at 0.25."),
    (name = "P2_conviction_tilt", tau1 = 0.40, tau2 = 0.20, tau3 = 0.00,
     description = "Tier 1 at 0.40; Tier 2 at 0.20."),
    (name = "P3_aggressive_tilt", tau1 = 0.50, tau2 = 0.25, tau3 = 0.00,
     description = "Tier 1 at 0.50; Tier 2 at 0.25."),
    (name = "P4_four_tier_probe", tau1 = 0.40, tau2 = 0.25, tau3 = 0.05,
     description = "Tier 1 at 0.40; Tier 2 at 0.25; Tier 3 (Under 1.5, Over 3.5) at 0.05."),
]

const MT_TAU1_GRID = round.(collect(0.25:0.05:0.50); digits = 2)
const MT_TAU2_GRID = round.(collect(0.10:0.05:0.35); digits = 2)

function mt_tier_label(family::AbstractString)
    family in MT_TIER1_FAMILIES && return "tier1"
    family in MT_TIER2_FAMILIES && return "tier2"
    family in MT_TIER3_FAMILIES && return "tier3"
    return "excluded"
end

function mt_selection_policy(tau1::Real, tau2::Real, tau3::Real = 0.0)
    table = Dict{Tuple{String,Float64,Symbol},Float64}()
    for row in MT_SELECTION_CATALOG
        weight = row.family in MT_TIER1_FAMILIES ? tau1 :
                 row.family in MT_TIER2_FAMILIES ? tau2 :
                 row.family in MT_TIER3_FAMILIES ? tau3 : 0.0
        table[(row.group, row.line, row.selection)] = Float64(weight)
    end
    return PolicySpec(
        trust = SC_PF.SelectionTrust(table; strict = true),
        risk = SlateDrawdown(23.0),
        cap = FixedCap(SC_CAP),
        grouping = DailySlate(),
    )
end

"Window metrics on the [from, to] slice: return, annualized Sharpe/Sortino, Calmar, max DD."
function mt_window_metrics(daily::DataFrame, from::Date, to::Date)
    frame = daily[(Date.(daily.date) .>= from) .& (Date.(daily.date) .<= to), :]
    nrow(frame) >= 2 || return (
        n_slates = nrow(frame), return_pct = NaN, annual_sharpe = NaN, annual_sortino = NaN,
        calmar_ratio = NaN, max_drawdown_pct = NaN, n_bets = sum(frame.n_bets),
        total_turnover = sum(frame.stake_frac),
    )
    pnl = Float64.(frame.pnl_frac)
    wealth = cumprod(1.0 .+ pnl)
    peak = accumulate(max, wealth)
    drawdown = 100 .* (wealth ./ peak .- 1.0)
    log_returns = log.(1.0 .+ pnl)
    span_days = Dates.value(Date(frame.date[end]) - Date(frame.date[1]))
    slates_per_year = span_days > 0 ? nrow(frame) * 365.25 / span_days : NaN
    annual_sharpe = span_days > 0 && std(log_returns) > 0 ?
        mean(log_returns) / std(log_returns) * sqrt(slates_per_year) : NaN
    down = log_returns[log_returns .< 0.0]
    sortino = isempty(down) ? Inf : mean(log_returns) / sqrt(mean(down .^ 2))
    annual_sortino = span_days > 0 && isfinite(sortino) ? sortino * sqrt(slates_per_year) : sortino
    return_pct = 100 * (wealth[end] - 1.0)
    max_drawdown_pct = minimum(drawdown)
    calmar_ratio = max_drawdown_pct < 0 ? return_pct / abs(max_drawdown_pct) : 0.0
    return (
        n_slates = nrow(frame),
        return_pct,
        annual_sharpe,
        annual_sortino,
        calmar_ratio,
        max_drawdown_pct,
        n_bets = sum(frame.n_bets),
        total_turnover = sum(frame.stake_frac),
    )
end

function mt_window_summary(daily::DataFrame, model::AbstractString, policy::AbstractString,
                           all_dates::Vector{Date}; tau1::Real, tau2::Real, tau3::Real)
    split_date = all_dates[cld(length(all_dates), 2)]
    windows = [
        (name = "full", from = all_dates[1], to = all_dates[end]),
        (name = "first_half", from = all_dates[1], to = split_date),
        (name = "second_half", from = split_date + Day(1), to = all_dates[end]),
    ]
    rows = NamedTuple[]
    for window in windows
        metrics = mt_window_metrics(daily, window.from, window.to)
        push!(rows, (; model = String(model), policy = String(policy),
                     tau1 = Float64(tau1), tau2 = Float64(tau2), tau3 = Float64(tau3),
                     window = window.name, date_from = window.from, date_to = window.to,
                     metrics...))
    end
    return DataFrame(rows)
end

function mt_policy_summary_row(result, daily::DataFrame, ledger::DataFrame,
                               model::AbstractString, policy_name::AbstractString;
                               tau1::Real, tau2::Real, tau3::Real)
    summary = result.summary
    full = mt_window_metrics(daily, Date(daily.date[1]), Date(daily.date[end]))
    return (
        model = String(model),
        policy = String(policy_name),
        tau1 = Float64(tau1),
        tau2 = Float64(tau2),
        tau3 = Float64(tau3),
        initial_bankroll = summary.initial_bankroll,
        final_bankroll = summary.final_bankroll,
        total_return_pct = summary.total_return_pct,
        annual_sharpe = summary.sharpe_ann,
        annual_sortino = full.annual_sortino,
        calmar_ratio = full.calmar_ratio,
        max_drawdown_pct = summary.mdd,
        n_bets = summary.n_bets,
        total_turnover = summary.total_stake,
        n_capped = summary.n_capped,
        win_rate_pct = 100 * summary.win_rate,
        mean_exposure = summary.mean_exposure,
        max_exposure = summary.max_exposure,
        mean_k_risk = summary.mean_k_risk,
        max_single_bet_amount = isempty(ledger) ? 0.0 : maximum(ledger.stake_amount),
        daily_pnl_volatility = std(daily.pnl_frac),
        n_slates = summary.n_slates,
    )
end

function mt_grid_row(model::AbstractString, tau1::Real, tau2::Real, result, daily::DataFrame)
    summary = result.summary
    full = mt_window_metrics(daily, Date(daily.date[1]), Date(daily.date[end]))
    return (
        model = String(model),
        tau1 = Float64(tau1),
        tau2 = Float64(tau2),
        final_bankroll = summary.final_bankroll,
        total_return_pct = summary.total_return_pct,
        annual_sharpe = summary.sharpe_ann,
        annual_sortino = full.annual_sortino,
        calmar_ratio = full.calmar_ratio,
        max_drawdown_pct = summary.mdd,
        n_bets = summary.n_bets,
        total_turnover = summary.total_stake,
        n_capped = summary.n_capped,
        mean_k_risk = summary.mean_k_risk,
    )
end

function mt_selection_summary(ledger::DataFrame)
    rows = NamedTuple[]
    for group in groupby(ledger, [:model, :policy, :market_key, :family, :selection])
        stake = sum(group.stake_amount)
        pnl = sum(group.pnl_amount)
        push!(rows, (
            model = String(group.model[1]),
            policy = String(group.policy[1]),
            market_key = String(group.market_key[1]),
            family = String(group.family[1]),
            tier = mt_tier_label(String(group.family[1])),
            selection = String(group.selection[1]),
            n_bets = nrow(group),
            stake_amount = stake,
            pnl_amount = pnl,
            win_rate_pct = 100 * mean(group.won),
            roi_pct = stake > 0 ? 100 * pnl / stake : NaN,
            mean_odds = mean(group.odds),
            mean_model_probability = mean(group.p_model),
            mean_market_probability = mean(group.p_market),
            mean_edge = mean(group.p_model .- group.p_market),
        ))
    end
    summary = DataFrame(rows)
    DataFrames.transform!(groupby(summary, [:model, :policy]),
        :stake_amount => (s -> 100 .* s ./ sum(s)) => :stake_share_pct)
    return summary
end

function mt_policy_definition_rows()
    rows = NamedTuple[]
    for spec in MT_POLICY_SPECS, row in MT_SELECTION_CATALOG
        tier = mt_tier_label(row.family)
        trust = tier == "tier1" ? spec.tau1 : tier == "tier2" ? spec.tau2 :
                tier == "tier3" ? spec.tau3 : 0.0
        push!(rows, (
            policy = spec.name,
            family = row.family,
            market_key = row.market_key,
            selection = String(row.selection),
            tier,
            tau1 = spec.tau1,
            tau2 = spec.tau2,
            tau3 = spec.tau3,
            trust,
            active = trust > 0.0,
            description = spec.description,
        ))
    end
    return DataFrame(rows)
end

function mt_build_report_row(ref::SCCanonicalRun, report, fit)
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

function mt_validate(summary::DataFrame, daily::DataFrame, ledger::DataFrame,
                     definitions::DataFrame, grid::DataFrame)
    expected_policies = Set(String.([spec.name for spec in MT_POLICY_SPECS]))
    Set(summary.policy) == expected_policies || error("Policy summary is incomplete.")
    nrow(summary) == 2 * length(MT_POLICY_SPECS) ||
        error("Expected $(2 * length(MT_POLICY_SPECS)) model-policy summary rows; " *
              "found $(nrow(summary)).")
    all(summary.n_slates .== 100) || error("Every policy must cover exactly 100 slate dates.")
    nrow(definitions) == length(MT_POLICY_SPECS) * length(MT_SELECTION_CATALOG) ||
        error("Policy definition table does not cover every directional selection.")
    nrow(grid) == 2 * length(MT_TAU1_GRID) * length(MT_TAU2_GRID) ||
        error("Grid sweep does not cover every (model, tau1, tau2) cell.")

    for row in eachrow(summary)
        selected_daily = daily[(daily.model .== row.model) .& (daily.policy .== row.policy), :]
        selected_ledger = ledger[(ledger.model .== row.model) .& (ledger.policy .== row.policy), :]
        nrow(selected_daily) == row.n_slates || error("$(row.model)/$(row.policy) daily row mismatch.")
        nrow(selected_ledger) == row.n_bets || error("$(row.model)/$(row.policy) ledger row mismatch.")
        isapprox(selected_daily.bankroll_close[end], row.final_bankroll; atol = 1e-9, rtol = 1e-12) ||
            error("$(row.model)/$(row.policy) final bankroll does not reproduce summary.")
        isapprox(sum(selected_daily.stake_frac), row.total_turnover; atol = 1e-9, rtol = 1e-12) ||
            error("$(row.model)/$(row.policy) turnover does not reproduce summary.")
        allowed = Set(String.(definitions.family[(definitions.policy .== row.policy) .&
                                                 definitions.active]))
        isempty(setdiff(Set(String.(selected_ledger.family)), allowed)) || error(
            "$(row.model)/$(row.policy) staked a gated selection.")
    end

    # P0_flat_benchmark is exactly the (tau1=0.30, tau2=0.30) grid cell; the two independent code
    # paths (named-policy factory vs. grid loop) must agree on both models.
    for row in eachrow(summary[summary.policy .== "P0_flat_benchmark", :])
        cell = grid[(grid.model .== row.model) .& (grid.tau1 .== 0.30) .& (grid.tau2 .== 0.30), :]
        nrow(cell) == 1 || error("$(row.model) grid is missing the P0 cross-check cell.")
        isapprox(cell.final_bankroll[1], row.final_bankroll; atol = 1e-6, rtol = 1e-9) ||
            error("$(row.model) grid cell (0.30, 0.30) disagrees with P0_flat_benchmark.")
    end
    return nothing
end

function eda_multitier_trust()
    println("\n", "="^122)
    println(" MULTI-TIER CATEGORICAL TRUST — SIX-MARKET CONVICTION-TILT POLICY SWEEP")
    println("="^122)
    mkpath(MT_OUTPUT_DIR)

    inventory = sc_verify_run_inventory()
    println("  Run inventory verified: $(nrow(inventory)) immutable addresses")
    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
    betfair = sc_betfair_closing_odds(ds)
    book_spec = sc_book_spec()

    books_by_model = Dict{String,Any}()
    reports_by_model = Dict{String,Any}()
    baseline_by_model = Dict{String,Any}()
    build_rows = NamedTuple[]

    println("\n[1/5] Building each model's six-market books once, via P0_flat_benchmark...")
    for ref in sc_policy_runs()
        fit = sc_load_fit(ref)
        result, books, report = run_portfolio_simulation(
            book_spec,
            mt_selection_policy(0.30, 0.30, 0.00),
            fit,
            betfair,
            ds;
            initial_bankroll = SC_INITIAL_BANKROLL,
            bootstrap = false,
            require_converged = false,
            quiet = true,
        )
        books_by_model[ref.name] = books
        reports_by_model[ref.name] = report
        baseline_by_model[ref.name] = result
        push!(build_rows, mt_build_report_row(ref, report, fit))
        @printf("  %-28s books=%3d slates=%3d baseline=%+8.2f%% Sharpe=%6.3f\n",
                ref.name, length(books), result.summary.n_slates,
                result.summary.total_return_pct, result.summary.sharpe_ann)
    end

    println("\n[2/5] Simulating the five predeclared tier policies...")
    summary_rows = NamedTuple[]
    daily_frames = DataFrame[]
    ledger_frames = DataFrame[]
    window_frames = DataFrame[]

    for ref in sc_policy_runs()
        model = ref.name
        books = books_by_model[model]
        report = reports_by_model[model]
        for spec in MT_POLICY_SPECS
            result = spec.name == "P0_flat_benchmark" ? baseline_by_model[model] :
                SC_PF.simulate_portfolio(
                    mt_selection_policy(spec.tau1, spec.tau2, spec.tau3),
                    books,
                    report;
                    initial_bankroll = SC_INITIAL_BANKROLL,
                    bootstrap = false,
                )
            daily = sc_policy_daily_frame(result, model, spec.name)
            ledger = sc_simulation_ledger(result, books, model, spec.name)
            all_dates = sort(unique(Date.(daily.date)))
            push!(summary_rows, mt_policy_summary_row(result, daily, ledger, model, spec.name;
                                                       tau1 = spec.tau1, tau2 = spec.tau2,
                                                       tau3 = spec.tau3))
            push!(daily_frames, daily)
            push!(ledger_frames, ledger)
            push!(window_frames, mt_window_summary(daily, model, spec.name, all_dates;
                                                    tau1 = spec.tau1, tau2 = spec.tau2,
                                                    tau3 = spec.tau3))
            @printf("  %-28s %-22s bank=%8.2f ret=%+8.2f%% Sharpe=%6.3f MDD=%7.2f%% bets=%4d\n",
                    model, spec.name, result.summary.final_bankroll,
                    result.summary.total_return_pct, result.summary.sharpe_ann,
                    result.summary.mdd, result.summary.n_bets)
        end
    end

    summary = DataFrame(summary_rows)
    daily = vcat(daily_frames...; cols = :union)
    ledger = vcat(ledger_frames...; cols = :union)
    windows = vcat(window_frames...; cols = :union)
    selection_summary = mt_selection_summary(ledger)
    definitions = mt_policy_definition_rows()
    builds = DataFrame(build_rows)

    println("\n[3/5] Running the 2D Pareto grid sweep " *
            "($(length(MT_TAU1_GRID))x$(length(MT_TAU2_GRID)) x 2 models)...")
    grid_rows = NamedTuple[]
    for ref in sc_policy_runs()
        model = ref.name
        books = books_by_model[model]
        report = reports_by_model[model]
        for tau1 in MT_TAU1_GRID, tau2 in MT_TAU2_GRID
            result = SC_PF.simulate_portfolio(
                mt_selection_policy(tau1, tau2, 0.0),
                books,
                report;
                initial_bankroll = SC_INITIAL_BANKROLL,
                bootstrap = false,
            )
            daily_grid = sc_policy_daily_frame(result, model, "P5_grid_sweep")
            push!(grid_rows, mt_grid_row(model, tau1, tau2, result, daily_grid))
        end
        @printf("  %-28s grid complete: %d cells\n", model, length(MT_TAU1_GRID) * length(MT_TAU2_GRID))
    end
    grid = DataFrame(grid_rows)

    println("\n[4/5] Validating reproducibility...")
    mt_validate(summary, daily, ledger, definitions, grid)
    println("  All policy/grid ledgers reconcile against engine summaries.")

    println("\n[5/5] Writing reproducibility artefacts...")
    outputs = [
        "multitier_policy_summary.csv" => summary,
        "multitier_policy_windows.csv" => windows,
        "multitier_policy_daily.csv" => daily,
        "multitier_policy_ledger.csv" => ledger,
        "multitier_selection_summary.csv" => selection_summary,
        "multitier_grid_sweep.csv" => grid,
        "multitier_policy_definitions.csv" => definitions,
        "multitier_build_report.csv" => builds,
    ]
    for (filename, frame) in outputs
        path = joinpath(MT_OUTPUT_DIR, filename)
        CSV.write(path, frame)
        println("  ", path)
    end

    return (; summary, windows, daily, ledger, selection_summary, grid, definitions, builds)
end

if abspath(PROGRAM_FILE) == @__FILE__
    eda_multitier_trust()
end
