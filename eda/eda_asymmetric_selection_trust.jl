# ==============================================================================
# eda_asymmetric_selection_trust.jl — directional SelectionTrust across six markets
# ==============================================================================
#
# This EDA reuses the exact six-market books from the stochastic-control audit and changes only
# the SelectionTrust table. It compares symmetric market-level pruning with directional gates:
# keeping Under 2.5 while pruning Over 2.5, expanding to other unders, and a descriptive
# "pure-alpha" subset selected by positive pooled information ratio on fold-held-out predictions.
#
# P_pure_alpha is intentionally labelled data-selected. Every match prediction is out of sample
# to its model fold, but the same 100 settlement dates are used to select and score the directional
# subset. Its full-period result is therefore a diagnostic upper bound, not production evidence.
# The four predeclared policies are the clean A/B evidence.
#
# This runner launches no MCMC. Saved-fit compatibility may require execution at repository
# commit 784c8ea81328760e75498b19d13c2dab762bde8e.
#
# Usage:
#   julia --project -t 8 eda/eda_asymmetric_selection_trust.jl
#
# Outputs:
#   eda/results/asymmetric_trust/asymmetric_policy_summary.csv
#   eda/results/asymmetric_trust/asymmetric_policy_windows.csv
#   eda/results/asymmetric_trust/asymmetric_policy_daily.csv
#   eda/results/asymmetric_trust/asymmetric_policy_ledger.csv
#   eda/results/asymmetric_trust/asymmetric_selection_summary.csv
#   eda/results/asymmetric_trust/asymmetric_pure_alpha_scores.csv
#   eda/results/asymmetric_trust/asymmetric_subset_search.csv
#   eda/results/asymmetric_trust/asymmetric_policy_definitions.csv
#   eda/results/asymmetric_trust/asymmetric_build_report.csv

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

const AS_OUTPUT_DIR = joinpath(@__DIR__, "results", "asymmetric_trust")
const AS_TRUST = 0.30

const AS_SELECTION_CATALOG = [
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

const AS_ALL_FAMILIES = Set(row.family for row in AS_SELECTION_CATALOG)
const AS_1X2_FAMILIES = Set(["1X2_home", "1X2_draw", "1X2_away"])
const AS_SYMMETRIC_CORE_FAMILIES = union(AS_1X2_FAMILIES,
    Set(["O/U 2.5_over_25", "O/U 2.5_under_25"]))
const AS_ASYMMETRIC_CORE_FAMILIES = union(AS_1X2_FAMILIES,
    Set(["O/U 2.5_under_25"]))
const AS_UNDER_EXPANSION_FAMILIES = union(AS_1X2_FAMILIES,
    Set(["O/U 1.5_under_15", "O/U 2.5_under_25", "O/U 3.5_under_35"]))

function as_selection_policy(active_families::AbstractSet{<:AbstractString})
    unknown = setdiff(Set(String.(collect(active_families))), AS_ALL_FAMILIES)
    isempty(unknown) || error("Unknown directional families: $(join(sort(collect(unknown)), ", ")).")
    table = Dict{Tuple{String,Float64,Symbol},Float64}()
    for row in AS_SELECTION_CATALOG
        table[(row.group, row.line, row.selection)] = row.family in active_families ? AS_TRUST : 0.0
    end
    return PolicySpec(
        trust = SC_PF.SelectionTrust(table; strict = true),
        risk = SlateDrawdown(23.0),
        cap = FixedCap(SC_CAP),
        grouping = DailySlate(),
    )
end

as_baseline_policy() = sc_status_quo_policy()
as_symmetric_core_policy() = as_selection_policy(AS_SYMMETRIC_CORE_FAMILIES)
as_asymmetric_core_policy() = as_selection_policy(AS_ASYMMETRIC_CORE_FAMILIES)
as_under_expansion_policy() = as_selection_policy(AS_UNDER_EXPANSION_FAMILIES)
as_pure_alpha_policy(active) = as_selection_policy(active)

function as_annualized_information_ratio(dates::Vector{Date}, returns::Vector{Float64})
    length(dates) == length(returns) || error("Information-ratio dates and returns are misaligned.")
    length(dates) >= 2 || return NaN
    span_days = Dates.value(dates[end] - dates[1])
    span_days > 0 || return NaN
    sigma = std(returns)
    sigma > 0 || return NaN
    slates_per_year = length(dates) * 365.25 / span_days
    return mean(returns) / sigma * sqrt(slates_per_year)
end

function as_directional_scores(baseline_ledgers::Dict{String,DataFrame})
    model_names = sort(collect(keys(baseline_ledgers)))
    all_dates = sort(unique(vcat([Date.(baseline_ledgers[m].date) for m in model_names]...)))
    families = sort(collect(AS_ALL_FAMILIES))
    rows = NamedTuple[]
    pooled_matrix = zeros(length(all_dates), length(families))

    for (family_idx, family) in enumerate(families)
        model_daily = Dict{String,Vector{Float64}}()
        for model in model_names
            frame = baseline_ledgers[model]
            selected = frame[frame.family .== family, :]
            by_date = isempty(selected) ? DataFrame(date = Date[], pnl = Float64[]) :
                combine(groupby(selected, :date), :pnl => sum => :pnl)
            pnl_map = Dict(Date(row.date) => Float64(row.pnl) for row in eachrow(by_date))
            returns = [get(pnl_map, date, 0.0) for date in all_dates]
            model_daily[model] = returns
            stake = sum(selected.stake_amount)
            pnl = sum(selected.pnl_amount)
            push!(rows, (
                scope = model,
                family,
                n_slates = length(all_dates),
                n_bets = nrow(selected),
                stake_amount = stake,
                pnl_amount = pnl,
                roi_pct = stake > 0 ? 100 * pnl / stake : NaN,
                oos_information_ratio = as_annualized_information_ratio(all_dates, returns),
                selected_for_pure_alpha = false,
            ))
        end

        pooled_returns = [mean(model_daily[model][idx] for model in model_names)
                          for idx in eachindex(all_dates)]
        pooled_matrix[:, family_idx] .= pooled_returns
        pooled_frame = vcat([baseline_ledgers[model][baseline_ledgers[model].family .== family, :]
                             for model in model_names]...; cols = :union)
        pooled_stake = sum(pooled_frame.stake_amount)
        pooled_pnl = sum(pooled_frame.pnl_amount)
        push!(rows, (
            scope = "POOLED",
            family,
            n_slates = length(all_dates),
            n_bets = nrow(pooled_frame),
            stake_amount = pooled_stake,
            pnl_amount = pooled_pnl,
            roi_pct = pooled_stake > 0 ? 100 * pooled_pnl / pooled_stake : NaN,
            oos_information_ratio = as_annualized_information_ratio(all_dates, pooled_returns),
            selected_for_pure_alpha = false,
        ))
    end

    subset_rows = NamedTuple[]
    for mask in 1:(2^length(families) - 1)
        returns = zeros(length(all_dates))
        active = String[]
        for family_idx in eachindex(families)
            if ((mask >> (family_idx - 1)) & 1) == 1
                returns .+= pooled_matrix[:, family_idx]
                push!(active, families[family_idx])
            end
        end
        push!(subset_rows, (
            mask,
            n_active = length(active),
            families = join(active, "; "),
            oos_information_ratio = as_annualized_information_ratio(all_dates, returns),
            mean_daily_return = mean(returns),
            daily_volatility = std(returns),
        ))
    end
    subset_search = DataFrame(subset_rows)
    sort!(subset_search, :oos_information_ratio; rev = true)
    subset_search.rank = 1:nrow(subset_search)
    subset_search.selected_for_pure_alpha = falses(nrow(subset_search))
    subset_search.selected_for_pure_alpha[1] = true
    active = Set(String.(split(String(subset_search.families[1]), "; ")))
    isempty(active) && error("Subset information-ratio search selected no directional families.")

    scores = DataFrame(rows)
    scores.selected_for_pure_alpha .= [row.scope == "POOLED" && row.family in active
                                       for row in eachrow(scores)]
    return (; scores, subset_search, active, all_dates)
end

function as_window_metrics(daily::DataFrame, from::Date, to::Date)
    frame = daily[(Date.(daily.date) .>= from) .& (Date.(daily.date) .<= to), :]
    nrow(frame) >= 2 || return (
        n_slates = nrow(frame), return_pct = NaN, annual_sharpe = NaN,
        max_drawdown_pct = NaN, n_bets = sum(frame.n_bets), total_turnover = sum(frame.stake_frac),
    )
    pnl = Float64.(frame.pnl_frac)
    wealth = cumprod(1.0 .+ pnl)
    peak = accumulate(max, wealth)
    drawdown = 100 .* (wealth ./ peak .- 1.0)
    log_returns = log.(1.0 .+ pnl)
    span_days = Dates.value(Date(frame.date[end]) - Date(frame.date[1]))
    annual_sharpe = span_days > 0 && std(log_returns) > 0 ?
        mean(log_returns) / std(log_returns) * sqrt(nrow(frame) * 365.25 / span_days) : NaN
    return (
        n_slates = nrow(frame),
        return_pct = 100 * (wealth[end] - 1.0),
        annual_sharpe,
        max_drawdown_pct = minimum(drawdown),
        n_bets = sum(frame.n_bets),
        total_turnover = sum(frame.stake_frac),
    )
end

function as_window_summary(daily::DataFrame, model::AbstractString, policy::AbstractString,
                           all_dates::Vector{Date})
    split_date = all_dates[cld(length(all_dates), 2)]
    windows = [
        (name = "full", from = all_dates[1], to = all_dates[end]),
        (name = "first_half", from = all_dates[1], to = split_date),
        (name = "second_half", from = split_date + Day(1), to = all_dates[end]),
    ]
    rows = NamedTuple[]
    for window in windows
        metrics = as_window_metrics(daily, window.from, window.to)
        push!(rows, (; model = String(model), policy = String(policy), window = window.name,
                     date_from = window.from, date_to = window.to, metrics...))
    end
    return DataFrame(rows)
end

function as_selection_summary(ledger::DataFrame)
    rows = NamedTuple[]
    for group in groupby(ledger, [:model, :policy, :market_key, :family, :selection])
        stake = sum(group.stake_amount)
        pnl = sum(group.pnl_amount)
        push!(rows, (
            model = String(group.model[1]),
            policy = String(group.policy[1]),
            market_key = String(group.market_key[1]),
            family = String(group.family[1]),
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
    return DataFrame(rows)
end

function as_policy_definition_rows(pure_alpha_active::Set{String})
    policies = [
        ("P_baseline", AS_ALL_FAMILIES, "all six markets, symmetric 0.30 trust"),
        ("P_symmetric_core", AS_SYMMETRIC_CORE_FAMILIES,
         "1X2 and both O/U 2.5 directions"),
        ("P_asymmetric_core", AS_ASYMMETRIC_CORE_FAMILIES,
         "1X2 plus Under 2.5; Over 2.5 pruned"),
        ("P_under_expansion", AS_UNDER_EXPANSION_FAMILIES,
         "1X2 plus Under 1.5, Under 2.5 and Under 3.5"),
        ("P_pure_alpha", pure_alpha_active,
         "subset maximizing pooled information ratio across fold-held-out baseline returns"),
    ]
    rows = NamedTuple[]
    for (policy, active, description) in policies, selection in AS_SELECTION_CATALOG
        push!(rows, (
            policy,
            family = selection.family,
            market_key = selection.market_key,
            selection = String(selection.selection),
            trust = selection.family in active ? AS_TRUST : 0.0,
            active = selection.family in active,
            data_selected = policy == "P_pure_alpha",
            description,
        ))
    end
    return DataFrame(rows)
end

function as_build_report_row(ref::SCCanonicalRun, report, fit)
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

function as_validate(summary::DataFrame, daily::DataFrame, ledger::DataFrame,
                     definitions::DataFrame)
    expected_policies = Set(["P_baseline", "P_symmetric_core", "P_asymmetric_core",
                             "P_under_expansion", "P_pure_alpha"])
    Set(summary.policy) == expected_policies || error("Policy summary is incomplete.")
    nrow(summary) == 10 || error("Expected ten model-policy summary rows; found $(nrow(summary)).")
    all(summary.n_slates .== 100) || error("Every policy must cover exactly 100 slate dates.")
    nrow(definitions) == 5 * length(AS_SELECTION_CATALOG) || error(
        "Policy definition table does not cover every directional selection.")

    for row in eachrow(summary)
        selected_daily = daily[(daily.model .== row.model) .& (daily.policy .== row.policy), :]
        selected_ledger = ledger[(ledger.model .== row.model) .& (ledger.policy .== row.policy), :]
        nrow(selected_daily) == row.n_slates || error("$(row.model)/$(row.policy) daily row mismatch.")
        nrow(selected_ledger) == row.n_bets || error("$(row.model)/$(row.policy) ledger row mismatch.")
        isapprox(selected_daily.bankroll_close[end], row.final_bankroll; atol = 1e-9, rtol = 1e-12) ||
            error("$(row.model)/$(row.policy) final bankroll does not reproduce summary.")
        isapprox(sum(selected_daily.stake_frac), row.total_turnover; atol = 1e-12, rtol = 1e-12) ||
            error("$(row.model)/$(row.policy) turnover does not reproduce summary.")
        allowed = Set(String.(definitions.family[(definitions.policy .== row.policy) .&
                                                 definitions.active]))
        isempty(setdiff(Set(String.(selected_ledger.family)), allowed)) || error(
            "$(row.model)/$(row.policy) staked a gated selection.")
    end
    return nothing
end

function eda_asymmetric_selection_trust()
    println("\n", "="^122)
    println(" ASYMMETRIC SELECTION-LEVEL TRUST — SIX-MARKET DIRECTIONAL POLICY A/B")
    println("="^122)
    mkpath(AS_OUTPUT_DIR)

    inventory = sc_verify_run_inventory()
    println("  Run inventory verified: $(nrow(inventory)) immutable addresses")
    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
    betfair = sc_betfair_closing_odds(ds)
    book_spec = sc_book_spec()

    baseline_results = Dict{String,Any}()
    books_by_model = Dict{String,Any}()
    reports_by_model = Dict{String,Any}()
    fits_by_model = Dict{String,Any}()
    baseline_ledgers = Dict{String,DataFrame}()
    build_rows = NamedTuple[]

    println("\n[1/4] Building each model's six-market books once...")
    for ref in sc_policy_runs()
        fit = sc_load_fit(ref)
        result, books, report = run_portfolio_simulation(
            book_spec,
            as_baseline_policy(),
            fit,
            betfair,
            ds;
            initial_bankroll = SC_INITIAL_BANKROLL,
            bootstrap = false,
            require_converged = false,
            quiet = true,
        )
        fits_by_model[ref.name] = fit
        baseline_results[ref.name] = result
        books_by_model[ref.name] = books
        reports_by_model[ref.name] = report
        baseline_ledgers[ref.name] = sc_simulation_ledger(result, books, ref.name, "P_baseline")
        push!(build_rows, as_build_report_row(ref, report, fit))
        @printf("  %-28s books=%3d slates=%3d baseline=%+8.2f%% Sharpe=%6.3f\n",
                ref.name, length(books), result.summary.n_slates,
                result.summary.total_return_pct, result.summary.sharpe_ann)
    end

    println("\n[2/4] Selecting the descriptive maximum-information-ratio subset...")
    pure_alpha = as_directional_scores(baseline_ledgers)
    println("  P_pure_alpha = ", join(sort(collect(pure_alpha.active)), ", "))
    definitions = as_policy_definition_rows(pure_alpha.active)

    policy_factories = [
        ("P_baseline", () -> as_baseline_policy()),
        ("P_symmetric_core", () -> as_symmetric_core_policy()),
        ("P_asymmetric_core", () -> as_asymmetric_core_policy()),
        ("P_under_expansion", () -> as_under_expansion_policy()),
        ("P_pure_alpha", () -> as_pure_alpha_policy(pure_alpha.active)),
    ]

    println("\n[3/4] Simulating directional policies...")
    summary_rows = NamedTuple[]
    daily_frames = DataFrame[]
    ledger_frames = DataFrame[]
    window_frames = DataFrame[]

    for ref in sc_policy_runs()
        model = ref.name
        books = books_by_model[model]
        report = reports_by_model[model]
        for (policy_name, factory) in policy_factories
            result = policy_name == "P_baseline" ? baseline_results[model] :
                SC_PF.simulate_portfolio(
                    factory(),
                    books,
                    report;
                    initial_bankroll = SC_INITIAL_BANKROLL,
                    bootstrap = false,
                )
            daily = sc_policy_daily_frame(result, model, policy_name)
            ledger = policy_name == "P_baseline" ? baseline_ledgers[model] :
                sc_simulation_ledger(result, books, model, policy_name)
            push!(summary_rows, sc_policy_summary(result, model, policy_name))
            push!(daily_frames, daily)
            push!(ledger_frames, ledger)
            push!(window_frames, as_window_summary(daily, model, policy_name, pure_alpha.all_dates))
            @printf("  %-28s %-22s bank=%8.2f ret=%+8.2f%% Sharpe=%6.3f MDD=%7.2f%% bets=%4d\n",
                    model, policy_name, result.summary.final_bankroll,
                    result.summary.total_return_pct, result.summary.sharpe_ann,
                    result.summary.mdd, result.summary.n_bets)
        end
    end

    summary = DataFrame(summary_rows)
    daily = vcat(daily_frames...; cols = :union)
    ledger = vcat(ledger_frames...; cols = :union)
    windows = vcat(window_frames...; cols = :union)
    selection_summary = as_selection_summary(ledger)
    builds = DataFrame(build_rows)
    as_validate(summary, daily, ledger, definitions)

    println("\n[4/4] Writing reproducibility artefacts...")
    outputs = [
        "asymmetric_policy_summary.csv" => summary,
        "asymmetric_policy_windows.csv" => windows,
        "asymmetric_policy_daily.csv" => daily,
        "asymmetric_policy_ledger.csv" => ledger,
        "asymmetric_selection_summary.csv" => selection_summary,
        "asymmetric_pure_alpha_scores.csv" => pure_alpha.scores,
        "asymmetric_subset_search.csv" => pure_alpha.subset_search,
        "asymmetric_policy_definitions.csv" => definitions,
        "asymmetric_build_report.csv" => builds,
    ]
    for (filename, frame) in outputs
        path = joinpath(AS_OUTPUT_DIR, filename)
        CSV.write(path, frame)
        println("  ", path)
    end

    return (; summary, windows, daily, ledger, selection_summary,
              pure_alpha_scores = pure_alpha.scores, definitions, builds)
end

if abspath(PROGRAM_FILE) == @__FILE__
    eda_asymmetric_selection_trust()
end
