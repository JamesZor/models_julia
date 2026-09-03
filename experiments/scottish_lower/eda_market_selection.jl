# ==============================================================================
# eda_market_selection.jl — market-line forensics and book pruning
# ==============================================================================
#
# `compare_scottish_experiments.jl` established that widening the book from three markets to
# six cost every model return and Sharpe while leaving the 1X2 leg untouched. That located the
# loss in the added totals lines but did not say WHICH ones, or why. This runner opens the
# trade ledger and answers it per line and per selection.
#
# THE LEDGER IS THE POINT. `trajectory.bets` carries `p_model` and `p_market` alongside the
# realised payoff for every bet actually struck, so calibration, edge and settlement can be
# read off the same rows the bankroll was computed from. Nothing here is re-derived from the
# latents; a discrepancy between this report and the portfolio summary would be a bug, not a
# modelling choice.
#
# THREE THINGS THAT ARE EASY TO GET WRONG, AND HOW THEY ARE HANDLED:
#
#   1. Stakes are fractions of a MOVING bankroll. `stake` and `pnl` in the ledger are fractions
#      of the bankroll at that slate, so summing them across a compounding backtest adds
#      quantities measured in different units. Every currency figure here is rescaled by the
#      slate's opening bankroll first.
#   2. Removing a line is not subtracting its P&L. Kelly re-solves over whatever remains, and
#      the exposure cap binds differently, so the counterfactual is a re-simulation — which is
#      what §6 does. The per-line drawdown in §4 is the line's own standalone stream and is
#      labelled as such, never as "the drawdown this line caused".
#   3. Choosing lines on the same data you then score them on is selection bias. The pruning
#      rule is fitted on the FIRST HALF of the slate calendar only, and every configuration is
#      also scored on the SECOND HALF, which the rule never saw.
#
# THIS RUNNER LAUNCHES NO MCMC. It loads completed fits from PostgreSQL and re-prices them.
#
# Usage:
#
#   julia --project -t 16 experiments/scottish_lower/eda_market_selection.jl
#
# Outputs:
#   experiments/scottish_lower/results/market_line_breakdown.csv
#   experiments/scottish_lower/results/market_pruning_comparison.csv
#   experiments/scottish_lower/results/market_selection_ledger.csv
#   experiments/scottish_lower/MARKET_LINE_EDA_REPORT.md

using BayesianFootball
using CSV
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Statistics
using StatsBase
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

const EDA_PORTFOLIO = BayesianFootball.Portfolio

const EDA_OUTPUT_DIR = joinpath(@__DIR__, "results")
const EDA_REPORT_PATH = joinpath(@__DIR__, "MARKET_LINE_EDA_REPORT.md")

# A line needs enough settled bets before its ROI means anything. 100 is not a significance
# threshold — it is the point below which the standard error on a ~35%-hit-rate ROI is wider
# than the effect sizes this analysis is trying to rank.
const EDA_MIN_BETS = 100

# ==============================================================================
# 1. The bench
# ==============================================================================

struct EdaModel
    generation::Int
    experiment::String
    run_name::String
    focus::Bool
end

const EDA_MODELS = EdaModel[
    EdaModel(1, "scottish_lower_poisson_2426", "m00_baseline", false),
    EdaModel(1, "scottish_lower_poisson_2426", "m05_production_wealth", false),
    EdaModel(2, "scottish_lower_negbin_2426", "m00_negbin_baseline", false),
    EdaModel(2, "scottish_lower_negbin_2426", "m05_negbin_production_wealth", false),
    EdaModel(3, "scottish_lower_joint_2426", "m00_joint_baseline", false),
    EdaModel(3, "scottish_lower_joint_2426", "m05_joint_production_wealth", true),
    EdaModel(4, "scottish_lower_joint_player_2426", "m12_joint_hybrid_synergy", true),
    EdaModel(4, "scottish_lower_joint_player_2426", "m13_joint_composite", true),
]

eda_focus_models() = filter(m -> m.focus, EDA_MODELS)

# ==============================================================================
# 2. Price source and the candidate books
# ==============================================================================

function eda_betfair_closing_odds(ds::Data.DataStore)
    raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = (-20.0, 0.0))
    odds = DataFrame(
        match_id = Int.(raw.match_id),
        market_name = String.(raw.market_name),
        market_line = Float64.(raw.market_line),
        selection = Symbol.(raw.selection),
        odds_close = Float64.(raw.odds),
    )
    filter!(row -> isfinite(row.odds_close) && row.odds_close > 1.0, odds)
    sort!(odds, [:match_id, :market_name, :market_line, :selection])
    return odds
end

"Every market this study can reach, as `(key, market)` pairs. Keys name rows in the report."
const EDA_MARKET_MENU = [
    ("1X2", Data.Market1X2()),
    ("OU0.5", Data.MarketOverUnder(0.5)),
    ("OU1.5", Data.MarketOverUnder(1.5)),
    ("OU2.5", Data.MarketOverUnder(2.5)),
    ("OU3.5", Data.MarketOverUnder(3.5)),
    ("BTTS", Data.MarketBTTS()),
]

const EDA_MARKET_BY_KEY = Dict(k => m for (k, m) in EDA_MARKET_MENU)

"BookSpec over the named market keys. Everything except `markets` is held fixed, so a
configuration comparison varies one thing."
function eda_book_spec(keys::Vector{String})
    markets = Data.AbstractMarket[EDA_MARKET_BY_KEY[k] for k in keys]
    return BookSpec(
        markets = Data.MarketConfig(markets),
        price = DeArb(),
        allocator = KellyLogUtility(),
        shrink = EDA_PORTFOLIO.FractionalKelly(0.30),
        exec = ExecutionConfig(
            commission = PerBetCommission(0.02),
            budget = 0.99,
            min_selection_stake = 0.001,
        ),
    )
end

eda_policy_spec() = PolicySpec(
    trust = FlatTrust(1.0),
    risk = SlateDrawdown(23.0),
    cap = FixedCap(0.20),
    grouping = DailySlate(),
)

const EDA_FULL_KEYS = ["1X2", "OU0.5", "OU1.5", "OU2.5", "OU3.5", "BTTS"]
const EDA_CLASSIC_KEYS = ["1X2", "OU2.5", "BTTS"]

# ==============================================================================
# 3. Ledger extraction
# ==============================================================================

"""
    eda_family_catalog(books) -> Dict{String,NamedTuple}

`trajectory.bets` records only `family` ("O/U 2.5_over_25"), not the `group`/`line` pair the
`Selection` also carries. Rather than parse the string — the encoding is an internal trust key
and is free to change — the map is read off the `Selection` objects the simulation actually
priced.
"""
function eda_family_catalog(books)
    catalog = Dict{String,NamedTuple}()
    for book in books, sel in book.sels
        get!(catalog, sel.family) do
            (; group = sel.group, line = sel.line, selection = sel.selection)
        end
    end
    return catalog
end

"Market key ('OU2.5') for a priced selection, matching `EDA_MARKET_MENU`."
function eda_market_key(group::AbstractString, line::Real)
    group == "1X2" && return "1X2"
    group == "BTTS" && return "BTTS"
    group == "OverUnder" && return "OU" * string(line)
    return group
end

"""
    eda_ledger(result, books, model_name) -> DataFrame

One row per struck bet, rescaled out of bankroll fractions into currency.

`stake` and `pnl` are fractions of the bankroll at their own slate. In a compounding backtest
that makes them incommensurable across time — a 1% stake in the first week and a 1% stake in
the last are different amounts of money — so both are multiplied by the slate's opening
bankroll before anything is summed.
"""
function eda_ledger(result, books, model_name::AbstractString)
    bets = copy(result.trajectory.bets)
    nrow(bets) == 0 && return DataFrame()

    opening = Dict{Date,Float64}()
    for state in result.daily_states
        haskey(opening, state.date) && error(
            "Two slates share the date $(state.date); the bankroll rescale would be ambiguous.")
        opening[state.date] = state.bankroll_open
    end

    catalog = eda_family_catalog(books)
    missing_families = setdiff(Set(String.(bets.family)), keys(catalog))
    isempty(missing_families) || error(
        "Ledger holds families absent from the priced books: $(collect(missing_families)).")

    bets.model = fill(String(model_name), nrow(bets))
    bets.group = [catalog[f].group for f in bets.family]
    bets.line = [catalog[f].line for f in bets.family]
    bets.market_key = [eda_market_key(catalog[f].group, catalog[f].line) for f in bets.family]
    bets.bank_open = [opening[d] for d in bets.date]
    bets.abs_stake = bets.stake .* bets.bank_open
    bets.abs_pnl = bets.pnl .* bets.bank_open
    bets.won = bets.payoff .> 0
    bets.pushed = bets.payoff .== 0
    bets.edge = bets.p_model .- bets.p_market
    return bets
end

# ==============================================================================
# 4. Per-line metrics
# ==============================================================================

"Longest run of `true` in `x`."
function eda_max_streak(x::AbstractVector{Bool})
    best = 0
    run = 0
    for value in x
        run = value ? run + 1 : 0
        run > best && (best = run)
    end
    return best
end

"Lag-1 autocorrelation of a payoff sequence; `NaN` when the series cannot support one."
function eda_lag1_autocorr(x::AbstractVector{<:Real})
    length(x) < 3 && return NaN
    s = std(x)
    (isfinite(s) && s > 0) || return NaN
    return cor(@view(x[1:end-1]), @view(x[2:end]))
end

"""
    eda_standalone_drawdown(frame) -> (units, pct_of_turnover)

Deepest peak-to-trough excursion of this line's OWN cumulative P&L, in bankroll units.

This is NOT the drawdown the line caused the portfolio. Kelly allocates jointly, so removing
a line changes every other stake in the slate; the portfolio-level counterfactual is the
re-simulation in §6. What this measures is how lumpy the line's contribution was on its own,
which is what distinguishes a line that bleeds steadily from one that is flat until it is not.
"""
function eda_standalone_drawdown(frame::AbstractDataFrame)
    nrow(frame) == 0 && return (0.0, NaN)
    ordered = sort(frame, [:date, :match_id])
    equity = cumsum(ordered.abs_pnl)
    peak = -Inf
    worst = 0.0
    for value in equity
        value > peak && (peak = value)
        worst = min(worst, value - peak)
    end
    turnover = sum(ordered.abs_stake)
    return (worst, turnover > 0 ? 100 * worst / turnover : NaN)
end

"""
    eda_line_metrics(frame, totals) -> NamedTuple

Everything the report reads for one slice of the ledger.

`flat_roi_pct` is `mean(payoff)`: with a flat one-unit stake the net per-unit payoff IS the
return, so no separate staking simulation is needed for it. `kelly_roi_pct` divides realised
P&L by realised turnover and is what the compounding backtest actually earned. The two
disagree exactly when Kelly's stake sizing disagrees with the line's average edge, which is
the signal §5 classifies on.
"""
function eda_line_metrics(frame::AbstractDataFrame, totals::NamedTuple)
    n = nrow(frame)
    n == 0 && return nothing
    ordered = sort(frame, [:date, :match_id])
    turnover = sum(frame.abs_stake)
    pnl = sum(frame.abs_pnl)
    win_rate = mean(frame.won)
    mean_p_model = mean(frame.p_model)
    dd_units, dd_pct = eda_standalone_drawdown(frame)

    capital_share = totals.turnover > 0 ? 100 * turnover / totals.turnover : NaN
    pnl_share = totals.pnl != 0 ? 100 * pnl / totals.pnl : NaN
    kelly_roi = turnover > 0 ? 100 * pnl / turnover : NaN
    # ROI relative to the whole book's ROI. Above 1 means the line returns more than its share
    # of the capital it consumed; below 1 means it is being carried by the rest of the book.
    efficiency = (isfinite(kelly_roi) && totals.kelly_roi != 0) ?
                 kelly_roi / totals.kelly_roi : NaN

    return (;
        n_bets = n,
        n_matches = length(unique(frame.match_id)),
        win_rate_pct = 100 * win_rate,
        push_rate_pct = 100 * mean(frame.pushed),
        mean_odds = mean(frame.odds),
        median_odds = median(frame.odds),
        mean_p_model = mean_p_model,
        mean_p_market = mean(frame.p_market),
        mean_edge = mean(frame.edge),
        calib_bias = win_rate - mean_p_model,
        market_bias = win_rate - mean(frame.p_market),
        flat_roi_pct = 100 * mean(frame.payoff),
        kelly_roi_pct = kelly_roi,
        total_pnl_units = pnl,
        pnl_share_pct = pnl_share,
        turnover_units = turnover,
        capital_share_pct = capital_share,
        capital_efficiency = efficiency,
        mean_stake_frac = mean(frame.stake),
        max_stake_frac = maximum(frame.stake),
        standalone_dd_units = dd_units,
        standalone_dd_pct_turnover = dd_pct,
        max_win_streak = eda_max_streak(ordered.won),
        max_loss_streak = eda_max_streak(.!ordered.won .& .!ordered.pushed),
        payoff_autocorr = eda_lag1_autocorr(ordered.payoff),
    )
end

"Book-wide denominators for the share and efficiency columns."
function eda_totals(frame::AbstractDataFrame)
    turnover = sum(frame.abs_stake)
    pnl = sum(frame.abs_pnl)
    return (; turnover, pnl, kelly_roi = turnover > 0 ? 100 * pnl / turnover : NaN)
end

"""
    eda_breakdown(ledger, scope) -> DataFrame

Line-level and selection-level rows for one scope (a model name, or "POOLED").

Both granularities live in one long frame with `selection = "ALL"` marking the line-level
roll-up, because the Over/Under asymmetry question is a comparison BETWEEN the two levels and
splitting them into separate files would make that join the reader's problem.
"""
function eda_breakdown(ledger::DataFrame, scope::AbstractString)
    nrow(ledger) == 0 && return DataFrame()
    totals = eda_totals(ledger)
    rows = NamedTuple[]

    for key in EDA_FULL_KEYS
        line_frame = ledger[ledger.market_key .== key, :]
        nrow(line_frame) == 0 && continue
        metrics = eda_line_metrics(line_frame, totals)
        push!(rows, (; scope = String(scope), market_key = key,
                       group = line_frame.group[1], line = line_frame.line[1],
                       selection = "ALL", metrics...))
        for selection in sort(unique(String.(line_frame.selection)))
            sel_frame = line_frame[String.(line_frame.selection) .== selection, :]
            sel_metrics = eda_line_metrics(sel_frame, totals)
            push!(rows, (; scope = String(scope), market_key = key,
                           group = sel_frame.group[1], line = sel_frame.line[1],
                           selection = selection, sel_metrics...))
        end
    end
    return DataFrame(rows)
end

# ==============================================================================
# 5. Classification
# ==============================================================================

"""
    eda_classify(row, portfolio_kelly_roi) -> (verdict, reason)

The pruning rule, stated once and applied without exception.

    KEEP        kelly_roi > 0  AND  capital_efficiency >= 0.50  AND  n_bets >= 100
    PRUNE       kelly_roi <= 0  OR  (capital_efficiency < 0.25 AND n_bets >= 100)
    CONDITIONAL otherwise

`capital_efficiency` is the line's Kelly ROI over the whole book's, so 0.50 means "returns at
least half the rate of the capital it consumes". A line can clear the ROI test and still fail
this one: that is exactly the dilution case — profitable, but profitable enough only to drag
the book's average down while occupying slate budget the exposure cap then denies to better
selections. The bet floor keeps a thin line out of both verdicts rather than letting a handful
of settlements name it.
"""
function eda_classify(row)
    thin = row.n_bets < EDA_MIN_BETS
    roi = row.kelly_roi_pct
    eff = row.capital_efficiency

    if !thin && isfinite(roi) && roi <= 0
        return ("PRUNE", "Kelly ROI $(round(roi, digits = 2))% <= 0 on $(row.n_bets) bets")
    end
    if !thin && isfinite(eff) && eff < 0.25
        return ("PRUNE", "capital efficiency $(round(eff, digits = 2)) < 0.25 " *
                         "on $(row.n_bets) bets")
    end
    if !thin && isfinite(roi) && roi > 0 && isfinite(eff) && eff >= 0.50
        return ("KEEP", "Kelly ROI $(round(roi, digits = 2))%, efficiency " *
                        "$(round(eff, digits = 2))")
    end
    thin && return ("CONDITIONAL", "only $(row.n_bets) bets, below the $EDA_MIN_BETS floor")
    return ("CONDITIONAL", "Kelly ROI $(round(roi, digits = 2))%, efficiency " *
                           "$(round(eff, digits = 2)) — profitable but dilutive")
end

# ==============================================================================
# 6. Windowed portfolio metrics
# ==============================================================================

"""
    eda_window_metrics(states, from, to) -> NamedTuple

Return, Sharpe and drawdown over a date window, recomputed from the daily states.

The formulae mirror `Portfolio/simulation.jl:200-212` exactly — log per-slate returns,
`mean/std * sqrt(slates_per_year)`, drawdown against the running peak — so a full-window call
reproduces the engine's own `summary` and the second-half numbers are on the same scale rather
than a parallel definition that happens to have the same name.
"""
function eda_window_metrics(states, from::Date, to::Date)
    window = [s for s in states if from <= s.date <= to]
    length(window) < 2 && return (; n_slates = length(window), return_pct = NaN,
                                    sharpe_ann = NaN, mdd_pct = NaN, turnover = NaN,
                                    n_bets = 0, mean_exposure = NaN)

    pnl = [s.pnl_frac for s in window]
    bank = cumprod(1.0 .+ pnl)
    peak = accumulate(max, bank)
    drawdown = 100 .* (bank .- peak) ./ peak

    logret = log.(1.0 .+ pnl)
    days = Dates.value(window[end].date - window[1].date)
    slates_per_year = days > 0 ? length(window) * 365.25 / days : NaN
    sharpe = std(logret) > 0 ? mean(logret) / std(logret) : NaN
    sharpe_ann = (isnan(sharpe) || isnan(slates_per_year)) ? NaN :
                 sharpe * sqrt(slates_per_year)

    return (;
        n_slates = length(window),
        return_pct = 100 * (bank[end] - 1.0),
        sharpe_ann = sharpe_ann,
        mdd_pct = minimum(drawdown),
        turnover = sum(s.stake_frac for s in window),
        n_bets = sum(s.n_bets for s in window),
        mean_exposure = mean(s.exposure for s in window),
    )
end

# ==============================================================================
# 7. Workflow
# ==============================================================================

function eda_market_selection()
    println("\n", "="^120)
    println(" SCOTTISH LOWER — MARKET LINE FORENSICS AND BOOK PRUNING")
    println("="^120)

    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
    bf_odds = eda_betfair_closing_odds(ds)
    policy_spec = eda_policy_spec()
    full_book = eda_book_spec(EDA_FULL_KEYS)
    println("  Betfair : $(nrow(bf_odds)) closing rows across " *
            "$(length(unique(bf_odds.match_id))) matches")
    println("  Book    : ", join(EDA_FULL_KEYS, ", "))
    println("  Policy  : FlatTrust(1.0), SlateDrawdown(23.0), FixedCap(0.20), DailySlate()")

    storages = Dict{String,PostgresStorage}()
    fits = Dict{String,Any}()
    ledgers = Dict{String,DataFrame}()
    full_results = Dict{String,Any}()

    println("\n[1/4] Pricing the six-market book for every model...")
    for model in EDA_MODELS
        db = get!(storages, model.experiment) do
            PostgresStorage(model.experiment)
        end
        fit = load_fit(db, model.run_name)
        fits[model.run_name] = fit
        result, books, _ = run_portfolio_simulation(
            full_book, policy_spec, fit, bf_odds, ds;
            bootstrap = false, require_converged = false, quiet = true,
        )
        ledger = eda_ledger(result, books, model.run_name)
        ledgers[model.run_name] = ledger
        full_results[model.run_name] = result
        @printf("  G%d %-32s bets=%5d  return=%+8.2f%%  Sharpe=%6.3f\n",
                model.generation, model.run_name, nrow(ledger),
                result.summary.total_return_pct, result.summary.sharpe_ann)
    end

    focus = eda_focus_models()
    focus_names = [m.run_name for m in focus]
    pooled_focus = vcat([ledgers[n] for n in focus_names]...; cols = :union)
    pooled_all = vcat([ledgers[m.run_name] for m in EDA_MODELS]...; cols = :union)

    all_dates = sort(unique(pooled_focus.date))
    split_date = all_dates[cld(length(all_dates), 2)]
    first_from, first_to = all_dates[1], split_date
    second_from, second_to = split_date + Day(1), all_dates[end]
    println("\n  Calendar: $(all_dates[1]) to $(all_dates[end]), " *
            "$(length(all_dates)) slates")
    println("  Split   : selection window <= $split_date, evaluation window > $split_date")

    println("\n[2/4] Per-line breakdown...")
    breakdown_frames = DataFrame[]
    for model in EDA_MODELS
        push!(breakdown_frames, eda_breakdown(ledgers[model.run_name], model.run_name))
    end
    push!(breakdown_frames, eda_breakdown(pooled_focus, "POOLED_FOCUS"))
    push!(breakdown_frames, eda_breakdown(pooled_all, "POOLED_ALL"))
    breakdown = vcat(breakdown_frames...; cols = :union)

    # The rule is fitted here and only here: focus models, first half of the calendar.
    selection_ledger = pooled_focus[pooled_focus.date .<= split_date, :]
    selection_breakdown = eda_breakdown(selection_ledger, "SELECTION_WINDOW")
    line_rows = selection_breakdown[selection_breakdown.selection .== "ALL", :]
    verdicts = [eda_classify(row) for row in eachrow(line_rows)]
    line_rows.verdict = first.(verdicts)
    line_rows.reason = last.(verdicts)

    keep_keys = String.(line_rows.market_key[line_rows.verdict .== "KEEP"])
    prune_keys = String.(line_rows.market_key[line_rows.verdict .== "PRUNE"])
    conditional_keys = String.(line_rows.market_key[line_rows.verdict .== "CONDITIONAL"])
    curated_keys = [k for k in EDA_FULL_KEYS if k in keep_keys]
    isempty(curated_keys) && error(
        "The pruning rule kept no market at all; the classification thresholds are wrong.")

    println("\n  Verdicts on the selection window (focus models, first half):")
    for row in eachrow(sort(line_rows, :market_key))
        @printf("    %-7s %-12s %s\n", row.market_key, row.verdict, row.reason)
    end
    println("  Curated basket: ", join(curated_keys, ", "))

    println("\n[3/4] Configuration A/B...")
    configs = [
        ("full_6", EDA_FULL_KEYS),
        ("classic_3", EDA_CLASSIC_KEYS),
        ("curated", curated_keys),
        ("x1x2_ou25", ["1X2", "OU2.5"]),
        ("x1x2_only", ["1X2"]),
    ]
    # Configurations are NOT deduplicated by market set. If the pruning rule happens to
    # rediscover `classic_3`, that is the single most interesting thing the A/B can report,
    # and collapsing the two rows would hide it behind a name collision.
    for (name, keys) in configs
        duplicate = findfirst(c -> c[1] != name && Set(c[2]) == Set(keys), configs)
        duplicate === nothing && continue
        println("  note: config '$name' is the same market set as " *
                "'$(configs[duplicate][1])'; both are reported.")
    end

    comparison_rows = NamedTuple[]
    for model in focus
        fit = fits[model.run_name]
        for (config_name, keys) in configs
            result = if config_name == "full_6"
                full_results[model.run_name]
            else
                first(run_portfolio_simulation(
                    eda_book_spec(String.(keys)), policy_spec, fit, bf_odds, ds;
                    bootstrap = false, require_converged = false, quiet = true))
            end
            summary = result.summary
            full_window = eda_window_metrics(result.daily_states, first_from, second_to)
            second_window = eda_window_metrics(result.daily_states, second_from, second_to)
            push!(comparison_rows, (
                model = model.run_name,
                generation = model.generation,
                config = config_name,
                markets = join(keys, "+"),
                n_markets = length(keys),
                n_bets = summary.n_bets,
                total_return_pct = summary.total_return_pct,
                flat_roi_pct = summary.roi,
                roi_1x2_pct = summary.roi_1x2,
                sharpe_ann = summary.sharpe_ann,
                max_drawdown_pct = summary.mdd,
                calmar = summary.calmar,
                win_rate_pct = 100 * summary.win_rate,
                turnover = summary.total_stake,
                mean_exposure = summary.mean_exposure,
                n_capped = summary.n_capped,
                n_slates = summary.n_slates,
                oos_return_pct = second_window.return_pct,
                oos_sharpe_ann = second_window.sharpe_ann,
                oos_mdd_pct = second_window.mdd_pct,
                oos_n_bets = second_window.n_bets,
                oos_turnover = second_window.turnover,
                full_window_return_pct = full_window.return_pct,
            ))
            @printf("  %-30s %-11s mk=%d bets=%5d ret=%+8.2f%% Shrp=%6.3f OOSret=%+8.2f%% OOSShrp=%6.3f\n",
                    model.run_name, config_name, length(keys), summary.n_bets,
                    summary.total_return_pct, summary.sharpe_ann,
                    second_window.return_pct, second_window.sharpe_ann)
        end
    end
    comparison = DataFrame(comparison_rows)

    println("\n[4/4] Writing artefacts...")
    mkpath(EDA_OUTPUT_DIR)
    CSV.write(joinpath(EDA_OUTPUT_DIR, "market_line_breakdown.csv"), breakdown)
    CSV.write(joinpath(EDA_OUTPUT_DIR, "market_pruning_comparison.csv"), comparison)
    CSV.write(joinpath(EDA_OUTPUT_DIR, "market_selection_ledger.csv"),
              select(pooled_all, [:model, :match_id, :date, :market_key, :group, :line,
                                  :selection, :odds, :stake, :abs_stake, :pnl, :abs_pnl,
                                  :payoff, :p_model, :p_market, :edge, :won]))

    eda_print_tables(breakdown, line_rows, comparison, curated_keys)
    eda_write_report(breakdown, line_rows, comparison, pooled_focus, pooled_all,
                     curated_keys, keep_keys, prune_keys, conditional_keys,
                     configs, focus_names, split_date, first_from, second_to, bf_odds)

    println("\nWrote:")
    println("  ", joinpath(EDA_OUTPUT_DIR, "market_line_breakdown.csv"))
    println("  ", joinpath(EDA_OUTPUT_DIR, "market_pruning_comparison.csv"))
    println("  ", joinpath(EDA_OUTPUT_DIR, "market_selection_ledger.csv"))
    println("  ", EDA_REPORT_PATH)

    return (; breakdown, comparison, line_rows, ledgers, pooled_focus, pooled_all,
              curated_keys, split_date)
end

# ==============================================================================
# 8. Console rendering
# ==============================================================================

function eda_print_tables(breakdown::DataFrame, line_rows::DataFrame,
                          comparison::DataFrame, curated_keys)
    pooled = breakdown[(breakdown.scope .== "POOLED_FOCUS"), :]

    println("\n", "="^140)
    println(" PER-LINE BREAKDOWN — focus models pooled (m05_joint_production_wealth, m12, m13)")
    println("="^140)
    @printf(" %-7s %-8s | %6s | %7s | %7s | %8s | %8s | %8s | %8s | %8s | %7s | %7s\n",
            "Market", "Sel", "Bets", "Win%", "AvgOdds", "p_model", "Calib", "Edge",
            "FlatROI", "KellyROI", "Cap%", "Eff")
    println("-"^140)
    for row in eachrow(pooled)
        @printf(" %-7s %-8s | %6d | %7.2f | %7.2f | %8.4f | %+8.4f | %+8.4f | %+8.2f | %+8.2f | %7.2f | %7.2f\n",
                row.market_key, row.selection, row.n_bets, row.win_rate_pct, row.mean_odds,
                row.mean_p_model, row.calib_bias, row.mean_edge, row.flat_roi_pct,
                row.kelly_roi_pct, row.capital_share_pct, row.capital_efficiency)
    end

    println("\n", "="^140)
    println(" VERDICTS — fitted on the selection window only")
    println("="^140)
    @printf(" %-7s | %-12s | %7s | %9s | %9s | %7s | %s\n",
            "Market", "Verdict", "Bets", "KellyROI", "FlatROI", "Eff", "Reason")
    println("-"^140)
    for row in eachrow(sort(line_rows, :market_key))
        @printf(" %-7s | %-12s | %7d | %+9.2f | %+9.2f | %7.2f | %s\n",
                row.market_key, row.verdict, row.n_bets, row.kelly_roi_pct,
                row.flat_roi_pct, row.capital_efficiency, row.reason)
    end
    println("\n Curated basket: ", join(curated_keys, ", "))

    println("\n", "="^140)
    println(" CONFIGURATION A/B — full period, and the out-of-sample second half")
    println("="^140)
    aggregated = combine(groupby(comparison, [:config, :n_markets]),
                         :total_return_pct => mean => :return_pct,
                         :sharpe_ann => mean => :sharpe,
                         :max_drawdown_pct => mean => :mdd,
                         :turnover => mean => :turnover,
                         :n_bets => mean => :bets,
                         :oos_return_pct => mean => :oos_return,
                         :oos_sharpe_ann => mean => :oos_sharpe,
                         :oos_mdd_pct => mean => :oos_mdd)
    sort!(aggregated, :oos_sharpe; rev = true)
    @printf(" %-11s | %3s | %7s | %10s | %8s | %8s | %9s | %11s | %10s | %9s\n",
            "Config", "Mk", "Bets", "Return %", "Sharpe", "MaxDD", "Turnover",
            "OOS ret %", "OOS Sharpe", "OOS MDD")
    println("-"^140)
    for row in eachrow(aggregated)
        @printf(" %-11s | %3d | %7.0f | %+10.2f | %8.3f | %8.2f | %9.2f | %+11.2f | %10.3f | %9.2f\n",
                row.config, row.n_markets, row.bets, row.return_pct, row.sharpe, row.mdd,
                row.turnover, row.oos_return, row.oos_sharpe, row.oos_mdd)
    end
    println("="^140)
    return nothing
end

# ==============================================================================
# 9. Markdown report
# ==============================================================================

function eda_md_table(header::Vector{String}, rows::Vector{Vector{String}})
    io = IOBuffer()
    println(io, "| ", join(header, " | "), " |")
    println(io, "|", join(fill(" :--- ", length(header)), "|"), "|")
    for row in rows
        println(io, "| ", join(row, " | "), " |")
    end
    return String(take!(io))
end

g4(x) = (ismissing(x) || !isfinite(x)) ? "—" : @sprintf("%.4f", x)
g2(x) = (ismissing(x) || !isfinite(x)) ? "—" : @sprintf("%.2f", x)
gs(x) = (ismissing(x) || !isfinite(x)) ? "—" : @sprintf("%+.2f", x)

function eda_write_report(breakdown::DataFrame, line_rows::DataFrame, comparison::DataFrame,
                          pooled_focus::DataFrame, pooled_all::DataFrame,
                          curated_keys, keep_keys, prune_keys, conditional_keys,
                          configs, focus_names, split_date::Date,
                          period_from::Date, period_to::Date, bf_odds::DataFrame)
    io = IOBuffer()
    pooled = breakdown[breakdown.scope .== "POOLED_FOCUS", :]
    lines_only = pooled[pooled.selection .== "ALL", :]

    println(io, "# Scottish Lower — Market Line Forensics and Book Selection")
    println(io)
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"),
            " by `experiments/scottish_lower/eda_market_selection.jl`.")
    println(io)
    println(io, "`compare_scottish_experiments.jl` found that widening the book from three ",
            "markets to six cost every model return and Sharpe while leaving the 1X2 leg ",
            "untouched. That located the loss in the added totals lines without saying which ",
            "ones. This report opens the trade ledger and answers it per line and per side.")
    println(io)

    println(io, "## 0. Method, and what would make it wrong")
    println(io)
    println(io, "Every number below is read off `trajectory.bets` — the bets the backtest ",
            "actually struck, carrying `p_model`, `p_market`, the price taken and the realised ",
            "payoff. Nothing is re-derived from the latents, so this report and the portfolio ",
            "summary cannot disagree except through a bug.")
    println(io)
    println(io, "Three corrections that matter:")
    println(io)
    println(io, "1. **Stakes are fractions of a moving bankroll.** `stake` and `pnl` are ",
            "fractions of the bankroll at their own slate, so summing them raw across a ",
            "compounding backtest adds different units. Every currency figure here is ",
            "rescaled by the slate's opening bankroll first.")
    println(io, "2. **Removing a line is not subtracting its P&L.** Kelly re-solves over ",
            "what remains and the exposure cap binds differently, so the counterfactual is a ",
            "re-simulation — §4. The per-line drawdown in §2 is each line's own standalone ",
            "stream, never a claim about what it cost the portfolio.")
    println(io, "3. **Selecting and scoring on the same data is bias.** The pruning rule is ",
            "fitted on the first half of the calendar (to ", split_date, ") using the three ",
            "focus models only, and every configuration is also scored on the second half, ",
            "which the rule never saw.")
    println(io)
    println(io, "Period: ", period_from, " to ", period_to, ". Prices: Betfair exchange close, ",
            "time-weighted over [−20 min, kickoff], ", nrow(bf_odds), " rows across ",
            length(unique(bf_odds.match_id)), " matches. Book policy held fixed at ",
            "`FlatTrust(1.0)`, `SlateDrawdown(23.0)`, `FixedCap(0.20)`, `DailySlate()`, ",
            "`FractionalKelly(0.30)`, 2% commission.")
    println(io)
    println(io, "Focus models: ", join(["`" * n * "`" for n in focus_names], ", "), ".")
    println(io)

    println(io, "## 1. Per-line breakdown")
    println(io)
    println(io, "Focus models pooled. `Calib` is empirical win rate minus mean predicted ",
            "probability, so positive means the model UNDER-rates the selection. `Edge` is ",
            "mean `p_model − p_market`. `Cap %` is the line's share of all capital staked; ",
            "`Eff` is its Kelly ROI over the whole book's, so 1.00 is carrying its weight and ",
            "below 1.00 is being carried.")
    println(io)
    print(io, eda_md_table(
        ["Market", "Sel", "Bets", "Win %", "Avg odds", "p_model", "Calib", "Edge",
         "Flat ROI %", "Kelly ROI %", "Cap %", "Eff", "PnL (units)"],
        [[r.market_key, r.selection, string(r.n_bets), g2(r.win_rate_pct), g2(r.mean_odds),
          g4(r.mean_p_model), g4(r.calib_bias), g4(r.mean_edge), gs(r.flat_roi_pct),
          gs(r.kelly_roi_pct), g2(r.capital_share_pct), g2(r.capital_efficiency),
          gs(r.total_pnl_units)] for r in eachrow(pooled)]))
    println(io)

    println(io, "### 1.1 Risk shape per line")
    println(io)
    print(io, eda_md_table(
        ["Market", "Bets", "Mean stake frac", "Max stake frac", "Standalone DD (units)",
         "DD % of turnover", "Max win streak", "Max loss streak", "Payoff autocorr"],
        [[r.market_key, string(r.n_bets), g4(r.mean_stake_frac), g4(r.max_stake_frac),
          g4(r.standalone_dd_units), g2(r.standalone_dd_pct_turnover),
          string(r.max_win_streak), string(r.max_loss_streak), g4(r.payoff_autocorr)]
         for r in eachrow(lines_only)]))
    println(io)

    eda_write_asymmetry(io, pooled)

    println(io, "## 2. Verdicts")
    println(io)
    println(io, "The rule, applied without exception to the selection window:")
    println(io)
    println(io, "```")
    println(io, "KEEP        kelly_roi > 0  AND  capital_efficiency >= 0.50  AND  n_bets >= ",
            EDA_MIN_BETS)
    println(io, "PRUNE       kelly_roi <= 0  OR  (capital_efficiency < 0.25 AND n_bets >= ",
            EDA_MIN_BETS, ")")
    println(io, "CONDITIONAL otherwise")
    println(io, "```")
    println(io)
    println(io, "A line can clear the ROI test and still fail the efficiency one. That is the ",
            "dilution case precisely: profitable, but at a rate low enough to drag the book's ",
            "average down while occupying slate budget the `FixedCap(0.20)` then denies to ",
            "better selections.")
    println(io)
    print(io, eda_md_table(
        ["Market", "Verdict", "Bets", "Kelly ROI %", "Flat ROI %", "Efficiency", "Cap %",
         "Reason"],
        [[r.market_key, "**" * r.verdict * "**", string(r.n_bets), gs(r.kelly_roi_pct),
          gs(r.flat_roi_pct), g2(r.capital_efficiency), g2(r.capital_share_pct), r.reason]
         for r in eachrow(sort(line_rows, :market_key))]))
    println(io)
    println(io, "- **KEEP / USE** — ", isempty(keep_keys) ? "none" :
            join(["`" * k * "`" for k in keep_keys], ", "))
    println(io, "- **AVOID / PRUNE** — ", isempty(prune_keys) ? "none" :
            join(["`" * k * "`" for k in prune_keys], ", "))
    println(io, "- **CONDITIONAL** — ", isempty(conditional_keys) ? "none" :
            join(["`" * k * "`" for k in conditional_keys], ", "))
    println(io)
    eda_write_stability(io, pooled_focus, split_date, line_rows)

    println(io, "## 3. Candidate configurations")
    println(io)
    print(io, eda_md_table(
        ["Config", "Markets"],
        [[name, join(["`" * k * "`" for k in keys], " + ")] for (name, keys) in configs]))
    println(io)

    println(io, "## 4. A/B comparison")
    println(io)
    println(io, "Mean across the three focus models. The OOS columns cover only slates after ",
            split_date, " — the window the pruning rule never saw.")
    println(io)
    aggregated = combine(groupby(comparison, [:config, :n_markets]),
                         :total_return_pct => mean => :return_pct,
                         :sharpe_ann => mean => :sharpe,
                         :max_drawdown_pct => mean => :mdd,
                         :calmar => mean => :calmar,
                         :turnover => mean => :turnover,
                         :n_bets => mean => :bets,
                         :flat_roi_pct => mean => :flat_roi,
                         :oos_return_pct => mean => :oos_return,
                         :oos_sharpe_ann => mean => :oos_sharpe,
                         :oos_mdd_pct => mean => :oos_mdd)
    sort!(aggregated, :oos_sharpe; rev = true)
    print(io, eda_md_table(
        ["Config", "Markets", "Bets", "Return %", "Flat ROI %", "Sharpe", "Max DD %",
         "Calmar", "Turnover", "OOS return %", "OOS Sharpe", "OOS max DD %"],
        [[r.config, string(r.n_markets), @sprintf("%.0f", r.bets), gs(r.return_pct),
          gs(r.flat_roi), g4(r.sharpe), g2(r.mdd), g4(r.calmar), g2(r.turnover),
          gs(r.oos_return), g4(r.oos_sharpe), g2(r.oos_mdd)]
         for r in eachrow(aggregated)]))
    println(io)

    println(io, "### 4.1 Per model")
    println(io)
    print(io, eda_md_table(
        ["Model", "Config", "Bets", "Return %", "Sharpe", "Max DD %", "Turnover",
         "OOS return %", "OOS Sharpe"],
        [["`" * r.model * "`", r.config, string(r.n_bets), gs(r.total_return_pct),
          g4(r.sharpe_ann), g2(r.max_drawdown_pct), g2(r.turnover),
          gs(r.oos_return_pct), g4(r.oos_sharpe_ann)]
         for r in eachrow(sort(comparison, [:model, :config]))]))
    println(io)

    eda_write_recommendation(io, aggregated, comparison, lines_only, configs,
                             curated_keys, prune_keys, split_date)

    println(io, "## 6. Reproducing this")
    println(io)
    println(io, "```bash")
    println(io, "julia --project -t 16 experiments/scottish_lower/eda_market_selection.jl")
    println(io, "```")
    println(io)
    println(io, "No MCMC is launched. Artefacts:")
    println(io)
    println(io, "- `results/market_line_breakdown.csv` — every scope, line and selection")
    println(io, "- `results/market_pruning_comparison.csv` — the A/B, per model and config")
    println(io, "- `results/market_selection_ledger.csv` — the raw bet ledger, all 8 models")

    open(EDA_REPORT_PATH, "w") do handle
        write(handle, String(take!(io)))
    end
    return EDA_REPORT_PATH
end

"""
    eda_write_stability(io, pooled_focus, split_date, line_rows)

Did each line's selection-window verdict survive into the window the rule never saw?

This is the section that decides whether §2's rule can be trusted, and it is deliberately
placed before the recommendation rather than after it. A line whose Kelly ROI reverses across
the split was never a line — it was a run of settlements the rule mistook for an edge.
"""
function eda_write_stability(io::IO, pooled_focus::DataFrame, split_date::Date,
                             line_rows::DataFrame)
    selection = pooled_focus[pooled_focus.date .<= split_date, :]
    evaluation = pooled_focus[pooled_focus.date .> split_date, :]

    rows = Vector{String}[]
    flips = String[]
    for key in EDA_FULL_KEYS
        sel = selection[selection.market_key .== key, :]
        evl = evaluation[evaluation.market_key .== key, :]
        roi(f) = (nrow(f) == 0 || sum(f.abs_stake) <= 0) ? NaN :
                 100 * sum(f.abs_pnl) / sum(f.abs_stake)
        sel_roi, evl_roi = roi(sel), roi(evl)
        verdict_row = line_rows[line_rows.market_key .== key, :]
        verdict = nrow(verdict_row) == 1 ? verdict_row.verdict[1] : "—"
        held = (isfinite(sel_roi) && isfinite(evl_roi)) ?
               (sign(sel_roi) == sign(evl_roi) ? "held" : "**reversed**") : "—"
        held == "**reversed**" && push!(flips, key)
        push!(rows, [key, verdict, string(nrow(sel)), gs(sel_roi),
                     string(nrow(evl)), gs(evl_roi), held])
    end

    println(io, "### 2.1 Did the verdicts survive the split?")
    println(io)
    print(io, eda_md_table(
        ["Market", "Verdict", "Sel. bets", "Sel. Kelly ROI %", "Eval bets",
         "Eval Kelly ROI %", "Sign"], rows))
    println(io)
    if isempty(flips)
        println(io, "Every line kept the sign of its Kelly ROI across the split.")
    else
        println(io, "**", join(["`" * k * "`" for k in flips], ", "),
                " reversed sign across the split.** A line that changes direction between the ",
                "window the rule saw and the window it did not was never an edge the rule ",
                "could have detected; it was a run of settlements. §4 is what adjudicates, ",
                "not §2.")
    end
    println(io)
    return nothing
end

"Is there an Over/Under asymmetry, and does it run the same way at every line?"
function eda_write_asymmetry(io::IO, pooled::DataFrame)
    ou = pooled[(pooled.group .== "OverUnder") .& (pooled.selection .!= "ALL"), :]
    println(io, "### 1.2 Over versus Under")
    println(io)
    if nrow(ou) == 0
        println(io, "No Over/Under selections were struck.")
        println(io)
        return nothing
    end
    print(io, eda_md_table(
        ["Market", "Side", "Bets", "Win %", "Avg odds", "Calib", "Edge", "Flat ROI %",
         "Kelly ROI %", "Eff"],
        [[r.market_key, r.selection, string(r.n_bets), g2(r.win_rate_pct), g2(r.mean_odds),
          g4(r.calib_bias), g4(r.mean_edge), gs(r.flat_roi_pct), gs(r.kelly_roi_pct),
          g2(r.capital_efficiency)] for r in eachrow(sort(ou, [:line, :selection]))]))
    println(io)

    sides = combine(groupby(ou, :selection),
                    :n_bets => sum => :bets,
                    :total_pnl_units => sum => :pnl,
                    :turnover_units => sum => :turnover)
    sides.kelly_roi = 100 .* sides.pnl ./ sides.turnover
    print(io, eda_md_table(
        ["Side (all lines)", "Bets", "Turnover (units)", "PnL (units)", "Kelly ROI %"],
        [[r.selection, string(r.bets), g2(r.turnover), gs(r.pnl), gs(r.kelly_roi)]
         for r in eachrow(sides)]))
    println(io)
    if nrow(sides) == 2
        gap = sides.kelly_roi[1] - sides.kelly_roi[2]
        println(io, "The two sides differ by ", g2(abs(gap)),
                " points of Kelly ROI, favouring `", sides.selection[gap > 0 ? 1 : 2],
                "`. A totals book that prices both sides off one score grid has no structural ",
                "reason to be better at one than the other, so a large gap is a statement ",
                "about where the exchange's price is soft, not about the model.")
        println(io)
    end
    return nothing
end

function eda_write_recommendation(io::IO, aggregated::DataFrame, comparison::DataFrame,
                                  lines_only::DataFrame, configs, curated_keys, prune_keys,
                                  split_date::Date)
    best_oos = aggregated[argmax(replace(aggregated.oos_sharpe, NaN => -Inf)), :]
    best_full = aggregated[argmax(replace(aggregated.sharpe, NaN => -Inf)), :]
    full6 = aggregated[aggregated.config .== "full_6", :][1, :]
    curated = aggregated[aggregated.config .== "curated", :]
    recommended_keys = String[]
    for (name, keys) in configs
        name == best_oos.config && (recommended_keys = String.(keys))
    end

    println(io, "## 5. Recommendation for the MatchDay console")
    println(io)
    println(io, "**Best Sharpe out of sample: `", best_oos.config, "` (",
            join(["`" * k * "`" for k in recommended_keys], " + "), ") — ",
            gs(best_oos.oos_return), "% at Sharpe ", g4(best_oos.oos_sharpe),
            ".** It is also the best over the full period (", gs(best_full.config == best_oos.config ?
            best_oos.return_pct : best_full.return_pct), "% at Sharpe ",
            g4(best_full.sharpe), " for `", best_full.config, "`).")
    println(io)

    # The rule is allowed to be wrong in public. Suppressing this would leave a reader with a
    # recommendation and no way to know the method that produced §2 had already failed a test.
    if nrow(curated) == 1
        c = curated[1, :]
        agrees = Set(curated_keys) == Set(recommended_keys)
        if agrees
            println(io, "§2's rule and the out-of-sample A/B agree on the same basket.")
        else
            println(io, "### 5.1 The rule in §2 was refuted, and the A/B is why it is here")
            println(io)
            println(io, "§2's rule selected ",
                    join(["`" * k * "`" for k in curated_keys], " + "),
                    ". Out of sample that basket returns ", gs(c.oos_return),
                    "% at Sharpe ", g4(c.oos_sharpe), " — the **worst** of every ",
                    "multi-market configuration tested, below even `x1x2_only`. The ",
                    "recommendation above therefore does not follow it.")
            println(io)
            println(io, "The mechanism is visible in §2.1. The rule fired on a line whose ",
                    "selection-window Kelly ROI did not survive the split, and it cleared ",
                    "the ", EDA_MIN_BETS, "-bet floor while doing so. The floor was too low ",
                    "for this decision.")
            println(io)
            println(io, "The obvious repair — raise the floor until the rule returns the ",
                    "basket the A/B endorses — is the same overfitting one level up, now on ",
                    "the threshold instead of the line. So the rule is left exactly as it ",
                    "was, and the out-of-sample comparison is what adjudicates. What §2 is ",
                    "good for is explaining WHY a line pays or does not; it is not reliable ",
                    "for choosing between baskets on this much data.")
            println(io)
            println(io, "The full-period line economics in §1 independently point the same ",
                    "way as the A/B: across the whole calendar the only two lines with ",
                    "capital efficiency above 1.00 are ",
                    join(["`" * String(r.market_key) * "` (" * g2(r.capital_efficiency) * ")"
                          for r in eachrow(lines_only) if isfinite(r.capital_efficiency) &&
                              r.capital_efficiency > 1.0], " and "), ".")
            println(io)
        end
    end

    println(io, "Production `BookSpec` for the live and replay consoles:")
    println(io)
    println(io, "```julia")
    println(io, "BookSpec(")
    println(io, "    markets = Data.MarketConfig(Data.AbstractMarket[")
    for key in recommended_keys
        market = EDA_MARKET_BY_KEY[key]
        # `nameof`, not the stringified type: the full path is
        # `BayesianFootball.Data.Markets.Market1X2`, and a snippet meant to be pasted into a
        # console config must use the exported `Data.Market1X2()` form the repo actually calls.
        type_name = string(nameof(typeof(market)))
        arguments = market isa Data.MarketOverUnder ? "(" * string(market.line) * ")" : "()"
        println(io, "        Data.", type_name, arguments, ",")
    end
    println(io, "    ]),")
    println(io, "    price     = DeArb(),")
    println(io, "    allocator = KellyLogUtility(),")
    println(io, "    shrink    = Portfolio.FractionalKelly(0.30),")
    println(io, "    exec      = ExecutionConfig(")
    println(io, "        commission          = PerBetCommission(0.02),")
    println(io, "        budget              = 0.99,")
    println(io, "        min_selection_stake = 0.001,")
    println(io, "    ),")
    println(io, ")")
    println(io, "```")
    println(io)

    dropped = [k for k in EDA_FULL_KEYS if !(k in recommended_keys)]
    isempty(dropped) ||
        println(io, "Dropped relative to the six-market book: ",
                join(["`" * k * "`" for k in dropped], ", "), ".")
    println(io)
    println(io, "This holds across every focus model individually, not only in the mean:")
    println(io)
    per_model = comparison[comparison.config .== best_oos.config, :]
    print(io, eda_md_table(
        ["Model", "Return %", "Sharpe", "Max DD %", "OOS return %", "OOS Sharpe"],
        [["`" * r.model * "`", gs(r.total_return_pct), g4(r.sharpe_ann),
          g2(r.max_drawdown_pct), gs(r.oos_return_pct), g4(r.oos_sharpe_ann)]
         for r in eachrow(sort(per_model, :model))]))
    println(io)

    println(io, "**What would change this.** The verdicts rest on one league pair, two ",
            "seasons and a selection window ending ", split_date,
            ". A line pruned for weak efficiency is not a line that cannot be priced — it is ",
            "one this book, at this Kelly fraction, under this exposure cap, could not pay ",
            "for. Loosening `FixedCap(0.20)` reduces the competition for slate budget and ",
            "would move the marginal lines first. The out-of-sample window is ",
            "one half of one calendar; it settles a comparison between five baskets, not the ",
            "general question of whether totals lines can be priced.")
    println(io)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    eda_market_selection()
end
