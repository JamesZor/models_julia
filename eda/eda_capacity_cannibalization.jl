# ==============================================================================
# eda_capacity_cannibalization.jl — capacity, opportunity cost, and calibration audit
# ==============================================================================
#
# Part A first inventories the exact PostgreSQL portfolio artefacts attached to the four named
# canonical 40-fold runs. Those production artefacts contain only 1X2, O/U 2.5 and BTTS, so they
# cannot identify cannibalization by O/U 0.5, 1.5 or 3.5. The script records that fact, then runs
# one controlled six-market status-quo counterfactual for m12 and m13 and compares it with a true
# core-only re-solve. Opportunity cost is never estimated by simply subtracting fringe P&L.
#
# Part B prices every requested selection for all four fits and computes Brier, ECE, and ten-bin
# reliability curves from the same aligned fixture-selection rows.
#
# This runner launches no MCMC.
#
# Usage:
#   julia --project -t 8 eda/eda_capacity_cannibalization.jl
#
# Required prerequisite for the controlled policy outputs:
#   julia --project -t 8 eda/eda_policy_ab_test.jl
#
# Outputs live under eda/results/stochastic_control_capacity/.

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

const SC_RUN_INVENTORY_PATH = joinpath(SC_OUTPUT_DIR, "canonical_run_inventory.csv")
const SC_DB_BETS_PATH = joinpath(SC_OUTPUT_DIR, "canonical_database_bets.csv")
const SC_DB_PORTFOLIOS_PATH = joinpath(SC_OUTPUT_DIR, "canonical_portfolio_inventory.csv")
const SC_DB_CAPACITY_PATH = joinpath(SC_OUTPUT_DIR, "canonical_portfolio_capacity.csv")
const SC_CAPACITY_PATH = joinpath(SC_OUTPUT_DIR, "capacity_segment_summary.csv")
const SC_OPPORTUNITY_PATH = joinpath(SC_OUTPUT_DIR, "capacity_opportunity_cost.csv")
const SC_CALIBRATION_PATH = joinpath(SC_OUTPUT_DIR, "line_calibration_summary.csv")
const SC_RELIABILITY_PATH = joinpath(SC_OUTPUT_DIR, "line_reliability_curves.csv")
const SC_ALIGNMENT_PATH = joinpath(SC_OUTPUT_DIR, "calibration_alignment_audit.csv")

function sc_portfolio_market_list(book::BookSpec)
    return join(string.(book.markets.markets), "; ")
end

function sc_portfolio_inventory()
    rows = NamedTuple[]
    for ref in SC_CANONICAL_RUNS
        bundle = sc_load_portfolio_bundle(ref)
        result = bundle.result
        policy = bundle.policy
        book = bundle.book
        policy.cap isa FixedCap || error(
            "Canonical portfolio $(ref.portfolio_uuid) does not use FixedCap.")
        push!(rows, (
            experiment = ref.experiment,
            model = ref.name,
            run_uuid = string(ref.run_uuid),
            portfolio_uuid = string(ref.portfolio_uuid),
            portfolio_created_at = bundle.created_at,
            book_spec_hash = bundle.book_spec_hash,
            policy_spec_hash = bundle.policy_spec_hash,
            markets = sc_portfolio_market_list(book),
            n_markets = length(book.markets.markets),
            trust_type = string(nameof(typeof(policy.trust))),
            risk_type = string(nameof(typeof(policy.risk))),
            cap = policy.cap.cap,
            grouping_type = string(nameof(typeof(policy.grouping))),
            n_bets = result.summary.n_bets,
            n_slates = result.summary.n_slates,
            final_bankroll = result.summary.final_bankroll,
            total_return_pct = result.summary.total_return_pct,
            annual_sharpe = result.summary.sharpe_ann,
            max_drawdown_pct = result.summary.mdd,
            metadata = bundle.metadata,
        ))
    end
    return DataFrame(rows)
end

function sc_database_capacity_summary(database_bets::DataFrame, inventory::DataFrame)
    rows = NamedTuple[]
    for inv in eachrow(inventory)
        frame = database_bets[database_bets.portfolio_run_id .== inv.portfolio_uuid, :]
        daily = combine(
            groupby(frame, :kickoff_date),
            :stake_fraction => sum => :exposure,
            :stake_amount => sum => :stake_amount,
            :pnl => sum => :pnl,
            nrow => :n_bets,
        )
        threshold = SC_CAP_THRESHOLD * inv.cap
        daily.constrained = daily.exposure .>= threshold
        for regime in ("constrained", "unconstrained", "all")
            dates = regime == "all" ? Set(Date.(daily.kickoff_date)) :
                    Set(Date.(daily.kickoff_date[daily.constrained .==
                                                   (regime == "constrained")]))
            slice = frame[in.(Date.(frame.kickoff_date), Ref(dates)), :]
            for bucket in ("core", "fringe")
                selected = slice[slice.capacity_bucket .== bucket, :]
                stake = sum(selected.stake_amount)
                pnl = sum(selected.pnl)
                push!(rows, (
                    experiment = inv.experiment,
                    model = inv.model,
                    portfolio_uuid = inv.portfolio_uuid,
                    regime,
                    capacity_bucket = bucket,
                    n_slates = length(dates),
                    n_bets = nrow(selected),
                    stake_amount = stake,
                    stake_share_pct = sum(slice.stake_amount) > 0 ?
                                      100 * stake / sum(slice.stake_amount) : NaN,
                    pnl_amount = pnl,
                    win_rate_pct = nrow(selected) > 0 ? 100 * mean(selected.won) : NaN,
                    kelly_efficiency_pct = stake > 0 ? 100 * pnl / stake : NaN,
                    mean_slate_exposure = isempty(dates) ? NaN :
                        mean(daily.exposure[in.(Date.(daily.kickoff_date), Ref(dates))]),
                    cap = inv.cap,
                    threshold,
                ))
            end
        end
    end
    return DataFrame(rows)
end

function sc_load_policy_outputs()
    summary_path = joinpath(SC_OUTPUT_DIR, "policy_ab_summary.csv")
    daily_path = joinpath(SC_OUTPUT_DIR, "policy_ab_daily.csv")
    ledger_path = joinpath(SC_OUTPUT_DIR, "policy_ab_ledger.csv")
    for path in (summary_path, daily_path, ledger_path)
        isfile(path) || error(
            "Missing $path. Run `julia --project -t 8 eda/eda_policy_ab_test.jl` first.")
    end
    return (;
        summary = CSV.read(summary_path, DataFrame),
        daily = CSV.read(daily_path, DataFrame),
        ledger = CSV.read(ledger_path, DataFrame),
    )
end

function sc_reconstruct_policy_result_inputs(outputs, model::AbstractString,
                                             policy::AbstractString)
    daily = outputs.daily[(outputs.daily.model .== model) .&
                          (outputs.daily.policy .== policy), :]
    ledger = outputs.ledger[(outputs.ledger.model .== model) .&
                            (outputs.ledger.policy .== policy), :]
    summary = outputs.summary[(outputs.summary.model .== model) .&
                              (outputs.summary.policy .== policy), :]
    nrow(summary) == 1 || error("Expected one summary row for $model/$policy.")
    return (; daily, ledger, summary = summary[1, :])
end

function sc_capacity_from_policy_outputs(outputs)
    frames = DataFrame[]
    opportunity = NamedTuple[]
    for model in ("m12_joint_hybrid_synergy", "m13_joint_composite")
        status = sc_reconstruct_policy_result_inputs(outputs, model, "P1_status_quo")
        core = sc_reconstruct_policy_result_inputs(outputs, model, "P2_hard_pruning")

        # `sc_capacity_segment_summary` needs DailyState-like property access only.
        push!(frames, sc_capacity_segment_summary(
            status.ledger, eachrow(status.daily), model; cap = SC_CAP))

        status_states = Dict(Date(r.date) => r for r in eachrow(status.daily))
        core_states = Dict(Date(r.date) => r for r in eachrow(core.daily))
        dates = sort(collect(keys(status_states)))
        constrained = Set(d for d in dates
                          if status_states[d].exposure >= SC_CAP_THRESHOLD * SC_CAP)
        status_core = sc_stake_by_date(status.ledger, "core")
        status_fringe = sc_stake_by_date(status.ledger, "fringe")
        core_only = sc_stake_by_date(core.ledger, "core")
        status_binding_growth = prod(1.0 + status_states[d].pnl_frac for d in constrained)
        core_binding_growth = prod(1.0 + core_states[d].pnl_frac for d in constrained)
        hybrid_growth = prod(1.0 + (d in constrained ? core_states[d].pnl_frac :
                                   status_states[d].pnl_frac) for d in dates)
        removed = sum(get(status_fringe, d, 0.0) for d in constrained)
        redirected = sum(get(core_only, d, 0.0) - get(status_core, d, 0.0)
                         for d in constrained)
        pnl_delta = sum(core_states[d].pnl_frac - status_states[d].pnl_frac
                        for d in constrained)
        status_wealth = status.summary.final_bankroll / status.summary.initial_bankroll
        push!(opportunity, (
            model,
            n_slates = length(dates),
            n_constrained_slates = length(constrained),
            constrained_share_pct = 100 * length(constrained) / length(dates),
            fringe_stake_removed_frac = removed,
            core_stake_change_frac = redirected,
            capacity_recaptured_pct = removed > 0 ? 100 * redirected / removed : NaN,
            status_return_constrained_pct = 100 * (status_binding_growth - 1.0),
            core_return_constrained_pct = 100 * (core_binding_growth - 1.0),
            constrained_return_delta_pp = 100 * (core_binding_growth - status_binding_growth),
            status_full_return_pct = status.summary.total_return_pct,
            core_full_return_pct = core.summary.total_return_pct,
            hybrid_binding_only_return_pct = 100 * (hybrid_growth - 1.0),
            hybrid_opportunity_cost_pp = 100 * (hybrid_growth - status_wealth),
            realized_shadow_value_pct_per_stake = removed > 0 ? 100 * pnl_delta / removed : NaN,
        ))
    end
    return (; capacity = vcat(frames...; cols = :union),
              opportunity = DataFrame(opportunity))
end

function sc_alignment_row(ref::SCCanonicalRun, alignment)
    return (
        experiment = ref.experiment,
        model = ref.name,
        run_uuid = string(ref.run_uuid),
        n_odds_rows = alignment.n_odds_rows,
        n_priced_fixtures = alignment.n_priced_fixtures,
        n_matched = alignment.n_matched,
        n_scored = alignment.n_scored,
        n_unpriced = alignment.n_unpriced,
        n_no_market = alignment.n_no_market,
        n_no_outcome = alignment.n_no_outcome,
        duplicate_keys = alignment.duplicate_keys,
        mismatched_ids = alignment.mismatched_ids,
        passed = alignment.ok,
    )
end

function eda_capacity_cannibalization()
    println("\n", "="^122)
    println(" STOCHASTIC CONTROL — CAPACITY CANNIBALIZATION AND LINE CALIBRATION")
    println("="^122)
    mkpath(SC_OUTPUT_DIR)

    println("\n[1/4] Verifying immutable run and portfolio addresses...")
    inventory = sc_verify_run_inventory()
    portfolios = sc_portfolio_inventory()
    database_bets = sc_query_canonical_database_bets()
    database_capacity = sc_database_capacity_summary(database_bets, portfolios)
    @printf("  runs=%d portfolios=%d database bets=%d\n",
            nrow(inventory), nrow(portfolios), nrow(database_bets))
    for row in eachrow(portfolios)
        println("  $(row.model): $(row.n_markets) persisted markets — $(row.markets)")
    end

    println("\n[2/4] Reading controlled six-market policy outputs...")
    policy_outputs = sc_load_policy_outputs()
    controlled = sc_capacity_from_policy_outputs(policy_outputs)
    for row in eachrow(controlled.opportunity)
        @printf("  %-28s constrained=%2d/%2d core Δ=%+7.2fpp hybrid Δ=%+7.2fpp shadow=%+7.2f%%/stake\n",
                row.model, row.n_constrained_slates, row.n_slates,
                row.core_full_return_pct - row.status_full_return_pct,
                row.hybrid_opportunity_cost_pp,
                row.realized_shadow_value_pct_per_stake)
    end

    println("\n[3/4] Pricing all lines and computing calibration geometry...")
    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
    betfair = sc_betfair_closing_odds(ds)
    enriched = sc_enrich_odds(betfair, ds)
    calibration_frames = DataFrame[]
    curve_frames = DataFrame[]
    alignment_rows = NamedTuple[]
    for ref in SC_CANONICAL_RUNS
        fit = sc_load_fit(ref)
        tables = sc_calibration_tables(fit, enriched, ds, ref.name)
        push!(calibration_frames, tables.summary)
        push!(curve_frames, tables.curves)
        push!(alignment_rows, sc_alignment_row(ref, tables.alignment))
        @printf("  %-28s scored=%5d alignment=%s\n",
                ref.name, tables.alignment.n_scored,
                tables.alignment.ok ? "PASS" : "FAIL")
    end
    calibration = vcat(calibration_frames...; cols = :union)
    reliability = vcat(curve_frames...; cols = :union)
    alignment = DataFrame(alignment_rows)
    all(alignment.passed) || error("At least one calibration alignment audit failed.")

    println("\n[4/4] Writing reproducible artefacts...")
    CSV.write(SC_RUN_INVENTORY_PATH, inventory)
    CSV.write(SC_DB_PORTFOLIOS_PATH, portfolios)
    CSV.write(SC_DB_BETS_PATH, database_bets)
    CSV.write(SC_DB_CAPACITY_PATH, database_capacity)
    CSV.write(SC_CAPACITY_PATH, controlled.capacity)
    CSV.write(SC_OPPORTUNITY_PATH, controlled.opportunity)
    CSV.write(SC_CALIBRATION_PATH, calibration)
    CSV.write(SC_RELIABILITY_PATH, reliability)
    CSV.write(SC_ALIGNMENT_PATH, alignment)

    println("Wrote:")
    for path in (SC_RUN_INVENTORY_PATH, SC_DB_PORTFOLIOS_PATH, SC_DB_BETS_PATH,
                 SC_DB_CAPACITY_PATH, SC_CAPACITY_PATH, SC_OPPORTUNITY_PATH,
                 SC_CALIBRATION_PATH, SC_RELIABILITY_PATH, SC_ALIGNMENT_PATH)
        println("  ", path)
    end
    return (; inventory, portfolios, database_bets, database_capacity,
              capacity = controlled.capacity, opportunity = controlled.opportunity,
              calibration, reliability, alignment)
end

if abspath(PROGRAM_FILE) == @__FILE__
    eda_capacity_cannibalization()
end
