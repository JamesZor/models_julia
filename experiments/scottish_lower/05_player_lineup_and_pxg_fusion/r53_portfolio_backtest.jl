# ==============================================================================
# r53 — Fractional-Kelly Betfair portfolio backtest and PostgreSQL persistence
# ==============================================================================

using BayesianFootball
using CSV
using DataFrames
using Dates
using LinearAlgebra
using Printf
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l50_loader.jl"))

function r53_betfair_closing_odds(ds)
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

println("\n" * "="^110)
println(" EXPERIMENT 05 · 1X2 + O/U 2.5 + BTTS FRACTIONAL-KELLY BACKTEST")
println("="^110)
println("  shrinkage : FractionalKelly(0.30)")
println("  commission: 2% per winning exchange bet")
println("  cap       : 20% per daily slate")

bf_odds = r53_betfair_closing_odds(ds)
rows = NamedTuple[]
ledger_frames = DataFrame[]

for name in L50_MODEL_NAMES
    println("\n--- $name ---")
    fit = load_fit(db, name)
    fit.diagnostics.passed || error("$name failed convergence; refusing portfolio simulation")
    run_id = save_fit(fit, db) # config-hash deduplication returns the immutable existing UUID.

    result, books, report = run_portfolio_simulation(
        l50_book, l50_policy, fit, bf_odds, ds;
        bootstrap = true,
        require_converged = true,
        quiet = true,
    )
    portfolio_id = save_portfolio_db(
        result, run_id, db;
        book_spec = l50_book,
        policy_spec = l50_policy,
        metadata = (; candidate = name, odds_source = "betfair_twa_minus20_to_close"),
    )

    bets = copy(result.trajectory.bets)
    if nrow(bets) > 0
        insertcols!(bets, 1, :model => fill(name, nrow(bets)))
        insertcols!(bets, 2, :model_run_id => fill(string(run_id), nrow(bets)))
        insertcols!(bets, 3, :portfolio_run_id => fill(string(portfolio_id), nrow(bets)))
        push!(ledger_frames, bets)
    end

    s = result.summary
    push!(rows, (
        model = name,
        model_run_id = string(run_id),
        portfolio_run_id = string(portfolio_id),
        n_books = length(books),
        n_skipped = n_skipped(report),
        n_bets = s.n_bets,
        total_return_pct = s.total_return_pct,
        flat_roi_pct = s.roi,
        roi_1x2_pct = s.roi_1x2,
        max_drawdown_pct = s.mdd,
        sharpe_ann = s.sharpe_ann,
        win_rate = s.win_rate,
    ))
end

summary = sort(DataFrame(rows), :total_return_pct; rev = true)
ledger = isempty(ledger_frames) ? DataFrame() : vcat(ledger_frames...; cols = :union)
output_dir = joinpath(@__DIR__, "results")
mkpath(output_dir)
CSV.write(joinpath(output_dir, "r53_portfolio_summary.csv"), summary)
CSV.write(joinpath(output_dir, "r53_trade_ledger.csv"), ledger)

println("\n" * "="^155)
@printf(" %-45s | %6s | %8s | %9s | %9s | %9s | %9s | %8s\n",
        "Model", "Bets", "Return", "Flat ROI", "1X2 ROI", "Max DD", "Sharpe", "Win rate")
println("-"^155)
for row in eachrow(summary)
    @printf(" %-45s | %6d | %8.2f | %9.2f | %9.2f | %9.2f | %9.3f | %8.2f\n",
            row.model, row.n_bets, row.total_return_pct, row.flat_roi_pct,
            row.roi_1x2_pct, row.max_drawdown_pct, row.sharpe_ann, 100 * row.win_rate)
end
println("="^155)
println("PostgreSQL now contains portfolio_runs, portfolio_bets, and portfolio_artifacts for all five models.")
println("CSV summary and trade ledger written under $output_dir")
