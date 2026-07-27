# current_development/trust_model_dev/r08_signals_backtest.jl

using BayesianFootball
using DataFrames
using Dates

println("\n🚀 Running Kelly Staking Simulation & Backtest...\n")

# Process raw Betfair tick data for tighter closing odds
println("Summarizing betfair market into tight close window...")
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds1  = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

# 1. Define the signals we want to test
signals = [
    Signals.BayesianKelly()
]

master_dfs = DataFrame[]

# Assume ppd_dict and ds are present in the REPL (Main)
for (model_name, (ppd, mock_exp)) in ppd_dict
    println("Processing Signals for: $model_name")
    
    # 2. Run the signal agent
    # We use :odds_close as the bookie price we are betting into (from the newly generated ds1)
    sig_result = Signals.process_signals(ppd, ds1.odds, signals; odds_column=:odds_close)
    df = sig_result.df
    
    # 3. Calculate PnL (assuming 1 unit bankroll normalization)
    df.pnl = map(eachrow(df)) do r
        if ismissing(r.is_winner) || r.stake == 0.0
            0.0
        elseif r.is_winner
            r.stake * (r.odds - 1.0)
        else
            -r.stake
        end
    end
    
    # 4. Attach Model Metadata
    df.model_name = fill(model_name, nrow(df))
    df.model_parameters = fill("none", nrow(df))
    
    push!(master_dfs, df)
end

# 5. Build Ledger & Tearsheet
master_ledger = vcat(master_dfs...)
bt_ledger = BackTesting.BacktestLedger(master_ledger)

println("\n📊 Generating Tearsheet (ROI/PnL)...\n")

tearsheet = BackTesting.generate_tearsheet(
    bt_ledger,
    groupby_cols = [:model_name, :signal_name, :signal_params, :selection]
)

display(tearsheet)
