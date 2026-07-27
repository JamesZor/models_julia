using BayesianFootball
using DataFrames
using Printf

println("\n🚀 Running Kelly Staking Simulation on 1X2, OverUnder & BTTS...\n")

# 1. Filter the Bookmaker odds to 1X2, OverUnder and BTTS
allowed_markets = Set(["1X2", "OverUnder", "BTTS"])
filtered_odds = subset(ds1.odds, :market_name => ByRow(m -> in(m, allowed_markets)))

# 2. Define the Kelly Signal
signals = [BayesianFootball.Signals.BayesianKelly()]
master_dfs = DataFrame[]

# 3. Process Signals for both L1 and L1+L2
for (model_name, (ppd, mock_exp)) in ppd_dict
    println("Processing $model_name...")
    
    sig_result = BayesianFootball.Signals.process_signals(ppd, filtered_odds, signals; odds_column=:odds_close)
    df = sig_result.df
    
    # Calculate PnL (assuming 1 unit bankroll normalized per match)
    df.pnl = map(eachrow(df)) do r
        if ismissing(r.is_winner) || r.stake == 0.0
            0.0
        elseif r.is_winner
            r.stake * (r.odds - 1.0)
        else
            -r.stake
        end
    end
    
    df.model_name = fill(model_name, nrow(df))
    df.model_parameters = fill("none", nrow(df))
    push!(master_dfs, df)
end

# 4. Generate Breakdown Tearsheet
master_ledger = vcat(master_dfs...)
bt_ledger = BayesianFootball.BackTesting.BacktestLedger(master_ledger)

tearsheet = BayesianFootball.BackTesting.generate_tearsheet(
    bt_ledger,
    groupby_cols = [:model_name, :selection]
)

println("\n📊 TEARSHEET: L1 vs L1+L2 (1X2, OverUnder & BTTS)\n")
sort!(tearsheet, [:selection, :model_name])
@printf("%-15s | %-35s | %9s | %9s | %8s | %8s | %8s | %10s\n", 
        "SELECTION", "MODEL", "TURNOVER", "PROFIT", "ROI %", "SHARPE", "EMP_G", "WEALTH")
println("-" ^ 118)
for r in eachrow(tearsheet)
    @printf("%-15s | %-35s | %9.2f | %9.2f | %7.1f%% | %8.2f | %8.4f | %10.4f\n", 
            r.selection, r.model_name, r.turnover, r.profit, r.roi_pct, 
            r.SharpeRatio, r.hurdle_G_emp, r.CumulativeWealth)
end
println("-" ^ 118)

# ==============================================================================
# 5. PORTFOLIO LEVEL METRICS (THE TRUE OVERALL PERFORMANCE)
# ==============================================================================
println("\n🏆 PORTFOLIO OVERALL METRICS (L1 vs L1+L2)\n")

portfolio_tearsheet = BayesianFootball.BackTesting.generate_tearsheet(
    bt_ledger,
    groupby_cols = [:model_name] # Group strictly by model to get total chronological portfolio performance
)

@printf("%-35s | %8s | %10s | %10s | %8s | %8s | %10s | %18s\n", 
        "MODEL", "BETS", "TURNOVER", "PROFIT", "ROI %", "SHARPE", "SORTINO", "CUMULATIVE WEALTH")
println("=" ^ 125)

for r in eachrow(portfolio_tearsheet)
    @printf("%-35s | %8d | %10.2f | %10.2f | %7.1f%% | %8.2f | %10.2f | %18.4f\n", 
            r.model_name, r.bets_placed, r.turnover, r.profit, r.roi_pct, 
            r.SharpeRatio, r.SortinoRatio, r.CumulativeWealth)
end
println("=" ^ 125)
println("\nBy combining all markets chronologically, we calculate the true Portfolio-level Sharpe and Wealth multiplier!")
