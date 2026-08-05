# Portfolio Backtesting Architecture Plan

## 1. Architectural Review (From Exploration to Production)
Our current script (`r001_basic_explore.jl`) successfully demonstrates the core mechanics for a single match: extracting odds, normalizing vig, solving the constrained Kelly portfolio via `Fminbox`, enforcing market exclusivity, and calculating realized P/L. 

To scale this over thousands of matches in a backtest, we need to adapt the code:
*   **Decouple Processing from Printing**: The current `run_portfolio_analysis` computes and immediately prints formatted strings. We need it to return structured data objects so we can aggregate metrics across the season.
*   **Fast-Failing**: Computing the `score_matrix` takes time. We should extract odds *first*; if a match has no Betfair odds available in the dataset, we should skip matrix generation entirely to save compute.
*   **Multithreading**: Since each match's portfolio optimization is completely independent and relies only on fixed inputs (`latents`, `odds_df`), the backtest loop is embarrassingly parallel. We can use `Threads.@threads` to make processing an entire season incredibly fast.

## 2. Proposed Loader Design (`l02_portfolio_backtest.jl`)

We will define clean, composable functions and structs.

### Data Structures
Instead of loose variables, we'll store results in strict types to easily convert them into DataFrames later:
```julia
# Stores the P/L and Staking for a specific alpha on a specific match
struct AlphaResult
    alpha::Float64
    total_stake::Float64
    net_pl::Float64
    roi::Float64
end

# Stores the aggregated results for a single match across all alphas
struct MatchPortfolioResult
    match_id::Int
    date::DateTime
    home_team::String
    away_team::String
    alpha_results::Dict{Float64, AlphaResult}
    # Optional: store individual selection stakes for deep-dive analysis
end
```

### Core Functions
1.  `get_normalized_odds(odds_df, match_id, markets_config)`: Combines extraction and normalization. Returns `nothing` if the match lacks sufficient odds depth.
2.  `optimize_match_portfolio(...)`: The core math engine. Takes the score matrix and normalized odds, applies `Fminbox` and market netting. Returns raw stakes per alpha.
3.  `evaluate_match(latents_row, expr, odds_df, markets_config, alphas, commission)`: 
    *   Calls `get_normalized_odds`. Fast-fails if `nothing`.
    *   Generates `score_matrix` and `match_model_prob`.
    *   Calls `optimize_match_portfolio` loop.
    *   Calculates realized P/L using `is_winner`.
    *   Returns a `MatchPortfolioResult`.
4.  `run_backtest(latents_df, expr, odds_df, markets_config; alphas, commission)`:
    *   Iterates over all rows in `latents_df`.
    *   Uses multithreading (`Threads.@threads`) to process matches concurrently.
    *   Aggregates all `MatchPortfolioResult`s into a single master `DataFrame` for easy plotting and summarization.

## 3. Proposed Runner Design (`r02_smoke_test.jl`)

The runner will ensure the loader works perfectly before we unleash it on the entire dataset.

1.  **Setup**: Load `ds`, `odds`, `expr`, `latents`. Define `markets_config`.
2.  **Smoke Test 1 (Single Match)**: Run `evaluate_match` on Match #12476705 (our known Alloa Athletic match). Print the struct and `@test` that the P/L matches our known values exactly (-3.72% at $\alpha=1.0$).
3.  **Smoke Test 2 (Mini-Batch)**: Run `run_backtest` on `latents.df[1:20, :]`. 
4.  **Reporting**: Write a quick summary function `print_backtest_summary(results_df)` that shows:
    *   Total Matches Evaluated vs Skipped.
    *   Total Cumulative Stake across the batch.
    *   Total Net Profit / Loss.
    *   Global ROI.

## 4. Ideas & Considerations for Exploration

Before we write the code, I want to get your thoughts on a few design decisions:

### A. Bankroll Compounding & Simultaneous Matches
Standard Kelly criterion assumes sequential bets where your bankroll updates after every event: $W_t = W_{t-1} \times (1 + P/L)$. 
However, in football, 20 matches might kick off simultaneously at 3:00 PM on a Saturday. If we optimize each match assuming a 1-unit bankroll, the *total* stake across the 3:00 PM slate might exceed 1.0 (e.g., 5 matches each risking 25%).
*   **Idea 1 (Simple Additive)**: For this initial backtest, we treat the bankroll as fixed (e.g., always betting % of an initial £10,000 pot) and just sum the Net P/L.
*   **Idea 2 (Sequential Approximation)**: Sort matches strictly by Date/Time. If matches are simultaneous, we scale down the stakes proportionally so the total risk across the *entire time window* $\le 99\%$.
*   *Recommendation*: Let's stick to **Idea 1 (Simple Additive P/L tracking)** for the initial `l02` loader. We can build advanced slate-level optimization later.

### B. Fallback to Bookmaker Odds (`ds.odds`)
Our exploration showed Betfair window summaries can have artificial arbitrage (overround < 1.0). Normalizing them fixes the math, but the true matched price might be slightly different. 
*   *Recommendation*: In `get_normalized_odds`, we will continue normalizing Betfair odds to 1.0. We can optionally pass `ds.odds` to the backtester later if we want to simulate against true bookmaker vig (which is much harder to beat).

### C. Output Format
The `run_backtest` function should return a standard `DataFrame` where each row is a match, with columns like `alpha_10_pl`, `alpha_10_stake`, etc. This makes it trivial to use `DataFrames.combine(sum)` or plot cumulative sum charts later.

---
**Next Steps:** Let me know if you agree with this architecture, specifically the handling of Simultaneous Matches (Idea 1 vs Idea 2). Once aligned, I will immediately write the `l02_portfolio_backtest.jl` and `r02_smoke_test.jl` files!
