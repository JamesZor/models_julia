# ==============================================================================
# r47 — Betfair Exchange portfolio backtest: two-arm joint model vs Poisson control
# Scottish Lower (tiers 56/57) · BayesianFootball.jl Unified V2 stack
# ==============================================================================
#
# WHAT THIS IS
#   The money test for r46. It prices the SAME 710 out-of-sample fixtures with each
#   arm's posterior, stakes them against real Betfair closing prices under one fixed
#   policy, and reports what the bankroll did.
#
#   QUESTION. r46 showed every joint arm beating the Poisson control by ~0.0028
#   out-of-sample log loss, closing almost the whole gap to the market (0.6571 against
#   the market's 0.6568). Does that convert into money, or is it a scoring improvement
#   that the closing line has already priced?
#
#   CONTROL.       m00_poisson_control — the identical spine with no Gamma arm. Every
#                  other input (prices, markets, allocator, commission, policy) is held
#                  fixed, so a difference in return is attributable to the second
#                  likelihood and to nothing else.
#   DECISION RULE. A joint arm earns its two extra parameters only if it beats the
#                  control on RISK-ADJUSTED return, not on raw return. A higher bankroll
#                  bought with a deeper drawdown is not an improvement.
#
# WHAT THIS IS NOT
#   Not a forward-looking projection. These are closing prices on settled matches with
#   perfect knowledge of which fixtures existed; slippage, liquidity limits and market
#   impact are not modelled. Treat the level as optimistic and the RANKING as the result.
#
#   A single 710-fixture backtest also cannot separate a real edge from a lucky draw.
#   Bootstrap resampling is enabled so the reported spread is at least honest about that.
#
# PRICE CONTRACT
#   Betfair closing odds, time-weighted average over [-20min, 0min] to kickoff — the same
#   estimator r22 and r33 use, so the numbers here are comparable to experiments 01 and 02.
#
# USAGE
#   julia --project -t 16
#   julia> include("current_development/scottish_lower/r47_portfolio_betfair_joint.jl")
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Printf
using Statistics

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================

const R47_FIT_ROOT = get(ENV, "R47_FIT_ROOT", "./data/scottish_lower_2426_joint")
const R47_CONTROL  = "m00_poisson_control"

# Held fixed across every arm. Identical to r22/r33 so the returns are comparable to
# experiments 01 and 02 rather than only to each other.
const R47_COMMISSION   = 0.02
const R47_TRUST        = 0.30
const R47_SLATE_DD     = 23.0
const R47_CAP          = 0.20
const R47_TWA_WINDOW   = (-20.0, 0.0)

const R47_ARMS = [
    "m00_joint_baseline",
    "m02_joint_squad_wealth",
    "m03_joint_distance",
    "m04_joint_wealth_distance",
    "m05_joint_production_wealth",
    "m07_joint_bench_depth",
    "m08_joint_composite",
    R47_CONTROL,
]

println("\n" * "="^140)
println(" r47 · BETFAIR PORTFOLIO BACKTEST — TWO-ARM JOINT vs POISSON CONTROL (24/25 + 25/26)")
println("="^140)

# %%
# ==============================================================================
# 3. Data snapshot and Betfair closing prices
# ==============================================================================

println("\n[1/4] Loading DataStore ...")
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

println("[2/4] Summarising Betfair closing prices (TWA $(R47_TWA_WINDOW)) ...")
bf_raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = R47_TWA_WINDOW)
bf_odds = DataFrame(
    match_id    = Int.(bf_raw.match_id),
    market_name = String.(bf_raw.market_name),
    market_line = Float64.(bf_raw.market_line),
    selection   = Symbol.(bf_raw.selection),
    odds_close  = Float64.(bf_raw.odds),
)
@printf("  %d quotes across %d matches\n", nrow(bf_odds), length(unique(bf_odds.match_id)))

# %%
# ==============================================================================
# 4. Book and policy — identical for every arm
# ==============================================================================
#
# The comparability claim of this experiment IS that these two blocks are the same for
# all eight arms. Only the posterior changes.

book_spec = BookSpec(
    markets   = Data.MarketConfig([
        Data.Market1X2(),
        Data.MarketOverUnder(2.5),
        Data.MarketBTTS(),
    ]),
    price     = DeArb(),
    allocator = KellyLogUtility(),
    shrink    = BakerMcHale(),
    exec      = ExecutionConfig(
        commission          = PerBetCommission(R47_COMMISSION),
        budget              = 0.99,
        min_selection_stake = 0.001,
    ),
)

policy_spec = PolicySpec(
    trust    = FlatTrust(R47_TRUST),
    risk     = SlateDrawdown(R47_SLATE_DD),
    cap      = FixedCap(R47_CAP),
    grouping = DailySlate(),
)

# %%
# ==============================================================================
# 5. Simulation
# ==============================================================================

"Resolve a timestamped fit directory to the one holding `results.jld2`."
function r47_resolve_fit_dir(path::AbstractString)
    isfile(joinpath(path, "results.jld2")) && return path
    isdir(path) || return path
    stamped = filter(d -> isfile(joinpath(path, d, "results.jld2")), readdir(path))
    isempty(stamped) && return path
    sort!(stamped; rev = true)
    return joinpath(path, first(stamped))
end

println("\n[3/4] Simulating portfolios on Betfair closing prices ...")

r47_results = Dict{String, Any}()
for name in R47_ARMS
    path = r47_resolve_fit_dir(joinpath(R47_FIT_ROOT, name))
    if !isfile(joinpath(path, "results.jld2"))
        println("  MISSING: $name  ($path)")
        continue
    end
    fit = load_fit(path; quiet = true)
    res, _, _ = run_portfolio_simulation(book_spec, policy_spec, fit, bf_odds, ds;
                                         bootstrap = true, require_converged = false)
    r47_results[name] = res
    s = res.summary
    @printf("  %-30s bets %5d  return %+8.2f%%  ROI %+6.2f%%  MDD %6.2f%%  Sharpe %6.3f\n",
            name, s.n_bets, s.total_return_pct, s.roi, s.mdd, s.sharpe_ann)
end

# %%
# ==============================================================================
# 6. Leaderboard
# ==============================================================================
#
# Sorted by SHARPE, not by return. A bankroll bought with a deeper drawdown is not an
# improvement, and the decision rule in the header is explicit that risk-adjusted return
# is what an arm has to beat.

println("\n[4/4] Portfolio leaderboard ...")

r47_control = get(r47_results, R47_CONTROL, nothing)
base_sharpe = r47_control === nothing ? NaN : r47_control.summary.sharpe_ann
base_return = r47_control === nothing ? NaN : r47_control.summary.total_return_pct

ranked = sort([(name, res) for (name, res) in r47_results];
              by = x -> -x[2].summary.sharpe_ann)

println("="^140)
@printf(" %-30s | %6s | %10s | %9s | %9s | %8s | %8s | %9s | %9s\n",
        "Model", "Bets", "Return %", "Flat ROI", "1X2 ROI", "Max DD", "Sharpe", "ΔSharpe", "ΔReturn")
println("-"^140)
for (name, res) in ranked
    s = res.summary
    dsharpe = isnan(base_sharpe) ? NaN : s.sharpe_ann - base_sharpe
    dreturn = isnan(base_return) ? NaN : s.total_return_pct - base_return
    marker = name == R47_CONTROL ? " (control)" : ""
    @printf(" %-30s | %6d | %+9.2f%% | %+8.2f%% | %+8.2f%% | %7.2f%% | %8.3f | %+9.3f | %+8.2f%%%s\n",
            name, s.n_bets, s.total_return_pct, s.roi, s.roi_1x2, s.mdd, s.sharpe_ann,
            dsharpe, dreturn, marker)
end
println("="^140)

if r47_control !== nothing
    beat = [n for (n, r) in ranked
            if n != R47_CONTROL && r.summary.sharpe_ann > base_sharpe]
    if isempty(beat)
        println(" No joint arm beats the Poisson control on risk-adjusted return.")
        println(" The log-loss gain in r46 did NOT convert into money at these prices.")
    else
        println(" Beat the control on Sharpe: ", join(beat, ", "))
    end
end
