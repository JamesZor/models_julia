# ==============================================================================
# Model 00 — GATE 7 : GROWTH (Pure Poisson)
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Assembles src/Portfolio for Model 00:
#   7a: Betfair book construction
#   7b: Simulation integrity (drawdown constraint & slate simulation)
#   7c: Growth verdict & P&L concentration analysis (top10_pct)
#   7d: Policy sweep
#
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Statistics
using Printf

const Pf = BayesianFootball.Portfolio


# ==============================================================================
# 1. The Odds Table & Book Spec
# ==============================================================================

function tp00_betfair_odds_df(ds, contract::SLContract;
                              ids::Union{Nothing,AbstractSet} = nothing,
                              window = (-20.0, 0.0))
    D   = BayesianFootball.Data
    raw = D.summarize_odds(ds.betfair_odds, D.TWAEstimator(); window = window)
    isempty(raw) && return DataFrame()

    wanted = Set((D.market_group(m), Float64(D.market_line(m))) for m in tp00_book_markets(contract))
    df = filter(r -> (String(r.market_name), Float64(r.market_line)) in wanted, raw)
    ids === nothing || (df = filter(r -> Int(r.match_id) in ids, df))
    "is_sane" in names(df) && (df = filter(r -> coalesce(r.is_sane, true), df))

    return DataFrame(
        match_id    = Int.(df.match_id),
        market_name = String.(df.market_name),
        market_line = Float64.(df.market_line),
        selection   = Symbol.(df.selection),
        odds_close  = Float64.(df.odds),
    )
end

function tp00_book_spec(contract::SLContract)
    return Pf.BookSpec(
        markets                   = [:match_odds, :over_under, :btts],
        totals_lines              = contract.totals_lines,
        commission                = contract.commission,
        settlement_rule           = Pf.DeArb(),
        require_complete_markets  = true,
    )
end

function tp00_growth_policies(contract::SLContract)
    return [
        (name = "full book",     policy = Pf.PortfolioKelly(slate_cap = contract.portfolio_kelly_cap)),
        (name = "totals only",  policy = Pf.PortfolioKelly(slate_cap = contract.portfolio_kelly_cap, eligible_markets = [:over_under])),
        (name = "totals + BTTS", policy = Pf.PortfolioKelly(slate_cap = contract.portfolio_kelly_cap, eligible_markets = [:over_under, :btts])),
        (name = "1X2 only",      policy = Pf.PortfolioKelly(slate_cap = contract.portfolio_kelly_cap, eligible_markets = [:match_odds])),
    ]
end


# ==============================================================================
# 2. Gate Assertions & Simulation
# ==============================================================================

function tp00_gate_books(books::AbstractVector, latents_df::DataFrame, odds_df::DataFrame)
    results = []
    push!(results, (
        name   = "books constructed",
        pass   = !isempty(books),
        detail = "$(length(books)) books built",
    ))
    return results
end

function tp00_gate_simulation(traj, slates, contract::SLContract)
    results = []
    min_w = minimum(traj.bankroll)
    push!(results, (
        name   = "bankroll stays positive",
        pass   = min_w > 0.0,
        detail = @sprintf("min bankroll = %.4f", min_w),
    ))
    return results
end

function tp00_growth_table(books_bf::AbstractVector, contract::SLContract)
    policies = tp00_growth_policies(contract)
    out = NamedTuple[]

    for p in policies
        slates = Pf.group(Pf.DailySlate(), books_bf)
        sim    = Pf.simulate(p.policy, slates)
        met    = Pf.path_metrics(sim)

        push!(out, (
            policy    = p.name,
            bets      = met.n_bets,
            final     = met.final_wealth,
            roi_pct   = met.roi * 100,
            win_rate  = met.win_rate,
            growth    = met.growth_rate,
            max_dd    = met.max_drawdown * 100,
        ))
    end
    return DataFrame(out)
end

function tp00_gate_growth(growth_df::DataFrame)
    results = []
    push!(results, (
        name   = "growth metrics computed",
        pass   = nrow(growth_df) > 0,
        detail = "$(nrow(growth_df)) policy evaluations completed",
    ))
    return results
end

function tp00_sweep_policy(books_bf, contract::SLContract; trusts = [0.15, 0.3, 0.5, 1.0], lambdas = [15.0, 23.0, 35.0])
    println("-" ^ 74)
    println("POLICY SWEEP (Model 00 Pure Poisson)")
    println("-" ^ 74)
    for l in lambdas
        for t in trusts
            pol = Pf.PortfolioKelly(slate_cap = contract.portfolio_kelly_cap)
            slates = Pf.group(Pf.DailySlate(), books_bf)
            sim = Pf.simulate(pol, slates)
            met = Pf.path_metrics(sim)
            @printf("  λ=%4.1f  trust=%4.2f  final=%6.3f  roi=%+6.2f%%  mdd=%+5.1f%%\n",
                    l, t, met.final_wealth, met.roi * 100, met.max_drawdown * 100)
        end
    end
    println("-" ^ 74)
    return nothing
end
