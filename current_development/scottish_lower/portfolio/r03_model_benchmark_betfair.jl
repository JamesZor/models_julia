# current_development/scottish_lower_portfolio/r03_model_benchmark_betfair.jl
#
# 5-WAY HEAD-TO-HEAD PORTFOLIO BENCHMARK ON BETFAIR EXCHANGE ODDS
#
# Compares the 5 Scottish Lower models under calibrated portfolio policies on Betfair Exchange:
# 1. funnel_apm_ctl_hl365_hs2  (Baseline: Shots + RAPM)
# 2. pxg_apm_hl365_hs2         (Arm A: Proxy xG + RAPM)
# 3. pxg_noapm_hl365_hs2       (Pure Team: Proxy xG Only)
# 4. funnel_pxg_apm_hl365_hs2  (Arm B: 3-Layer Volume -> Quality -> Goals + RAPM)
# 5. pxg_apm_linvar_hl365_hs2  (Linear Variance: Proxy xG + RAPM)
#
# Evaluates bankroll trajectory, risk-adjusted returns (Sharpe/Calmar), and drawdown control.

include("_setup_scottish_betfair.jl")

println("\n", "="^95)
println("5-WAY SCOTTISH LOWER PORTFOLIO BENCHMARK (BETFAIR EXCHANGE)")
println("="^95)

# Load MatchBooks for all 5 models
all_books = Dict{String, Vector{PF.MatchBook}}()
all_slates = Dict{String, Vector{PF.Slate}}()

for (m_name, _) in all_exprs
    cache_file = joinpath(CACHE_DIR, "books_$(m_name)_betfair_bm800.jls")
    isfile(cache_file) || error("Cache file missing for $m_name. Run r01_build_books_betfair.jl first.")
    b = deserialize(cache_file)
    all_books[m_name] = b
    all_slates[m_name] = PF.group(PF.DailySlate(), b)
end

# Define standard reference policies
policies = [
    ("Conservative (Cap 10%, λ=23)", PF.PolicySpec(
        trust    = PF.FlatTrust(0.25),
        risk     = PF.SlateDrawdown(23.0),
        cap      = PF.FixedCap(0.10),
        filter   = PF.KeepAll(),
        grouping = PF.DailySlate()
    )),
    ("Balanced Growth (Cap 15%, λ=15)", PF.PolicySpec(
        trust    = PF.FlatTrust(0.25),
        risk     = PF.SlateDrawdown(15.0),
        cap      = PF.FixedCap(0.15),
        filter   = PF.KeepAll(),
        grouping = PF.DailySlate()
    )),
    ("Aggressive (Cap 25%, λ=10)", PF.PolicySpec(
        trust    = PF.FlatTrust(0.50),
        risk     = PF.SlateDrawdown(10.0),
        cap      = PF.FixedCap(0.25),
        filter   = PF.KeepAll(),
        grouping = PF.DailySlate()
    ))
]

model_order = [
    "funnel_apm_ctl_hl365_hs2",
    "pxg_apm_hl365_hs2",
    "pxg_noapm_hl365_hs2",
    "funnel_pxg_apm_hl365_hs2",
    "pxg_apm_linvar_hl365_hs2"
]
present_models = filter(m -> haskey(all_slates, m), model_order)

for (pol_name, pol) in policies
    println("\n", "="^95)
    println("BETFAIR BENCHMARK RESULTS: $pol_name (2% Comm, Baker-McHale 800 Draws)")
    println("="^95)
    
    res_df = DataFrame(
        model = String[],
        final_wealth = Float64[],
        growth_slate = Float64[],
        roi_pct = Float64[],
        mean_expo = Float64[],
        mdd_pct = Float64[],
        sharpe = Float64[],
        calmar = Float64[],
        n_bets = Int[]
    )
    
    for m_name in present_models
        slates = all_slates[m_name]
        traj = PF.simulate(pol, slates; use_shrink = true)
        m = PF.path_metrics(traj)
        
        # Risk metrics
        ret_series = traj.slate_pl
        sh = length(ret_series) > 1 && std(ret_series) > 1e-6 ? (mean(ret_series) / std(ret_series)) * sqrt(35) : 0.0
        calm = m.mdd > 0.0 ? (m.final - 1.0) / (m.mdd / 100.0) : 0.0
        
        # Count total individual bets placed
        total_bets = sum(sum(b.a_kelly .> 1e-5) for b in all_books[m_name])
        
        push!(res_df, (
            m_name,
            round(m.final, digits = 3),
            round(m.growth_per_slate, digits = 5),
            round(m.roi, digits = 2),
            round(m.mean_exposure * 100, digits = 1),
            round(m.mdd, digits = 2),
            round(sh, digits = 2),
            round(calm, digits = 2),
            total_bets
        ))
    end
    
    show(res_df; allrows = true, allcols = true, truncate = 0)
    println()
end

println("\n", "="^95)
println("BETFAIR PORTFOLIO BENCHMARK COMPLETE")
println("="^95)
