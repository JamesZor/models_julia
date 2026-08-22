# current_development/scottish_lower/open_play/r14_trust_sensitivity_sweep.jl
#
# EMPIRICAL SWEEP: Model Trust Weight (w in [0.05, 1.00]) Sensitivity Analysis
#
# Analyzes what happens to ROI, Max Drawdown, Final Wealth, and Sharpe ratio
# as we increase trust in the model (w * p_model + (1 - w) * p_market).

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const Portfolio = BayesianFootball.Portfolio
const CACHE_DIR = joinpath(@__DIR__, "cache")

println("="^95)
println("🔬 TRUST SENSITIVITY SWEEP: What Happens When We Trust the Model More?")
println("="^95)

# Load cached books
model_keys = [
    ("recomb_pois_wealth_integrated_hl365_hs2", "Recomb Pois + Squad Wealth (Champion)"),
    ("recomb_pois_integrated_hl365_hs2",        "Recomb Pois Integrated"),
    ("recomb_negbin_integrated_hl365_hs2",      "Recomb NegBin Integrated"),
    ("goals_pois_ctl_hl365_hs2",                "Gross Goals Poisson (Control)")
]

books_map = Dict{String, Vector{Portfolio.MatchBook}}()
for (m_name, label) in model_keys
    cache_file = joinpath(CACHE_DIR, "books_bf_$(m_name)_bm800.jls")
    if isfile(cache_file)
        books_map[m_name] = deserialize(cache_file)
    else
        @warn "Cache file missing for $m_name: $cache_file"
    end
end

slates_map = Dict{String, Vector{Portfolio.Slate}}()
for (m_name, b) in books_map
    slates_map[m_name] = Portfolio.group(Portfolio.DailySlate(), b)
end

trust_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

# ==============================================================================
# 1. FIXED CAP 15%, SLATE DRAWDOWN LAMBDA = 15.0
# ==============================================================================
println("\n" * "="^100)
println("📊 EXPERIMENT 1: BALANCED RISK REGIME (Cap = 15%, λ = 15.0, 710 Test Matches)")
println("="^100)

champ_name = "recomb_pois_wealth_integrated_hl365_hs2"
champ_slates = slates_map[champ_name]

sweep_df = DataFrame(
    trust_w      = Float64[],
    model_pct    = String[],
    final_wealth = Float64[],
    growth_slate = Float64[],
    roi_pct      = Float64[],
    mean_expo    = Float64[],
    mdd_pct      = Float64[],
    sharpe       = Float64[]
)

for w in trust_levels
    pol = Portfolio.PolicySpec(
        trust    = Portfolio.FlatTrust(w),
        risk     = Portfolio.SlateDrawdown(15.0),
        cap      = Portfolio.FixedCap(0.15),
        filter   = Portfolio.KeepAll(),
        grouping = Portfolio.DailySlate()
    )
    traj = Portfolio.simulate(pol, champ_slates; use_shrink = true)
    m = Portfolio.path_metrics(traj)
    
    ret_series = traj.slate_pl
    sh = length(ret_series) > 1 && std(ret_series) > 1e-6 ? (mean(ret_series) / std(ret_series)) * sqrt(35) : 0.0
    
    push!(sweep_df, (
        trust_w      = w,
        model_pct    = "$(Int(round(w*100)))% Model / $(Int(round((1-w)*100)))% Market",
        final_wealth = round(m.final, digits=3),
        growth_slate = round(m.growth_per_slate, digits=5),
        roi_pct      = round(m.roi, digits=2),
        mean_expo    = round(m.mean_exposure * 100, digits=1),
        mdd_pct      = round(m.mdd, digits=2),
        sharpe       = round(sh, digits=2)
    ))
end

show(stdout, MIME("text/plain"), sweep_df)
println()

# ==============================================================================
# 2. AGGRESSIVE RISK REGIME (Cap = 25%, LAMBDA = 10.0)
# ==============================================================================
println("\n" * "="^100)
println("📊 EXPERIMENT 2: AGGRESSIVE RISK REGIME (Cap = 25%, λ = 10.0, 710 Test Matches)")
println("="^100)

sweep_agg_df = DataFrame(
    trust_w      = Float64[],
    model_pct    = String[],
    final_wealth = Float64[],
    growth_slate = Float64[],
    roi_pct      = Float64[],
    mean_expo    = Float64[],
    mdd_pct      = Float64[],
    sharpe       = Float64[]
)

for w in trust_levels
    pol = Portfolio.PolicySpec(
        trust    = Portfolio.FlatTrust(w),
        risk     = Portfolio.SlateDrawdown(10.0),
        cap      = Portfolio.FixedCap(0.25),
        filter   = Portfolio.KeepAll(),
        grouping = Portfolio.DailySlate()
    )
    traj = Portfolio.simulate(pol, champ_slates; use_shrink = true)
    m = Portfolio.path_metrics(traj)
    
    ret_series = traj.slate_pl
    sh = length(ret_series) > 1 && std(ret_series) > 1e-6 ? (mean(ret_series) / std(ret_series)) * sqrt(35) : 0.0
    
    push!(sweep_agg_df, (
        trust_w      = w,
        model_pct    = "$(Int(round(w*100)))% Model / $(Int(round((1-w)*100)))% Market",
        final_wealth = round(m.final, digits=3),
        growth_slate = round(m.growth_per_slate, digits=5),
        roi_pct      = round(m.roi, digits=2),
        mean_expo    = round(m.mean_exposure * 100, digits=1),
        mdd_pct      = round(m.mdd, digits=2),
        sharpe       = round(sh, digits=2)
    ))
end

show(stdout, MIME("text/plain"), sweep_agg_df)
println()

# ==============================================================================
# 3. UNCONSTRAINED FULL-KELLY (No Risk Drawdown Shading, Cap = 50%)
# ==============================================================================
println("\n" * "="^100)
println("📊 EXPERIMENT 3: UNCONSTRAINED FULL-KELLY (Cap = 50%, No Risk Constraint, 710 Test Matches)")
println("="^100)

sweep_uncon_df = DataFrame(
    trust_w      = Float64[],
    model_pct    = String[],
    final_wealth = Float64[],
    growth_slate = Float64[],
    roi_pct      = Float64[],
    mean_expo    = Float64[],
    mdd_pct      = Float64[],
    sharpe       = Float64[]
)

for w in [0.10, 0.25, 0.30, 0.50, 0.75, 1.00]
    pol = Portfolio.PolicySpec(
        trust    = Portfolio.FlatTrust(w),
        risk     = Portfolio.NoRiskConstraint(),
        cap      = Portfolio.FixedCap(0.50),
        filter   = Portfolio.KeepAll(),
        grouping = Portfolio.DailySlate()
    )
    traj = Portfolio.simulate(pol, champ_slates; use_shrink = true)
    m = Portfolio.path_metrics(traj)
    
    ret_series = traj.slate_pl
    sh = length(ret_series) > 1 && std(ret_series) > 1e-6 ? (mean(ret_series) / std(ret_series)) * sqrt(35) : 0.0
    
    push!(sweep_uncon_df, (
        trust_w      = w,
        model_pct    = "$(Int(round(w*100)))% Model / $(Int(round((1-w)*100)))% Market",
        final_wealth = round(m.final, digits=3),
        growth_slate = round(m.growth_per_slate, digits=5),
        roi_pct      = round(m.roi, digits=2),
        mean_expo    = round(m.mean_exposure * 100, digits=1),
        mdd_pct      = round(m.mdd, digits=2),
        sharpe       = round(sh, digits=2)
    ))
end

show(stdout, MIME("text/plain"), sweep_uncon_df)
println()
