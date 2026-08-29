# ==============================================================================
# Scottish Lower — Compare Legacy vs Unified V2 Model Runs
# BayesianFootball.jl
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, Printf, Statistics, MCMCChains

println("\n" * "="^90)
println(" SCOTTISH LOWER: LEGACY VS UNIFIED V2 COMPARISON BENCHMARK")
println("="^90)

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

models_to_compare = [
    "m00_baseline",
    "m02_wealth",
    "m03_distance",
    "m04_joint"
]

save_root = "./data/scottish_lower_unified"

println("\nScanning for unified model fits in $save_root...")
fits = Dict{String, Fit}()

for m_name in models_to_compare
    dir = joinpath(save_root, m_name)
    if isdir(dir)
        found = list_fits(dir)
        if !isempty(found)
            latest = last(found)
            println(" Loading $(m_name): $(latest.path)")
            fits[m_name] = load_fit(latest.path)
        end
    end
end

if isempty(fits)
    println("No unified fits found yet in $save_root. Please run r11_train_all_2425_unified.jl first.")
    exit(0)
end

# Generate comparison leaderboard
println("\n" * "="^100)
println(" 1. CROSS-MODEL EVALUATION LEADERBOARD (Out-Of-Sample 24/25 Season)")
println("="^100)

fit_list = collect(values(fits))
lb = leaderboard(fit_list, ds; metric = :logloss)
display(lb)

println("\n" * "="^100)
println(" 2. PARAMETER POSTERIOR COMPARISON (Fold 1 Mean & 90% CrI)")
println("="^100)

@printf(" %-15s | %10s | %10s | %10s | %10s | %10s | %10s\n",
        "Model", "μ (Base)", "γ (HA)", "σ_att", "σ_def", "w_wealth", "w_dist")
println("-"^100)

for (name, fit) in fits
    ch = fit.folds[1].chain
    ch_vars = Set(Symbol.(names(ch)))
    
    get_stat(sym) = sym in ch_vars ? @sprintf("%.3f", mean(ch[sym])) : "—"
    
    mu_str   = get_stat(Symbol("inter.μ_base[1]"))
    ha_str   = Symbol("ha.γ") in ch_vars ? get_stat(Symbol("ha.γ")) : get_stat(Symbol("ha.γ_raw"))
    sa_str   = get_stat(Symbol("dyn.σ_a"))
    sd_str   = get_stat(Symbol("dyn.σ_d"))
    w_w_str  = get_stat(Symbol("wealth.w"))
    w_d_str  = get_stat(Symbol("distance.w"))
    
    @printf(" %-15s | %10s | %10s | %10s | %10s | %10s | %10s\n",
            name, mu_str, ha_str, sa_str, sd_str, w_w_str, w_d_str)
end

println("="^100)
println(" Benchmark report generated successfully.")
