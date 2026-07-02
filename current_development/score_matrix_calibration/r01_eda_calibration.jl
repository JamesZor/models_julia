#=
r01_eda_calibration.jl

Runner file to:
1. Load `li_smile50` model and the Betfair odds data.
2. Evaluate historical bias on key markets (btts_yes, home, over_25).
3. Compute the Global Bias (all data pooled) vs. Walk-Forward time-decay bias.
4. Compare the parameters and the resulting calibration on the Score Matrix.
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics
using Dates

const Evaluation = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Data = BayesianFootball.Data

# Resolve loader paths from the package root so this runs identically whether the file is
# `include()`d (nested includes are otherwise resolved relative to THIS file's dir) or pasted.
const _ROOT = pkgdir(BayesianFootball)
include(joinpath(_ROOT, "current_development/split_market_pillar/l03_local_intensity_poisson.jl"))
include(joinpath(_ROOT, "current_development/score_matrix_calibration/l01_score_matrix_calibration.jl"))

println("[INFO] Loading Ireland dataset...")
ds = Data.load_datastore_cached(Data.Ireland(); max_age_hours=99999)
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds1 = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

smile_dir = "./data/double_poisson_smile_grid/"
exps = Experiments.list_experiments(smile_dir; data_dir="")
all_results = Experiments.load_experiments(exps)
exp_smile = all_results[findfirst(r -> r.config.name == "li_smile50", all_results)]

println("[INFO] Extracting out-of-sample predictions for li_smile50...")
latents = Experiments.extract_oos_predictions(ds1, exp_smile)
ppd = Predictions.model_inference(latents)

mf = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
select!(mf, :match_id, :market_name, :market_line, :selection, :prob_model)
adf = innerjoin(ds1.odds, mf, on = [:match_id, :market_name, :market_line, :selection])
adf = innerjoin(adf, ds1.matches[:, [:match_id, :match_date]], on=:match_id)
dropmissing!(adf, [:prob_fair_close, :is_winner])
adf.spread = Float64.(adf.prob_model) .- Float64.(adf.prob_fair_close)

selections_to_test = [:btts_yes, :home, :over_25]

println("\n" * "="^80)
println("1. PARAMETER COMPARISON: Global vs Walk-Forward (Half-life: 90 days)")
println("="^80)

match_ids = unique(adf.match_id)
N_matches = length(match_ids)

# We will store the gammas for later score matrix calibration
gammas_global = Dict{Symbol, Float64}()
gammas_wf = Dict{Symbol, Dict{eltype(match_ids), Float64}}()

for sel in selections_to_test
    df_sel = adf[Symbol.(adf.selection) .== sel, :]
    
    println("Market: $(sel) | Found $(nrow(df_sel)) matches")
    
    g_global = fit_global_bias(df_sel)
    g_wf = fit_walk_forward_bias(df_sel; half_life_days=90.0)
    
    gammas_global[sel] = g_global
    gammas_wf[sel] = g_wf
    
    # Analyze walk forward stats
    wf_vals = collect(values(g_wf))
    filter!(v -> v != 0.0, wf_vals) # filter out uninitialized days
    med_wf = isempty(wf_vals) ? 0.0 : median(wf_vals)
    min_wf = isempty(wf_vals) ? 0.0 : minimum(wf_vals)
    max_wf = isempty(wf_vals) ? 0.0 : maximum(wf_vals)
    
    println("Market: $(sel)")
    println("  Global Gamma       : $(round(g_global, digits=4))")
    println("  Walk-Forward Median: $(round(med_wf, digits=4))  (Range: [$(round(min_wf, digits=4)), $(round(max_wf, digits=4))])")
end

println("\n" * "="^80)
println("2. SCORE MATRIX TILT VALIDATION (on btts_yes)")
println("="^80)

# Let's test the exponential tilt directly on the score matrix for btts_yes
target_sel = :btts_yes
mask_btts = mask_for("BTTS", "", "btts_yes")

# We'll calculate the model probability of btts_yes before and after tilting.
# For simplicity, we just rebuild the state probability matrices.
results = DataFrame(match_id=eltype(match_ids)[], raw_p=Float64[], tilted_global_p=Float64[], tilted_wf_p=Float64[])

for r in eachrow(latents.df)
    mid = r.match_id
    # Ensure this match is in our adf
    mid in match_ids || continue
    
    λh, λa = r.λ_h, r.λ_a
    S = length(λh)
    P_raw = Matrix{Float64}(undef, GG * GG, S)
    for s in 1:S
        ph = pdf.(Poisson(λh[s]), 0:GG-1)
        pa = pdf.(Poisson(λa[s]), 0:GG-1)
        g = vec(ph * pa')
        P_raw[:, s] = g ./ sum(g)
    end
    
    # 1. Raw Probability
    raw_prob = mean(sum(P_raw[mask_btts, :], dims=1))
    
    # 2. Global Tilted Probability
    P_global = copy(P_raw)
    tilt_score_matrix!(P_global, [mask_btts], [gammas_global[target_sel]])
    global_prob = mean(sum(P_global[mask_btts, :], dims=1))
    
    # 3. Walk-Forward Tilted Probability
    g_wf = get(gammas_wf[target_sel], mid, 0.0)
    P_wf = copy(P_raw)
    tilt_score_matrix!(P_wf, [mask_btts], [g_wf])
    wf_prob = mean(sum(P_wf[mask_btts, :], dims=1))
    
    push!(results, (mid, raw_prob, global_prob, wf_prob))
end

# Merge with market data to compute bias and t-stats.
# NOTE: adf.selection is a Symbol column (mf.selection from model_inference is Symbol[]),
# so we must compare against :btts_yes, not the String "btts_yes" (which matches nothing).
df_eval = innerjoin(results, adf[Symbol.(adf.selection) .== :btts_yes, [:match_id, :prob_fair_close]], on=:match_id)

function print_bias_stats(name, model_p, market_p)
    sp = model_p .- market_p
    n = length(sp)
    b = mean(sp)
    se = std(sp) / sqrt(n)
    t = b / se
    println("  $(rpad(name, 15)): bias = $(round(b, digits=4)), t = $(round(t, digits=2))")
end

println("BTTS YES Calibration Results (n = $(nrow(df_eval)) matches):")
print_bias_stats("Raw Model", df_eval.raw_p, df_eval.prob_fair_close)
print_bias_stats("Global Tilt", df_eval.tilted_global_p, df_eval.prob_fair_close)
print_bias_stats("WF Tilt", df_eval.tilted_wf_p, df_eval.prob_fair_close)

println("\nDone.")
