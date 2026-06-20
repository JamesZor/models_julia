# current_development/MetaModels/r04_queued_fold_runner.jl
#
# Full Meta Model runner with per-fold Queued NUTS training.
# One Meta Model chain per fold per CPU thread. Mirrors the Layer 1
# QueuedNUTSConfig approach so all cores stay busy throughout training.
#
# Prerequisites:
#   - julia --project -t 32  (or however many threads you have)
#   - A completed Layer 1 ExperimentResults saved under ./data/

using Pkg; Pkg.activate(".")
using BayesianFootball
using DataFrames
using Dates
using Statistics
using LogExpFunctions: logistic

using ThreadPinning
pinthreads(:cores)

# --- Force fresh load of the MetaModels prototype module ---
# If MetaModels is already defined in Main, remove it first so that
# struct redefinitions (MetaExperimentResults, MetaFoldResult, etc.) take effect.
if isdefined(Main, :MetaModels)
    println("Reloading MetaModels module...")
end
include("src/MetaModels.jl")

include("./current_development/MetaModels/src/MetaModels.jl")
using .MetaModels

# Also force-reload staking so it picks up the new MetaFoldResult type
include("src/staking.jl")

println("="^65)
println("  META MODEL — Queued Fold Runner (r04)")
println("="^65)

# ===========================================================================
# 1. LOAD DATA AND LAYER 1 RESULTS
# ===========================================================================

println("\n[1] Loading DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())

println("[2] Loading Layer 1 Experiment Results...")
save_dir   = "./data/meta_model_layer1/"
saved_files = BayesianFootball.Experiments.list_experiments(save_dir, data_dir="")
exp_results = BayesianFootball.Experiments.load_experiment(saved_files, 1)

println("    Loaded: $(exp_results.config.name)")
println("    Folds:  $(length(exp_results.training_results.items))")

# ===========================================================================
# 2. CONFIGURE META MODEL
# ===========================================================================
# Swap hierarchy_config to GlobalMetaHierarchyConfig() to remove team biases
# and run a faster global-only model.

println("\n[3] Configuring ConvexMixtureMetaModel...")

meta_model = ConvexMixtureMetaModel(
    dynamics_config  = MetaGRWDynamicsConfig(σ_prior=0.1),
    # hierarchy_config = HierarchicalMetaTeamConfig(σ_team_prior=0.1)
    hierarchy_config = GlobalMetaHierarchyConfig()  # ← swap for faster global-only run
)

# QueuedNUTSConfig: each chain runs on its own thread.
# max_concurrent_tasks controls how many chains run simultaneously.
# Set this to Threads.nthreads() to saturate the CPU.
sampler_config = BayesianFootball.Samplers.QueuedNUTSConfig(
    n_samples = 500,
    n_chains  = 4,
    n_warmup  = 200,
    accept_rate = 0.65,
    max_depth   = 10,
    initialisation = BayesianFootball.Samplers.UniformInit(-2, 2),
)

# ===========================================================================
# 3. CREATE TASK AND RUN
# ===========================================================================
# Change target_selection to whichever market you want to study.
# Good candidates to try in order of likely L1 edge:
#   :under_25, :home, :away, :over_25, :btts_yes

TARGET_SELECTION = :under_25

println("[4] Creating MetaExperimentTask for selection: $TARGET_SELECTION")
meta_task = MetaExperimentTask(
    exp_results,
    meta_model,
    sampler_config,
    exp_results.config.splitter,
    TARGET_SELECTION
)

println("[5] Running Queued Fold Experiment...")
println("    $(Threads.nthreads()) threads available\n")
_raw_result = MetaModels.run_meta_experiment(meta_task; ds=ds)

# Robust unpacking: handle old tuple-return (r03 cache) and new single-return
meta_results = if _raw_result isa Tuple
    @warn "Detected old tuple return from run_meta_experiment. Using first element.\n" *
          "Please restart your REPL and re-run r04 for a fully fresh load."
    _raw_result[1]
else
    _raw_result
end

n_folds = length(meta_results.fold_results)
println("\n    $(n_folds) fold(s) completed.")

# ===========================================================================
# 4. PER-FOLD CHAIN DIAGNOSTICS
# ===========================================================================

println("\n" * "="^65)
println("  PER-FOLD DIAGNOSTICS")
println("="^65)

using MCMCChains

for fr in meta_results.fold_results
    chain = fr.chain
    α_s   = vec(chain[:α_intercept])
    θ_g   = round(logistic(mean(α_s)), digits=3)
    acc   = round(mean(chain[:acceptance_rate]), digits=3)

    rhats    = MCMCChains.rhat(chain)
    n_bad_rhat = count(x -> !ismissing(x) && x > 1.05, rhats[:, :rhat])

    println("  Fold $(fr.fold_idx):  " *
            "n=$(nrow(fr.fold_data))  " *
            "θ_global=$(θ_g)  " *
            "accept=$(acc)  " *
            "rhat_issues=$(n_bad_rhat)/$(length(rhats[:, :rhat]))")
end

# ===========================================================================
# 5. STAKING ANALYSIS USING FULL Q POSTERIOR
# ===========================================================================
println("\n" * "="^65)
println("  STAKING ANALYSIS — Full Q Posterior (McHale BayesianKelly)")
println("="^65)

# Use the last fold's chain for staking (it was trained on most data)
# In production you'd use each fold's chain for its next-season predictions.
last_fold   = meta_results.fold_results[end]
chain       = last_fold.chain
joined_data = meta_results.all_data

# Filter joined_data to only the LAST fold's holdout season
# (avoid training-set contamination in the staking ledger)
last_fold_idx = last_fold.fold_idx
oos_data = subset(joined_data, :fold_idx => ByRow(==(last_fold_idx)))
println("  Evaluating on last fold holdout: $(nrow(oos_data)) matches\n")

#=
julia> println("  evaluating on last fold holdout: $(nrow(oos_data)) matches\n")
  evaluating on last fold holdout: 348 matches
=#


# --- L1 Raw Baseline ---
using BayesianFootball.Signals: BayesianKelly, compute_stake

l1_stakes = [compute_stake(BayesianKelly(min_edge=0.00), dist, odds)
             for (dist, odds) in zip(oos_data.distribution, oos_data.odds_close)]
l1_pnls   = [s > 0 ? (Bool(w) ? s*(o-1.0) : -s) : 0.0
             for (s, w, o) in zip(l1_stakes, oos_data.is_winner, oos_data.odds_close)]

bets_l1 = count(>(0), l1_stakes)
pnl_l1  = sum(l1_pnls)
roi_l1  = bets_l1 > 0 ? pnl_l1 / sum(l1_stakes[l1_stakes .> 0]) * 100 : 0.0
println("  L1 Raw (BayesianKelly 0% edge):")
println("    Bets=$(bets_l1) | PnL=$(round(pnl_l1, digits=4)) | ROI=$(round(roi_l1, digits=2))%\n")



#=
julia> println("  L1 Raw (BayesianKelly 0% edge):")
  L1 Raw (BayesianKelly 0% edge):

julia> println("    Bets=$(bets_l1) | PnL=$(round(pnl_l1, digits=4)) | ROI=$(round(roi_l1, digits=2))%\n")
    Bets=144 | PnL=0.0734 | ROI=1.96%
=#


# --- Meta Model Q Posterior ---
meta_ledger_df = MetaModels.compute_meta_stakes(
    chain, oos_data;
    signal  = BayesianKelly(min_edge=0.02),
    verbose = true
)
MetaModels.meta_ledger_summary(meta_ledger_df; label="Meta Q Posterior (2% edge)")

#=
julia> MetaModels.meta_ledger_summary(meta_ledger_df; label="Meta Q Posterior (2% edge)")

=================================================================
  STAKING LEDGER: Meta Q Posterior (2% edge)
=================================================================
  Matches evaluated: 348
  Bets placed:       6  (1.7% bet rate)
  Total Turnover:    0.1576 units
  Total PnL:         -0.0174 units
  ROI:               -11.06%
  Win Rate:          33.3%
  Avg Q_mean:        0.4
  Avg Implied Prob:  0.3696
  Avg Q Edge:        0.0304

  Weekly PnL (first 10 active weeks):
  ▼ Week 158: PnL=-0.028   | 1 bets | ROI=-100.0%
  ▲ Week 159: PnL=0.04     | 1 bets | ROI=162.5%
  ▼ Week 160: PnL=-0.016   | 1 bets | ROI=-100.0%
  ▼ Week 169: PnL=-0.019   | 1 bets | ROI=-100.0%
  ▲ Week 172: PnL=0.051    | 1 bets | ROI=200.0%
  ▼ Week 195: PnL=-0.044   | 1 bets | ROI=-100.0%

  Cumulative PnL: Best=0.027 | Worst=-0.0284 | Final=-0.0174
=================================================================
=#


meta_ledger_relaxed = MetaModels.compute_meta_stakes(
    chain, oos_data;
    signal  = BayesianKelly(min_edge=0.0),
    verbose = false
)
MetaModels.meta_ledger_summary(meta_ledger_relaxed; label="Meta Q Posterior (0% edge)")

#=
julia> MetaModels.meta_ledger_summary(meta_ledger_relaxed; label="Meta Q Posterior (0% edge)")

=================================================================
  STAKING LEDGER: Meta Q Posterior (0% edge)
=================================================================
  Matches evaluated: 348
  Bets placed:       48  (13.8% bet rate)
  Total Turnover:    0.3316 units
  Total PnL:         0.0104 units
  ROI:               3.13%
  Win Rate:          43.8%
  Avg Q_mean:        0.422
  Avg Implied Prob:  0.4115
  Avg Q Edge:        0.0105

  Weekly PnL (first 10 active weeks):
  ▼ Week 158: PnL=-0.028   | 1 bets | ROI=-100.0%
  ▲ Week 159: PnL=0.04     | 1 bets | ROI=162.5%
  ▼ Week 160: PnL=-0.016   | 1 bets | ROI=-100.0%
  ▼ Week 161: PnL=-0.017   | 2 bets | ROI=-100.0%
  ▲ Week 162: PnL=0.005    | 2 bets | ROI=50.0%
  ▼ Week 164: PnL=-0.01    | 1 bets | ROI=-100.0%
  ▲ Week 165: PnL=0.0      | 1 bets | ROI=162.5%
  ▼ Week 166: PnL=-0.01    | 2 bets | ROI=-100.0%
  ▼ Week 167: PnL=-0.002   | 1 bets | ROI=-100.0%
  ▼ Week 169: PnL=-0.022   | 2 bets | ROI=-100.0%

  Cumulative PnL: Best=0.0355 | Worst=-0.0614 | Final=0.0104
=================================================================
=#


# --- Edge Distribution Diagnostic ---
println("\n  Edge Distribution Diagnostic:")
q_edges = meta_ledger_relaxed.Q_edge
l1_edges = meta_ledger_relaxed.L1_edge
println("  " * rpad("", 30) * rpad("Q (Meta)", 15) * "L1 (Raw)")
println("  " * rpad("Mean edge", 30)    * rpad(string(round(mean(q_edges),   digits=4)), 15) * string(round(mean(l1_edges),   digits=4)))
println("  " * rpad("Max edge",  30)    * rpad(string(round(maximum(q_edges), digits=4)), 15) * string(round(maximum(l1_edges), digits=4)))
println("  " * rpad("% positive", 30)   * rpad(string(round(mean(q_edges .> 0)*100, digits=1)) * "%", 15) * string(round(mean(l1_edges .> 0)*100, digits=1)) * "%")

# ===========================================================================
# 6. FULL HEADLESS ANALYSIS (all folds combined)
# ===========================================================================
println("\n[6] Running full headless analysis on last fold chain...")
include("analysis_headless.jl")

println("\n" * "="^65)
println("  r04 COMPLETE")
println("="^65)



  # --- Regime-filtered L1 Staking ---
    # Use the Meta Model's weekly θ_t as a gate, not a blender.

    # 1. Get weekly θ_t from the fold-3 chain (trained on 22/23–24/25, predicts 25/26)
    fold3_chain = meta_results.fold_results[3].chain

    α_s   = vec(fold3_chain[:α_intercept])
    σ_GRW = vec(fold3_chain[Symbol("dyn_θ_logit.σ_GRW")])

    n_weeks_fold3 = maximum(meta_results.fold_results[3].fold_data.W)
    z_w_mat = hcat([vec(fold3_chain[Symbol("dyn_θ_logit.z_w[$w]")]) for w in 1:n_weeks_fold3]...)
    θ_t_samples = hcat([logistic.(α_s .+ cumsum(z_w_mat[s,:] .* σ_GRW[s]) .- mean(cumsum(z_w_mat[s,:] .* σ_GRW[s]))) for s in 1:length(α_s)]...)'
    θ_t_means = vec(mean(θ_t_samples, dims=1))

    # 2. The OOS data is fold 4 (25/26 season)
    # Map its weeks back to the fold-3 week index
    oos_data = meta_results.fold_results[4].fold_data  # 25/26 only
    oos_data.θ_t = [get(θ_t_means, w, mean(θ_t_means)) for w in oos_data.W]

    # 3. Threshold: only bet weeks where L1 is in a "good regime"
    θ_threshold = mean(θ_t_means)  # or try quantile(θ_t_means, 0.6) for tighter filter

    good_regime = subset(oos_data, :θ_t => ByRow(>=(θ_threshold)))
    bad_regime  = subset(oos_data, :θ_t => ByRow(<(θ_threshold)))

    # 4. Apply L1 BayesianKelly staking ONLY in good-regime weeks
    function simple_ledger(df, min_edge=0.02)
        stakes = [compute_stake(BayesianKelly(min_edge=min_edge), dist, odds)
                  for (dist, odds) in zip(df.distribution, df.odds_close)]
	pnls = [s > 0 ? (Bool(w) ? s*(o-1.0) : -s) : 0.0
                for (s,w,o) in zip(stakes, df.is_winner, df.odds_close)]
	n_bets = count(>(0), stakes)
	pnl    = sum(pnls)
	roi    = n_bets > 0 ? pnl / sum(filter(>(0), stakes)) * 100 : 0.0
	return n_bets, round(pnl, digits=4), round(roi, digits=2)
	end

    b_all,  p_all,  r_all  = simple_ledger(oos_data,   0.02)
    b_good, p_good, r_good = simple_ledger(good_regime, 0.02)
    b_bad,  p_bad,  r_bad  = simple_ledger(bad_regime,  0.02)

    println("ALL weeks:        Bets=$b_all  PnL=$p_all  ROI=$r_all%")
    println("GOOD regime only: Bets=$b_good PnL=$p_good ROI=$r_good%")
    println("BAD regime only:  Bets=$b_bad  PnL=$p_bad  ROI=$r_bad%")


   # CORRECTED: α_s[s] (scalar) not α_s (2000-vector)
    θ_t_mat = Matrix{Float64}(undef, length(α_s), n_weeks_fold3)
    for s in 1:length(α_s)
        drift = cumsum(z_w_mat[s, :] .* σ_GRW[s])
        drift_centered = drift .- mean(drift)
	θ_t_mat[s, :] = logistic.(α_s[s] .+ drift_centered)
    end

    θ_t_means = vec(mean(θ_t_mat, dims=1))  # (n_weeks,)
    θ_t_stds  = vec(std(θ_t_mat,  dims=1))

    println("θ_t range: [$(round(minimum(θ_t_means), digits=3)), $(round(maximum(θ_t_means), digits=3))]")
    println("θ_t std across weeks: $(round(std(θ_t_means), digits=4))")

  # Then the regime filter:

    # Fold 4 OOS data (25/26 season) — use fold 3's chain for no-leakage
    oos_data = subset(meta_results.all_data, :fold_idx => ByRow(==(4)))

    # Map OOS weeks to fold-3 week index
    # The fold-3 W column covers weeks 1..145, fold-4 goes 1..196 globally.
    # We clamp to the max week fold-3 saw and use the last known θ for future weeks.
    oos_data.θ_t = [
	w <= n_weeks_fold3 ? θ_t_means[w] : θ_t_means[end]
	for w in oos_data.W
    ]

    θ_threshold = median(θ_t_means)
    good_regime = subset(oos_data, :θ_t => ByRow(>=(θ_threshold)))
    bad_regime  = subset(oos_data, :θ_t => ByRow(<(θ_threshold)))

    println("\nRegime split: $(nrow(good_regime)) good / $(nrow(bad_regime)) bad matches")

    function simple_ledger(df; min_edge=0.00)
	stakes = [compute_stake(BayesianKelly(min_edge=min_edge), dist, odds)
                  for (dist, odds) in zip(df.distribution, df.odds_close)]
	pnls = [s > 0 ? (Bool(w) ? s*(o-1.0) : -s) : 0.0
		for (s,w,o) in zip(stakes, df.is_winner, df.odds_close)]
	n    = count(>(0), stakes)
	pnl  = sum(pnls)
	roi  = n > 0 ? pnl / sum(filter(>(0), stakes)) * 100 : 0.0
	return (n_bets=n, pnl=round(pnl,digits=4), roi=round(roi,digits=2))
    end

    all_r  = simple_ledger(oos_data;   min_edge=0.00)
    good_r = simple_ledger(good_regime; min_edge=0.00)
    bad_r  = simple_ledger(bad_regime;  min_edge=0.00)

    println("\nALL  weeks: Bets=$(all_r.n_bets)  PnL=$(all_r.pnl)  ROI=$(all_r.roi)%")
    println("GOOD weeks: Bets=$(good_r.n_bets)  PnL=$(good_r.pnl)  ROI=$(good_r.roi)%")
    println("BAD  weeks: Bets=$(bad_r.n_bets)  PnL=$(bad_r.pnl)  ROI=$(bad_r.roi)%")
