# current_development/MetaModels/r03_meta_framework_runner.jl

using Pkg; Pkg.activate(".")
using BayesianFootball
using DataFrames
using Dates
using StatsPlots

using ThreadPinning
pinthreads(:cores)


# Include our new framework
include("src/MetaModels.jl")

include("./current_development/MetaModels/src/MetaModels.jl")
println("--- Meta Model Framework MVP Runner ---")
println("Note: This script assumes you have `ds` and `exp_results` loaded in your REPL from a previous Layer 1 run.")


ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())

save_dir = "./data/meta_model_layer1/"
saved_files = Experiments.list_experiments(save_dir, data_dir="")
exp_results = Experiments.load_experiment(saved_files, 1)

# 1. Configure Meta Model with Parametric Composition
println("1. Configuring ConvexMixtureMetaModel...")
meta_model = MetaModels.ConvexMixtureMetaModel(
    dynamics_config = MetaModels.MetaGRWDynamicsConfig(σ_prior=0.1),
    hierarchy_config = MetaModels.HierarchicalMetaTeamConfig(σ_team_prior=0.1)
)

sampler_config = Samplers.QueuedNUTSConfig(
    n_samples=500, n_chains=4, n_warmup=200,initialisation = Samplers.UniformInit(-2, 2),
)

# 2. Create Task
println("2. Creating MetaExperimentTask...")
meta_task = MetaModels.MetaExperimentTask(
    exp_results,
    meta_model,
    sampler_config,
    exp_results.config.splitter, # Reuse the L1 splitter
    :under_25 # Focus on the Over 1.5 goals market
)

# 3. Run Experiment
println("3. Running Meta Experiment (MVP)...")
meta_results, joined_data = MetaModels.run_meta_experiment(meta_task; ds=ds)

chain = meta_results.training_results[1]
describe(chain)

# 4. Extract and Plot Hierarchical Team Biases
println("4. Extracting Team Biases...")
team_names = unique(vcat(joined_data.home_team, joined_data.away_team))
team_map = Dict(t => i for (i, t) in enumerate(team_names))
n_teams = length(team_names)

delta_means = zeros(n_teams)
delta_stds = zeros(n_teams)

for (team, idx) in team_map
    sym = Symbol("δ_team[$idx]")
    if sym in keys(chain)
        delta_means[idx] = mean(chain[sym])
        delta_stds[idx] = std(chain[sym])
        println("Team Bias ($team): ", round(delta_means[idx], digits=3))
    end
end

p1 = bar(team_names, delta_means, yerror=delta_stds, 
         title="Hierarchical Team Bias (δ_team)\nPositive = L1 Underestimates Team", 
         rotation=45, legend=false, size=(800, 400))
display(p1)

println("\n=== Meta Model Framework MVP Complete! ===")



include("current_development/MetaModels/src/MetaModels.jl")
include("current_development/MetaModels/src/staking.jl")

# --- Compare 3 staking approaches ---

# 1. Raw L1 with BayesianKelly (using L1 PPD distribution directly)
# Note: joined_data.distribution contains the raw L1 PPD samples
l1_ledger = DataFrame(
match_id  = joined_data.match_id,
stake     = [compute_stake(BayesianKelly(min_edge=0.00), dist, odds)
                 for (dist, odds) in zip(joined_data.distribution, joined_data.odds_close)],
is_winner = joined_data.is_winner,
odds_close = joined_data.odds_close,
match_date = joined_data.match_date,
W         = joined_data.W
)

l1_ledger.pnl = [s > 0 ? (Bool(w) ? s*(o-1) : -s) : 0.0
                 for (s,w,o) in zip(l1_ledger.stake, l1_ledger.is_winner, l1_ledger.odds_close)]

# 2. Meta Model with BayesianKelly using full Q posterior
meta_ledger_df = compute_meta_stakes(
chain, joined_data;
signal = BayesianKelly(min_edge=0.00)
)

# 3. Meta Model with tighter edge filter
meta_ledger_tight = compute_meta_stakes(
chain, joined_data;
signal = BayesianKelly(min_edge=-0.01)
)

# --- Print comparisons ---
println("\n=== L1 Raw (BayesianKelly, 2% edge) ===")
bets_l1 = subset(l1_ledger, :stake => ByRow(>(0.0)))
println("Bets: $(nrow(bets_l1)) | PnL: $(round(sum(bets_l1.pnl), digits=4)) | ROI: $(round(sum(bets_l1.pnl)/sum(bets_l1.stake)*100, digits=2))%")

meta_ledger_summary(meta_ledger_df;   label="Meta Q Posterior (2% edge)")
meta_ledger_summary(meta_ledger_tight; label="Meta Q Posterior (5% edge)")


# Diagnostic: what does the Q edge distribution look like?
meta_ledger_diag = compute_meta_stakes(
    chain, joined_data;
signal = BayesianKelly(min_edge=0.0)
);

q_edges = meta_ledger_diag.Q_edge
println("Q Edge distribution (Meta vs Implied):")
println("  Mean:   ", round(mean(q_edges), digits=4))
println("  Median: ", round(median(q_edges), digits=4))
println("  Max:    ", round(maximum(q_edges), digits=4))
println("  % positive: ", round(mean(q_edges .> 0)*100, digits=1), "%")

# Compare: what were L1's edges on those same matches?
l1_edges = meta_ledger_diag.L1_edge
println("\nL1 Edge distribution (L1 vs Implied):")
println("  Mean:   ", round(mean(l1_edges), digits=4))
println("  Max:    ", round(maximum(l1_edges), digits=4))
println("  % positive: ", round(mean(l1_edges .> 0)*100, digits=1), "%")




