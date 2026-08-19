# current_development/scottish_lower/neg_bin/r06_grid_negbin_wealth.jl
#
# RUNNER: 40-Fold MCMC Rolling Grid for Scottish NegBin + Wealth Models
#
# Runs Queued NUTS MCMC (16 concurrent worker tasks, 3 chains x 500 samples)
# across 40 rolling out-of-sample splits on Scottish Lower leagues (56, 57).
#
# Grid Architectures:
# 1. goals_negbin_wealth_hl365_hs2   (Goals-Only NegBin + Starting XI Wealth)
# 2. pxg_apm_negbin_wealth_hl365_hs2 (Arm A: Proxy xG + RAPM + Wealth + NegBin)
# 3. funnel_pxg_apm_negbin_wealth_hl365_hs2 (Arm B: 3-Layer Funnel + Quality + RAPM + Wealth + NegBin)

using Revise
using BayesianFootball
using Turing
using Dates
using Printf

const Data        = BayesianFootball.Data
const PreGame     = BayesianFootball.Models.PreGame
const Samplers    = BayesianFootball.Samplers
const Experiments = BayesianFootball.Experiments

include("l02_negbin_wealth_engines.jl")

println("==================================================================")
println(" SCOTTISH LOWER NEGBIN + WEALTH 40-FOLD MCMC ROLLING GRID")
println("==================================================================")

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower())
println("✓ Loaded Scottish Lower DataStore (", length(ds.matches.match_id), " matches)")

# 2. Configure 40-Fold Rolling Splitter
splitter = PreGame.GroupedCVConfig(
    group_by = :date,
    n_splits = 40,
    min_train_size = 500,
    test_size = 15,
    step_size = 10,
    window_strategy = :expanding
)

# 3. High-Performance Queued NUTS Sampler
sampler_cfg = Samplers.QueuedNUTSConfig(
    n_samples            = 500,
    n_adapts             = 500,
    target_accept        = 0.85,
    n_chains             = 3,
    max_concurrent_tasks = 16
)

# 4. Grid Models
grid_configs = [
    (
        name = "goals_negbin_wealth_hl365_hs2",
        model = TeamGoalsNegBinWealthModel(
            dynamics_config = PreGame.TimeDecayDynamics(days_half_life = 365.0),
            homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage(),
            dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
            apm_on = true,
            league_ha_on = true
        )
    ),
    (
        name = "pxg_apm_negbin_wealth_hl365_hs2",
        model = TeamPxGGoalsAPMNegBinWealthModel(
            dynamics_config = PreGame.TimeDecayDynamics(days_half_life = 365.0),
            homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage(),
            dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
            apm_on = true,
            league_ha_on = true
        )
    ),
    (
        name = "funnel_pxg_apm_negbin_wealth_hl365_hs2",
        model = TeamFunnelPxGGoalsAPMNegBinWealthModel(
            dynamics_config = PreGame.TimeDecayDynamics(days_half_life = 365.0),
            homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage(),
            dispersion_config = SCOTTISH_HOMEAWAY_DISPERSION,
            apm_on = true,
            team_quality_on = true,
            league_ha_on = true
        )
    )
]

save_dir = "data/scottish_negbin_wealth_grid"
mkpath(save_dir)

# 5. Execute Grid
for (idx, cfg) in enumerate(grid_configs)
    println("\n==================================================================")
    println("▶ [", idx, "/", length(grid_configs), "] Launching Grid: ", cfg.name)
    println("==================================================================")
    
    t_start = time()
    task = Experiments.create_experiment_task(
        ds,
        cfg.model,
        cfg.name,
        save_dir;
        splitter = splitter,
        sampler_config = sampler_cfg,
        metadata = Dict(
            :league => "ScottishLower",
            :tournaments => [56, 57],
            :notes => "Scottish Lower Robust NegBin (NB2) + Starting XI Wealth Delta",
            :created_at => Dates.now()
        )
    )
    
    res = Experiments.run_experiment(task)
    Experiments.save_experiment(res)
    
    elapsed_min = (time() - t_start) / 60.0
    println("✓ Saved: ", cfg.name, " in ", round(elapsed_min, digits=1), " mins")
end

println("\n==================================================================")
println("✓ 40-Fold NegBin + Wealth Grid Execution Complete!")
println("==================================================================")
