# current_development/MetaModels/src/training/workflow.jl

using DataFrames
using Dates
using Turing
using Base.Threads
using ProgressMeter
using Printf
import BayesianFootball.Data
import BayesianFootball.Experiments
import BayesianFootball.Samplers
import BayesianFootball.Predictions
import MCMCChains: chainscat

export MetaExperimentTask, MetaExperimentResults, run_meta_experiment, build_meta_fold_data, run_multi_market_experiments

# ===========================================================================
# TYPES
# ===========================================================================

struct MetaExperimentTask
    base_results::Experiments.ExperimentResults
    meta_model::AbstractMetaModel
    sampler_config::Samplers.AbstractNUTSConfig
    splitter::Data.AbstractSplitter
    target_selection::Symbol # e.g. :over_15
end

struct MetaFoldResult
    chain::Any           # Combined MCMCChains.Chains for this fold
    fold_data::DataFrame # The OOS data this fold was trained on
    fold_idx::Int
end

struct MetaExperimentResults
    task::MetaExperimentTask
    fold_results::Vector{MetaFoldResult}
    # Combined data across all folds (for staking analysis)
    all_data::DataFrame
end

# ===========================================================================
# DATA PREPARATION
# ===========================================================================

"""
    build_meta_fold_data(latent_states, ppd, ds, target_selection) -> Vector{DataFrame}

For each L1 training fold's OOS window, builds the joined DataFrame
(PPD + odds + match metadata) for the Meta Model to train on.

The fold windows are inferred from the GroupedSplitMetaData stored in the
L1 ExperimentResults training_results.
"""
function build_meta_fold_data(
    latent_states,
    ppd,
    ds::Data.DataStore,
    target_selection::Symbol
)
    # 1. Filter PPD and odds to the target market
    ppd_sel  = subset(ppd.df,  :selection => ByRow(isequal(target_selection)))
    odds_sel = subset(ds.odds, :selection => ByRow(isequal(target_selection)))

    # 2. Join everything: PPD + odds + match date + teams
    joined = innerjoin(ppd_sel,  odds_sel[!, [:match_id, :prob_fair_close, :odds_close, :is_winner]], on=:match_id)
    joined = innerjoin(joined,   ds.matches[!, [:match_id, :home_team, :away_team, :match_date]], on=:match_id)
    dropmissing!(joined, [:is_winner, :prob_fair_close, :match_date, :distribution])
    sort!(joined, :match_date)

    # 3. Compute global week index (used to build θ_t trajectory)
    start_date = Date(minimum(joined.match_date))
    joined.W_global = [(Date(d) - start_date).value ÷ 7 + 1 for d in joined.match_date]

    # 4. Compute L1 mean probability from PPD distribution
    joined.p_L1 = [mean(dist) for dist in joined.distribution]

    # 5. Build team index (global across all data to keep consistent)
    unique_teams = unique(vcat(joined.home_team, joined.away_team))
    team_map     = Dict(t => i for (i, t) in enumerate(unique_teams))
    joined.home_idx = [team_map[t] for t in joined.home_team]
    joined.away_idx = [team_map[t] for t in joined.away_team]
    n_teams = length(unique_teams)

    # 6. Split into folds by match_date windows
    #    The L1 training_results carries GroupedSplitMetaData with season/week info.
    #    We reconstruct fold boundaries from the unique (season, week) pairs in that metadata.
    fold_dfs = DataFrame[]
    training_results = latent_states.model  # The L1 model is stored in latent_states

    # Simpler approach: derive fold windows from the sorted match dates in joined
    # Each fold covers ~1 season's worth of OOS weeks from the L1 GroupedCV splitter
    # We just use the actual L1 split metadata baked into joined via match_id lookup
    # For the MVP, we split evenly by match_date into n_splits buckets
    # matching the number of L1 folds:
    n_l1_folds = length(latent_states.model == nothing ? [] : [])

    return joined, team_map, n_teams
end


# ===========================================================================
# MAIN WORKFLOW
# ===========================================================================

"""
    run_meta_experiment(task::MetaExperimentTask; ds::Data.DataStore)

Executes the Meta Model cross-validation workflow using a Queued NUTS approach.
One Meta Model is trained per L1 fold window. All fold × chain tasks are
flattened into a single queue and run concurrently, keeping every CPU busy.
"""
function run_meta_experiment(task::MetaExperimentTask; ds::Data.DataStore)

    println("\n>> META MODEL EXPERIMENT: $(task.target_selection)")
    println("-"^60)

    # ------------------------------------------------------------------
    # 1. Extract L1 OOS predictions and generate PPD
    # ------------------------------------------------------------------
    println("[1] Extracting Layer 1 OOS Predictions...")
    latent_states = Experiments.extract_oos_predictions(ds, task.base_results)

    println("[2] Generating Posterior Predictive Distributions (PPD)...")
    ppd = Predictions.model_inference(latent_states)

    # ------------------------------------------------------------------
    # 2. Build per-fold data from L1 fold metadata
    # ------------------------------------------------------------------
    println("[3] Building fold data for selection: $(task.target_selection)")

    ppd_sel  = subset(ppd.df,  :selection => ByRow(isequal(task.target_selection)))
    odds_sel = subset(ds.odds, :selection => ByRow(isequal(task.target_selection)))

    joined = innerjoin(ppd_sel,  odds_sel[!, [:match_id, :prob_fair_close, :odds_close, :is_winner]], on=:match_id)
    joined = innerjoin(joined,   ds.matches[!, [:match_id, :home_team, :away_team, :match_date]], on=:match_id)
    dropmissing!(joined, [:is_winner, :prob_fair_close, :match_date, :distribution])
    sort!(joined, :match_date)

    # Build global week and team indices
    start_date  = Date(minimum(joined.match_date))
    joined.W    = [(Date(d) - start_date).value ÷ 7 + 1 for d in joined.match_date]
    joined.p_L1 = [mean(dist) for dist in joined.distribution]

    unique_teams = unique(vcat(joined.home_team, joined.away_team))
    team_map     = Dict(t => i for (i, t) in enumerate(unique_teams))
    joined.home_idx = [get(team_map, t, 1) for t in joined.home_team]
    joined.away_idx = [get(team_map, t, 1) for t in joined.away_team]
    n_teams = length(unique_teams)

    # ------------------------------------------------------------------
    # 3. Derive fold windows from weekly time steps
    # ------------------------------------------------------------------
    weeks = sort(unique(joined.W))
    n_folds = length(weeks)
    println("    Found $n_folds weekly fold windows.")

    # We map fold_idx directly to the sorted week index
    week_to_fold = Dict(w => i for (i, w) in enumerate(weeks))
    joined.fold_idx = [get(week_to_fold, w, 0) for w in joined.W]
    filter!(row -> row.fold_idx > 0, joined)

    # ------------------------------------------------------------------
    # 4. Build MetaModelData for each fold
    #    The Meta Model for fold k trains on all data UP TO and including
    #    fold k (OOS from L1). This is the correct no-leakage structure:
    #    - Fold 1: train meta on season 1 OOS
    #    - Fold 2: train meta on seasons 1+2 OOS (expanding window)
    #    - Fold k: train on seasons 1..k OOS
    # ------------------------------------------------------------------
    fold_datasets = Vector{Tuple{MetaModelData, DataFrame}}()
    for k in 1:n_folds
        fold_df = subset(joined, :fold_idx => ByRow(<=(k)))
        if nrow(fold_df) == 0
            continue
        end
        # Re-index weeks locally for this fold window
        fold_start = Date(minimum(fold_df.match_date))
        local_W    = [(Date(d) - fold_start).value ÷ 7 + 1 for d in fold_df.match_date]
        n_weeks    = maximum(local_W)

        meta_data  = MetaModelData(
            Int.(fold_df.is_winner),
            Float64.(fold_df.p_L1),
            Float64.(fold_df.prob_fair_close),
            local_W,
            fold_df.home_idx,
            fold_df.away_idx,
            n_weeks,
            n_teams
        )
        push!(fold_datasets, (meta_data, fold_df))
        println("    Fold $k: $(nrow(fold_df)) matches, $n_weeks weeks")
    end

    n_valid_folds = length(fold_datasets)

    # ------------------------------------------------------------------
    # 5. Queued NUTS: flatten (fold, chain) into a single task queue
    # ------------------------------------------------------------------
    cfg       = task.sampler_config
    n_chains  = cfg.n_chains
    is_queued = cfg isa Samplers.QueuedNUTSConfig

    total_tasks = n_valid_folds * n_chains
    println("[4] Queued NUTS: $n_valid_folds folds × $n_chains chains = $total_tasks tasks")

    # Pre-allocate storage
    fold_chain_store = [Vector{Any}(undef, n_chains) for _ in 1:n_valid_folds]
    chains_done      = zeros(Int, n_valid_folds)
    fold_results     = Vector{Union{Nothing, MetaFoldResult}}(fill(nothing, n_valid_folds))
    store_lock       = ReentrantLock()

    prog        = Progress(total_tasks, desc="Meta Sampling: ", showspeed=true)
    # max_concurrent_tasks lives on the Independent strategy (Layer 1), not QueuedNUTSConfig.
    # For the Meta Model we default to nthreads() — one chain per available thread.
    concurrency = min(total_tasks, Threads.nthreads())
    semaphore   = Base.Semaphore(concurrency)

    # Flattened task list: (fold_idx_1based, chain_id)
    tasks = [(f, c) for f in 1:n_valid_folds for c in 1:n_chains]

    @sync for (f, c_id) in tasks
        Threads.@spawn begin
            Base.acquire(semaphore)
            try
                meta_data, fold_df = fold_datasets[f]
                turing_model = build_turing_meta_model(task.meta_model, meta_data)

                # Run a single chain (mirrors QueuedNUTS approach in Layer 1)
                chain_result = if is_queued
                    all_inits = Samplers.get_init_params(turing_model, cfg.initialisation, n_chains)
                    sample(
                        turing_model,
                        NUTS(cfg.n_warmup, cfg.accept_rate, max_depth=cfg.max_depth),
                        cfg.n_samples;
                        progress  = false,
                        adtype    = AutoReverseDiff(compile=true),
                        initial_params = all_inits[c_id]
                    )
                else
                    sample(
                        turing_model,
                        NUTS(cfg.n_warmup, cfg.accept_rate, max_depth=cfg.max_depth),
                        cfg.n_samples;
                        progress = false
                    )
                end

                lock(store_lock) do
                    fold_chain_store[f][c_id] = chain_result
                    chains_done[f] += 1

                    # When all chains for this fold are done, combine and store
                    if chains_done[f] == n_chains
                        combined = chainscat(fold_chain_store[f]...)
                        _, fold_df_local = fold_datasets[f]
                        fold_results[f] = MetaFoldResult(combined, fold_df_local, f)
                        fold_chain_store[f] = []  # free memory
                        printstyled("\n  ✓ Fold $f complete ($(n_chains) chains merged)\n", color=:green)
                    end
                end

            catch e
                @error "Error in fold $f, chain $c_id" exception=(e, catch_backtrace())
            finally
                next!(prog)
                Base.release(semaphore)
            end
        end
    end

    valid_fold_results = filter(!isnothing, fold_results)

    println("\n[5] All folds complete. $(length(valid_fold_results)) / $n_valid_folds succeeded.")

    return MetaExperimentResults(
        task,
        MetaFoldResult[r for r in valid_fold_results],
        joined
    )
end

"""
    run_multi_market_experiments(
        target_selections::Vector{Symbol},
        exp_results::Experiments.ExperimentResults,
        meta_model::AbstractMetaModel,
        sampler_config::Samplers.AbstractNUTSConfig,
        ds::Data.DataStore;
        min_edge=0.00
    )

Runs the Meta Model pipeline sequentially across multiple target selections.
Returns two dictionaries:
- `multi_market_results`: Dict mapping selection to MetaExperimentResults
- `multi_market_ledgers`: Dict mapping selection to the OOS predictive ledger DataFrame
"""
function run_multi_market_experiments(
    target_selections::Vector{Symbol},
    exp_results::Experiments.ExperimentResults,
    meta_model::AbstractMetaModel,
    sampler_config::Samplers.AbstractNUTSConfig,
    ds::Data.DataStore;
    min_edge=0.00,
    staking_odds::Union{DataFrame, Nothing}=nothing
)
    multi_market_results = Dict{Symbol, Any}()
    multi_market_ledgers = Dict{Symbol, DataFrame}()

    for selection in target_selections
        println("\n" * "*"^65)
        println("  STARTING MARKET: $selection")
        println("*"^65)

        meta_task = MetaExperimentTask(
            exp_results,
            meta_model,
            sampler_config,
            exp_results.config.splitter,
            selection
        )

        println("\nRunning Queued Fold Experiment for $selection...")
        _raw_result = run_meta_experiment(meta_task; ds=ds)
        
        meta_results = if _raw_result isa Tuple
            _raw_result[1]
        else
            _raw_result
        end

        check_fold_rhats(meta_results)

        println("\nComputing Predictive Stakes for $selection...")
        ledger = compute_predictive_stakes(meta_results, meta_results.all_data; min_edge=min_edge, staking_odds=staking_odds)
        
        if nrow(ledger) == 0
            println("Warning: No OOS ledger data generated for $selection.")
            continue
        end

        multi_market_results[selection] = meta_results
        multi_market_ledgers[selection] = ledger
    end
    
    return multi_market_results, multi_market_ledgers
end

