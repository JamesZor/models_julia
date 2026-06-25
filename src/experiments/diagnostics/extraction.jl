# src/experiments/diagnostics/extraction.jl

"""
    extract_chains(ds::Data.DataStore, exp_results::Experiments.ExperimentResults) -> ExperimentChains

Extracts MCMC parameter summaries (mean, std, rhat, ess) across all folds in an experiment.
Automatically maps team and positional indices to their string equivalents.
"""
function extract_chains(ds::Data.DataStore, exp_results::Experiments.ExperimentResults)
    config = exp_results.config
    
    # 1. Reconstruct boundaries and metadata to match the training logic
    boundaries_with_meta = Data.create_id_boundaries(ds, config.splitter)
    
    # 2. Create feature sets to get the correct team_map per fold
    feature_sets = Features.create_features(
        boundaries_with_meta, 
        ds, 
        config.model, 
        config.splitter.dynamics_col
    )
    
    results_array = exp_results.training_results.items
    n_splits = length(results_array)

    all_fold_dfs = Vector{DataFrame}()
    println("[INFO] Extracting chain diagnostics for $n_splits splits...")

    for i in 1:n_splits
        chain, meta = results_array[i]
        feature_tuple = feature_sets[i]
        
        # Process this fold
        fold_df = _process_parameter_fold(config.model, feature_tuple, chain, meta)
        push!(all_fold_dfs, fold_df)
    end

    return ExperimentChains(vcat(all_fold_dfs...))
end

function _process_parameter_fold(model, feature_tuple, chain, meta)
    feature_set = feature_tuple[1]
    data = feature_set.data
    
    n_teams   = Int(get(data, :n_teams, 0))
    n_seasons = Int(get(data, :n_seasons, 0))
    n_months  = 12
    team_map  = get(data, :team_map, Dict())
    rev_team_map = Dict(v => k for (k, v) in team_map)
    
    # Pre-calculate MCMC summaries (rhat, ess) for raw parameters
    # This might fail if chain is empty or doesn't have multiple chains, so wrap in try-catch
    chain_summary = Dict{Symbol, NamedTuple}()
    
    if size(chain, 1) == 1
        @info "Point-mass chain (Optimization) detected. Skipping MCMC diagnostics (rhat, ess)."
    else
        try
            summ_df = DataFrame(MCMCChains.summarize(chain))
            for row in eachrow(summ_df)
                p = Symbol(row.parameters)
                chain_summary[p] = (
                    rhat = hasproperty(row, :rhat) ? row.rhat : NaN,
                    ess  = hasproperty(row, :ess) ? row.ess : NaN
                )
            end
        catch
            # Fallback if summarize fails
        end
    end

    fold_info = Dict(
        :fold => meta.time_step,
        :target_season => meta.target_season,
        :week => meta.time_step,
        :train_season => meta.train_season
    )
    
    rows = []

    # Helper to push a parameter
    function _add_param!(rows, param_name::AbstractString, entity::AbstractString, raw_symbol::Symbol, samples::AbstractVector)
        rhat = NaN
        ess = NaN
        if haskey(chain_summary, raw_symbol)
            rhat = chain_summary[raw_symbol].rhat
            ess  = chain_summary[raw_symbol].ess
        end
        push!(rows, merge(fold_info, Dict(
            :parameter => param_name,
            :entity => entity,
            :raw_symbol => raw_symbol,
            :mean => mean(samples),
            :std => std(samples),
            :rhat => rhat,
            :ess => ess
        )))
    end

    # --- 1. Global Scalar Parameters ---
    scalar_candidates = [:ν_xg, :σ_market, :market_σ, :σ_sup, :σ_lev]
    for p in scalar_candidates
        if p in keys(chain)
            samples = vec(Array(chain[p]))
            _add_param!(rows, string(p), "global", p, samples)
        end
    end

    # --- 2. Home Advantage (Hierarchical) ---
    if hasproperty(model, :homeadvantage_config)
        # Try to extract the centered HA first
        try
            ha_mat = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
            for t_idx in 1:n_teams
                team_name = get(rev_team_map, t_idx, "unknown")
                samples = ha_mat[:, t_idx]
                raw_sym = Symbol("ha.γ_team_raw[$t_idx]") # Best guess for rhat tracking
                _add_param!(rows, "home_advantage", team_name, raw_sym, samples)
            end
        catch e
            @warn "Failed to extract home advantage" exception=(e, catch_backtrace())
        end
        
        for p in [Symbol("ha.γ_base"), Symbol("ha.σ_γ"), Symbol("ha.γ_global")]
            if p in keys(chain)
                samples = vec(Array(chain[p]))
                _add_param!(rows, string(p), "global", p, samples)
            end
        end
    end

    # --- 3. Kappa (Conversion Rate) ---
    if hasproperty(model, :kappa_config)
        try
            kap_mat = PreGame.extract_kappa(chain, model.kappa_config, n_teams)
            for t_idx in 1:n_teams
                team_name = get(rev_team_map, t_idx, "unknown")
                samples = kap_mat[:, t_idx]
                raw_sym = Symbol("kap.κ_team_raw[$t_idx]")
                _add_param!(rows, "kappa", team_name, raw_sym, samples)
            end
        catch e
            @warn "Failed to extract kappa" exception=(e, catch_backtrace())
        end

        for p in [Symbol("kap.κ_base"), Symbol("kap.σ_κ"), Symbol("kap.κ_global")]
            if p in keys(chain)
                samples = vec(Array(chain[p]))
                _add_param!(rows, string(p), "global", p, samples)
            end
        end
    end

    # --- 4. Interception ---
    if hasproperty(model, :interception_config)
        try
            inter_mat = PreGame.extract_interception(chain, model.interception_config, max(1, n_seasons))
            for s_idx in 1:size(inter_mat, 2)
                samples = inter_mat[:, s_idx]
                raw_sym = Symbol("int.μ[$s_idx]")
                _add_param!(rows, "interception", "season_$s_idx", raw_sym, samples)
            end
        catch
            if Symbol("int.μ") in keys(chain)
                samples = vec(Array(chain[Symbol("int.μ")]))
                _add_param!(rows, "interception", "global", Symbol("int.μ"), samples)
            end
        end
    end

    # --- 5. Positional Dynamics (Weights) ---
    if hasproperty(model, :player_dynamics_config)
        # For positional parameters, extract from raw chain keys directly
        for p in keys(chain)
            ps = string(p)
            if startswith(ps, "p_dyn.")
                samples = vec(Array(chain[p]))
                _add_param!(rows, ps, "global", p, samples)
            end
        end
    end

    # --- 6. Dispersion (r) ---
    if hasproperty(model, :dispersion_config)
        try
            disp_nt = PreGame.extract_dispersion(chain, model.dispersion_config, n_teams, n_months)
            
            if hasproperty(disp_nt, :h)
                _add_param!(rows, "r_home", "global", Symbol("disp.log_r"), disp_nt.h)
                _add_param!(rows, "r_away", "global", Symbol("disp.log_r"), disp_nt.a)
            end

            if hasproperty(disp_nt, :team_vol)
                for t_idx in 1:n_teams
                    team_name = get(rev_team_map, t_idx, "unknown")
                    samples = disp_nt.team_vol[:, t_idx]
                    raw_sym = Symbol("disp.team_vol_raw[$t_idx]")
                    _add_param!(rows, "dispersion_vol", team_name, raw_sym, samples)
                end
                
                for p in [:base, :home_offset]
                    if hasproperty(disp_nt, p)
                        samples = disp_nt[p]
                        raw_sym = Symbol("disp." * string(p))
                        _add_param!(rows, "dispersion_" * string(p), "global", raw_sym, samples)
                    end
                end
            end
        catch e
            @warn "Failed to extract dispersion" exception=(e, catch_backtrace())
        end
    end
    # --- 7. Copula ---
    if hasproperty(model, :copula_config)
        try
            cop_nt = PreGame.extract_copula(chain, model.copula_config, "copula", n_teams)
            
            if hasproperty(cop_nt, :κ_base)
                _add_param!(rows, "copula_kappa_base", "global", Symbol("copula.κ_base"), cop_nt.κ_base)
            end
            if hasproperty(cop_nt, :σ_κ)
                _add_param!(rows, "copula_sigma", "global", Symbol("copula.σ_κ"), cop_nt.σ_κ)
            end
            if hasproperty(cop_nt, :κ) # For the non-hierarchical version
                _add_param!(rows, "copula_kappa", "global", Symbol("copula.κ"), cop_nt.κ)
            end
            
            if hasproperty(cop_nt, :δ_κ)
                for t_idx in 1:n_teams
                    team_name = get(rev_team_map, t_idx, "unknown")
                    samples = cop_nt.δ_κ[:, t_idx]
                    raw_sym = Symbol("copula.raw_κ[$t_idx]")
                    _add_param!(rows, "copula_delta", team_name, raw_sym, samples)
                end
            end
        catch e
            @warn "Failed to extract copula" exception=(e, catch_backtrace())
        end
    end

    # --- 8. Dynamics (Team Level) ---
    if hasproperty(model, :dynamics_config)
        try
            dyn_nt = PreGame.extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)
            
            if hasproperty(dyn_nt, :α)
                for t_idx in 1:n_teams
                    team_name = get(rev_team_map, t_idx, "unknown")
                    samples_a = dyn_nt.α[:, t_idx]
                    raw_sym_a = Symbol("dyn.raw_a[$t_idx]")
                    _add_param!(rows, "attack_dynamics", team_name, raw_sym_a, samples_a)
                    
                    samples_d = dyn_nt.β[:, t_idx]
                    raw_sym_d = Symbol("dyn.raw_d[$t_idx]")
                    _add_param!(rows, "defense_dynamics", team_name, raw_sym_d, samples_d)
                end
            end
            
            for p in [:σ_a, :σ_d]
                sym = Symbol("dyn." * string(p))
                if sym in keys(chain)
                    samples = vec(Array(chain[sym]))
                    _add_param!(rows, "dynamics_" * string(p), "global", sym, samples)
                end
            end
        catch e
            @warn "Failed to extract team dynamics" exception=(e, catch_backtrace())
        end
    end

    # Catch any remaining parameters we might care about like lp (log density)
    if :lp in keys(chain)
        _add_param!(rows, "log_density", "global", :lp, vec(Array(chain[:lp])))
    end

    return DataFrame(rows)
end
