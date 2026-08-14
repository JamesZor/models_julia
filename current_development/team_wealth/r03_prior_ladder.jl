# current_development/team_wealth/r03_prior_ladder.jl
#
# ==============================================================================
# RUNNER: Prior Ladder for Team Wealth Weight (w_wealth) Identification
# ==============================================================================
#
# PURPOSE:
#   Does the match and xG data carry genuine, identifiable information about the
#   latent team wealth parameter `w_wealth`?
#
# HYPOTHESIS & METHODOLOGY:
#   1. Under a tight prior (e.g. truncated Normal(0.20, 0.05)), the posterior
#      might simply mimic the prior if the likelihood carries no weight.
#   2. A 3-rung prior ladder separates prior-anchoring from genuine likelihood signal:
#        - Rung A: Tight Informative     w ~ truncated(Normal(0.20, 0.05), lower=0.0) [4 chains]
#        - Rung B: Wide Semi-Informative w ~ truncated(Normal(0.20, 0.30), lower=0.0) [6 chains]
#        - Rung C: Bounded-Flat          w ~ Uniform(0.0, 1.5)                        [6 chains]
#   3. If the parameter is identified by match outcomes and squad market valuations:
#        - The ratio (post_sd / prior_sd) will fall dramatically on wider/flat rungs (< 0.50).
#        - The posterior median will concentrate stably in [0.15, 0.28].
#        - The posterior mass P(w > 0.05) will be near 100%.
#
# EXECUTION (KAIMON / SERVER / MCMC-BEAST):
#   julia --project=. -t 16 current_development/team_wealth/r03_prior_ladder.jl
#   (Runs 16 chain-tasks concurrently in one wave across 16-32 threads)
# ==============================================================================

using BayesianFootball
using DataFrames, Dates, Distributions, Statistics, Printf, Serialization

include(joinpath(@__DIR__, "l01_wealth_data.jl"))
include(joinpath(@__DIR__, "l02_wealth_engine.jl"))
include(joinpath(@__DIR__, "l03_wealth_predict.jl"))

using Turing: MCMCChains

const Experiments = BayesianFootball.Experiments
const Samplers    = BayesianFootball.Samplers
const Training    = BayesianFootball.Training

# Configuration constants
const TW_OUT_DIR   = "./data/l2_ireland_engines"
const TW_LOG_PATH  = joinpath(@__DIR__, "r03_prior_ladder_out.txt")
const TW_PIN_PATH  = joinpath(TW_OUT_DIR, "ds_ire79.jls")

const TW_STEP      = 12        # 2026 biweek 12 — largest fold with mature rating histories
const TW_WARMUP    = 800       # Warmup draws per chain
const TW_SAMPLES   = 500       # Post-warmup samples per chain
const TW_MAX_DEPTH = 10
const TW_ACCEPT    = 0.65

println("="^90)
println("TEAM WEALTH (w_wealth) PRIOR LADDER RUNNER")
println("Tournament 79: League of Ireland Premier Division (2026 Fold $TW_STEP)")
println("="^90)

# ==============================================================================
# SECTION 1: DEFINE THE 3-RUNG LADDER
# ==============================================================================

tw_rungs = [
    (name = "A tight N(0.20, 0.05)",
     prior = truncated(Normal(0.20, 0.05), lower=0.0),
     chains = 4),
    (name = "B wide  N(0.20, 0.30)",
     prior = truncated(Normal(0.20, 0.30), lower=0.0),
     chains = 6),
    (name = "C flat  U(0.0, 1.50)",
     prior = Uniform(0.0, 1.50),
     chains = 6),
]

wealth_model_ladder(w_prior) = DynamicSmileDoublePoissonXGWealthPlayerTimeDecayModel(
    interception_config    = PreGame.HierarchicalMonthlyInterception(),
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0),
    homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
    kappa_config           = PreGame.HierarchicalTeamKappa(),
    player_ratings_feature = Features.PlayerRatingsFeature(
                                 Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)),
    wealth_feature         = TeamWealthFeature(),
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    smile_feature          = Features.MarketSmileFeature(Kmax = 4),
    w_wealth_prior         = w_prior,
    market_on              = true,
    supremacy_weight       = 0.4,
    smile_weight           = 0.4,
)

# ==============================================================================
# SECTION 2: LOAD PINNED DATASTORE & BUILD FEATURES ONCE
# ==============================================================================

println("\n1. Loading pinned dataset at $TW_PIN_PATH...")
if isfile(TW_PIN_PATH)
    tw_ds = deserialize(TW_PIN_PATH)
    println("   Successfully loaded pinned DataStore ($(nrow(tw_ds.matches)) matches).")
else
    @warn "Pinned DataStore not found at $TW_PIN_PATH! Falling back to cached DataStore."
    tw_ds = Data.load_datastore_cached(Data.IrelandPremier())
end

tw_splitter = Data.GroupedCVConfig(
    tournament_groups = [Data.tournament_ids(tw_ds.segment)],
    target_seasons    = ["2026"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = TW_STEP,
    end_dynamics      = TW_STEP,
    stop_early        = false,
)

println("2. Extracting fold boundaries and features...")
tw_bounds = Data.create_id_boundaries(tw_ds, tw_splitter)
println("   Available fold boundaries: ", [(string(md.target_season), md.time_step) for (_, md) in tw_bounds])

# Build features once using the base ladder model
tw_all_fs = Features.create_features(
    tw_bounds, 
    tw_ds, 
    wealth_model_ladder(tw_rungs[1].prior),
    tw_splitter.dynamics_col
)

tw_idx = findfirst(i -> tw_all_fs[i][2].time_step == TW_STEP, eachindex(tw_all_fs))
isnothing(tw_idx) && error("Fold at time_step $TW_STEP not found in feature set!")
tw_fs = tw_all_fs[tw_idx][1]

# Goal-pillar effective N calculation
tw_eff_N = sum(0.5 .^ (Vector{Int}(tw_fs.data[:dates]) ./ 60.0))
n_hist_matches = length(tw_fs.data[:dates])
@printf("   Fold biweek %d: %d history matches, Goals Effective N = %.1f (%d goal observations)\n",
        TW_STEP, n_hist_matches, tw_eff_N, round(Int, 2 * tw_eff_N))

# Check that flat_wealth_diff is present in feature set
if haskey(tw_fs.data, :flat_wealth_diff)
    w_diffs = tw_fs.data[:flat_wealth_diff]
    @printf("   Team Wealth differences loaded: min = %+.3f, max = %+.3f, SD = %.3f\n",
            minimum(w_diffs), maximum(w_diffs), std(w_diffs))
else
    @warn "flat_wealth_diff is missing from tw_fs.data! Check l01_wealth_data.jl integration."
end

# ==============================================================================
# SECTION 3: ASYNCHRONOUS MULTI-THREADED MCMC EXECUTION
# ==============================================================================

tw_training_cfg(nc) = Training.TrainingConfig(
    Samplers.QueuedNUTSConfig(
        n_samples      = TW_SAMPLES,
        n_chains       = nc,
        n_warmup       = TW_WARMUP,
        accept_rate    = TW_ACCEPT,
        max_depth      = TW_MAX_DEPTH,
        initialisation = Samplers.UniformInit(-2.0, 2.0),
        show_progress  = false
    ),
    Training.Independent(parallel = true, max_concurrent_tasks = Threads.nthreads()),
    nothing,
    false
)

tw_tasks = [(ri, c) for ri in eachindex(tw_rungs) for c in 1:tw_rungs[ri].chains]
tw_slots = Dict(ri => Vector{Any}(undef, tw_rungs[ri].chains) for ri in eachindex(tw_rungs))

println("\n", "="^90)
@printf("LAUNCHING MCMC SAMPLING: %d rungs, %d chain-tasks, %d threads\n",
        length(tw_rungs), length(tw_tasks), Threads.nthreads())
for r in tw_rungs
    @printf("  %-25s : Chains = %d | Prior SD(w) = %.4f | Prior Mean = %.4f\n", 
            r.name, r.chains, std(r.prior), mean(r.prior))
end
println("="^90)

tw_semaphore = Base.Semaphore(min(length(tw_tasks), Threads.nthreads()))
tw_lock      = ReentrantLock()
tw_t0        = time()

@sync for (ri, c) in tw_tasks
    Threads.@spawn begin
        Base.acquire(tw_semaphore)
        try
            @printf("  -> [START] Rung %d (%s) Chain %d on Thread %d...\n", 
                    ri, tw_rungs[ri].name, c, Threads.threadid())
            ch = Training.train(
                wealth_model_ladder(tw_rungs[ri].prior),
                tw_training_cfg(tw_rungs[ri].chains),
                tw_fs;
                chain_id = c
            )
            lock(tw_lock) do
                tw_slots[ri][c] = ch
            end
            @printf("  <- [DONE]  Rung %d (%s) Chain %d finished.\n", ri, tw_rungs[ri].name, c)
        catch e
            @error "Rung $ri Chain $c failed" exception = (e, catch_backtrace())
            lock(tw_lock) do
                tw_slots[ri][c] = nothing
            end
        finally
            Base.release(tw_semaphore)
        end
    end
end

elapsed_min = (time() - tw_t0) / 60.0
@printf("\n✓ All chains completed in %.2f minutes.\n", elapsed_min)

# ==============================================================================
# SECTION 4: DIAGNOSTICS & SUMMARY CALCULATIONS
# ==============================================================================

println("\n3. Processing MCMC chains and calculating ladder diagnostics...")

tw_rows = NamedTuple[]

for ri in eachindex(tw_rungs)
    rung = tw_rungs[ri]
    good = [c for c in tw_slots[ri] if !isnothing(c)]
    if isempty(good)
        @warn "Rung $(rung.name): all chains failed!"
        continue
    end
    
    ch = cat(good...; dims = 3)
    
    # MCMC summary table
    summ = DataFrame(MCMCChains.summarize(ch))
    getf(r, syms) = (for s in syms; hasproperty(r, s) && return Float64(getproperty(r, s)); end; NaN)
    sampler_names = ("lp","n_steps","acceptance_rate","tree_depth","numerical_error","step_size",
                     "nom_step_size","is_accept","hamiltonian_energy","hamiltonian_energy_error",
                     "max_hamiltonian_energy_error")
    par = filter(r -> !(string(r.parameters) in sampler_names) && !isnan(getf(r, (:rhat,))), summ)

    grab(sym) = begin
        row = findfirst(r -> string(r.parameters) == sym, eachrow(par))
        isnothing(row) ? (rhat = NaN, ess = NaN) :
        (rhat = getf(par[row, :], (:rhat,)), ess = getf(par[row, :], (:ess, :ess_bulk)))
    end

    w_samples = vec(Array(ch[:w_wealth]))
    prior_sd  = std(rung.prior)
    post_sd   = std(w_samples)
    ratio     = post_sd / prior_sd
    
    w_med  = median(w_samples)
    w_mean = mean(w_samples)
    w_lo   = quantile(w_samples, 0.05)
    w_hi   = quantile(w_samples, 0.95)
    
    p_pos  = mean(w_samples .> 0.0)
    p_sub  = mean(w_samples .> 0.10)
    
    div_pct = :numerical_error in Symbol.(names(ch)) ?
                  100 * mean(vec(Array(ch[:numerical_error]))) : 0.0

    push!(tw_rows, (
        rung        = rung.name,
        n_chains    = length(good),
        prior_sd    = prior_sd,
        post_sd     = post_sd,
        ratio       = ratio,
        w_med       = w_med,
        w_mean      = w_mean,
        w_lo        = w_lo,
        w_hi        = w_hi,
        p_pos       = p_pos,
        p_sub       = p_sub,
        max_rhat    = maximum(getf.(eachrow(par), Ref((:rhat,)))),
        rhat_w      = grab("w_wealth").rhat,
        ess_w       = grab("w_wealth").ess,
        div_pct     = div_pct,
        multiplier  = exp(w_med)  # Goal rate multiplier per +1 SD wealth disparity
    ))
end

tw_out = DataFrame(tw_rows)

# ==============================================================================
# SECTION 5: DISPLAY RESULTS & VERDICT
# ==============================================================================

println("\n", "="^95)
println("PRIOR LADDER SUMMARY: w_wealth PARAMETER IDENTIFICATION")
println("="^95)
show(stdout, MIME"text/plain"(), 
     select(tw_out, :rung, :n_chains, :prior_sd, :post_sd, :ratio, :w_med, :w_lo, :w_hi, :multiplier))
println()

println("\n--- CONVERGENCE & NUMERICAL HEALTH ---")
show(stdout, MIME"text/plain"(), 
     select(tw_out, :rung, :max_rhat, :rhat_w, :ess_w, :div_pct, :p_pos, :p_sub))
println()

# Extract key ladder rungs
tw_pick(pre) = (i = findfirst(startswith(pre), tw_out.rung); isnothing(i) ? nothing : tw_out[i, :])
rung_tight = something(tw_pick("A"), tw_out[1, :])
rung_wide  = something(tw_pick("B"), tw_out[2, :])
rung_flat  = something(tw_pick("C"), tw_out[end, :])

println("\n", "="^95)
println("STATISTICAL VERDICT & INTERPRETATION")
println("="^95)
@printf("Goals-Pillar Effective N = %.1f matches (Fold %d, %d total matches)\n", 
        tw_eff_N, TW_STEP, n_hist_matches)
@printf("Identification Ratio (Post SD / Prior SD): Tight = %.2f -> Wide = %.2f -> Flat = %.2f\n",
        rung_tight.ratio, rung_wide.ratio, rung_flat.ratio)
@printf("Posterior w_wealth Median under Flat Prior : %.4f [90%% CI: %.4f, %.4f]\n",
        rung_flat.w_med, rung_flat.w_lo, rung_flat.w_hi)
@printf("Goal Rate Ratio per +1σ Squad Wealth Disparity: %.2fx (Median)\n", rung_flat.multiplier)

if rung_flat.ratio < 0.50 && rung_flat.w_lo > 0.02
    println("""
VERDICT: [STRONG IDENTIFICATION]
  The data unequivocally informs and identifies the Team Wealth weight (w_wealth).
  Under the non-informative flat prior U(0, 1.5), the posterior variance collapses by $(round((1.0 - rung_flat.ratio)*100, digits=1))%
  (ratio = $(round(rung_flat.ratio, digits=3))) and concentrates firmly around w ≈ $(round(rung_flat.w_med, digits=3)).
  The probability P(w_wealth > 0) is $(round(100*rung_flat.p_pos, digits=1))%, and P(w_wealth > 0.10) is $(round(100*rung_flat.p_sub, digits=1))%.
  -> CONCLUSION: Team Wealth is a genuine likelihood signal, not a prior artifact.
  -> PROCEED TO: Route 2 Out-Of-Sample Benchmark (r04_route2_judge.jl).""")
elseif rung_flat.ratio > 0.85
    println("""
VERDICT: [UNIDENTIFIED / PRIOR-DOMINATED]
  The posterior heavily tracks the prior (ratio = $(round(rung_flat.ratio, digits=3))).
  The match and xG data alone does not carry enough resolution to identify w_wealth.
  -> Check rating half-life or investigate feature scaling.""")
else
    println("""
VERDICT: [MODERATE IDENTIFICATION]
  The data provides moderate constraint on w_wealth (ratio = $(round(rung_flat.ratio, digits=3))).
  Posterior median sits at $(round(rung_flat.w_med, digits=3)) [CI: $(round(rung_flat.w_lo, digits=3)), $(round(rung_flat.w_hi, digits=3))].""")
end
println("="^95)

# Save output to text artifact
open(TW_LOG_PATH, "w") do io
    println(io, "TEAM WEALTH PRIOR LADDER RUNNER OUTPUT")
    println(io, "Date: ", now())
    println(io, "Warmup: $TW_WARMUP, Samples: $TW_SAMPLES, Fold: Biweek $TW_STEP, Chains: $(length(tw_tasks))\n")
    show(io, MIME"text/plain"(), tw_out)
    println(io, "\n\nVerdict:\n")
    @printf(io, "Flat Prior Ratio: %.4f | Posterior Median: %.4f [CI: %.4f, %.4f]\n", 
            rung_flat.ratio, rung_flat.w_med, rung_flat.w_lo, rung_flat.w_hi)
end
println("\nDetailed diagnostic report written to $TW_LOG_PATH")

nothing
