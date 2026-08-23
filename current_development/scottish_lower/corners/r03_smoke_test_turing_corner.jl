# current_development/scottish_lower/corners/r03_smoke_test_turing_corner.jl
#
# Turing Smoke Test: Baseline 4-Way Corner Recombination Model
# Tests NUTS MCMC Sampling, Parameter R-hats, and Out-of-Sample ScoreMatrix Inference

using ThreadPinning
pinthreads(:cores)

include("l01_corner_data.jl")
include("l04_turing_corner_model.jl")

using Printf
using Dates
using Statistics
using MCMCChains
using ReverseDiff

# Enable high-speed ReverseDiff compiled tape
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

println("================================================================================")
println(" TURING MCMC SMOKE TEST: 4-WAY CORNER RECOMBINATION BASELINE")
println("================================================================================")

# 1. Ingest Data & Filter to Scottish Lower
df_all = fetch_scottish_corner_dataset()
df_lower = filter(r -> r.tournament_id in (56, 57), df_all)
sort!(df_lower, :match_datetime)

println("Total Scottish Lower Matches: ", nrow(df_lower))

# 2. Train / Test Walk-Forward Split (Split at 2024-08-01)
split_date = Date(2024, 8, 1)
df_train = filter(r -> Date(r.match_date) < split_date, df_lower)
df_test = filter(r -> Date(r.match_date) >= split_date && Date(r.match_date) < Date(2025, 1, 1), df_lower)

println("Training Set: ", nrow(df_train), " matches (", Dates.format(minimum(df_train.match_date), "yyyy-mm-dd"), " to ", Dates.format(maximum(df_train.match_date), "yyyy-mm-dd"), ")")
println("Test (OOS) Set: ", nrow(df_test), " matches (", Dates.format(minimum(df_test.match_date), "yyyy-mm-dd"), " to ", Dates.format(maximum(df_test.match_date), "yyyy-mm-dd"), ")\n")

# 3. Team & Feature Indexing
all_teams = sort(unique(vcat(df_train.home_team, df_train.away_team)))
team_to_idx = Dict(t => i for (i, t) in enumerate(all_teams))
n_teams = length(all_teams)

# Filter test set to teams present in training dictionary
filter!(r -> haskey(team_to_idx, r.home_team) && haskey(team_to_idx, r.away_team), df_test)

h_idx_train = [team_to_idx[t] for t in df_train.home_team]
a_idx_train = [team_to_idx[t] for t in df_train.away_team]

# Month & League indices
month_idx_train = [Dates.month(r.match_date) for r in eachrow(df_train)]
league_map = Dict(56 => 1, 57 => 2)
league_idx_train = [league_map[t] for t in df_train.tournament_id]
n_leagues = 2

# Exponential Time Decay (half-life = 365 days)
max_train_date = maximum(df_train.match_datetime)
decay_rate = log(2.0) / 365.0
match_weights = [exp(-decay_rate * max(0.0, (max_train_date - r.match_datetime).value / (1000 * 3600 * 24))) for r in eachrow(df_train)]

# 4. Instantiate Model & Run MCMC
config = DynamicCornerRecombModel()
turing_mod = build_corner_recomb_engine(
    h_idx_train,
    a_idx_train,
    month_idx_train,
    league_idx_train,
    Int.(df_train.open_goals_h),
    Int.(df_train.open_goals_a),
    Int.(df_train.corners_h),
    Int.(df_train.corners_a),
    Int.(df_train.corner_goals_h),
    Int.(df_train.corner_goals_a),
    Float64.(df_train.corners_h .> 0),
    Float64.(df_train.corners_a .> 0),
    match_weights,
    n_teams,
    n_leagues,
    config
)

println("--- Starting NUTS MCMC Sampling (500 warmup, 500 samples, 4 chains) ---")
t_start = time()
chain = sample(turing_mod, NUTS(500, 0.65), MCMCThreads(), 500, 4)
t_elapsed = time() - t_start
@printf("✓ Sampling complete in %.2f seconds (%.2f s/chain)\n\n", t_elapsed, t_elapsed / 4)

# 5. Diagnostic Assessment (R-hat & ESS)
println("--- MCMC CONVERGENCE DIAGNOSTICS ---")
chn_summary = describe(chain)[1]

# Extract core scalar parameters
core_params = ["μ_c_base", "γ_ha_c", "ϕ_c_inv", "σ_conv_att", "σ_conv_def"]
println("Core Latent Parameters:")
for p in core_params
    row_idx = findfirst(==(Symbol(p)), chn_summary[:, :parameters])
    if row_idx !== nothing
        m = chn_summary[row_idx, :mean]
        s = chn_summary[row_idx, :std]
        rhat = chn_summary[row_idx, :rhat]
        ess = chn_summary[row_idx, :ess]
        @printf("  %-15s | Mean: %+.3f | Std: %.3f | R-hat: %.4f | ESS: %6.1f\n", p, m, s, rhat, ess)
    end
end

# Check overall max R-hat
rhats = filter(isfinite, chn_summary[:, :rhat])
max_rhat = isempty(rhats) ? 1.0 : maximum(rhats)
@printf("\nGlobal Max R-hat across all %d parameters: %.4f\n", length(chn_summary[:, :parameters]), max_rhat)

if max_rhat <= 1.05
    println(">>> VERDICT: All parameters CONVERGED with R-hat <= 1.05! <<<")
else
    @warn "Some parameters have R-hat > 1.05"
end

# 6. Out-Of-Sample (OOS) ScoreMatrix Inference on Test Set
println("\n--- OUT-OF-SAMPLE (OOS) PREDICTION EVALUATION (N = $(nrow(df_test)) matches) ---")

# Extract posterior mean parameters
chain_df = DataFrame(chain)
mu_open_base = mean(chain_df[:, "inter.μ_base[1]"])
gamma_ha_open = mean(chain_df[:, "ha[1]"]) # approximate home adv
mu_c_base = mean(chain_df[:, "μ_c_base"])
gamma_ha_c = mean(chain_df[:, "γ_ha_c"])
sigma_conv_att = mean(chain_df[:, "σ_conv_att"])
sigma_conv_def = mean(chain_df[:, "σ_conv_def"])

alpha_open_means = [mean(chain_df[:, "dyn.α[$i]"]) for i in 1:n_teams]
beta_open_means = [mean(chain_df[:, "dyn.β[$i]"]) for i in 1:n_teams]
alpha_c_means = [mean(chain_df[:, "α_c_raw[$i]"]) for i in 1:n_teams]
beta_c_means = [mean(chain_df[:, "β_c_raw[$i]"]) for i in 1:n_teams]
alpha_c_means .-= mean(alpha_c_means)
beta_c_means .-= mean(beta_c_means)

z_att_means = [mean(chain_df[:, "z_att_raw[$i]"]) for i in 1:n_teams]
z_def_means = [mean(chain_df[:, "z_def_raw[$i]"]) for i in 1:n_teams]
z_att_means .-= mean(z_att_means)
z_def_means .-= mean(z_def_means)

sample_matches = first(df_test, 5)
println("Sample OOS Fixtures Predictions:")
for r in eachrow(sample_matches)
    th = team_to_idx[r.home_team]
    ta = team_to_idx[r.away_team]

    # Open Play
    μ_op_h = exp(mu_open_base + gamma_ha_open + alpha_open_means[th] - beta_open_means[ta])
    μ_op_a = exp(mu_open_base +                 alpha_open_means[ta] - beta_open_means[th])

    # Corners
    λ_c_h = exp(mu_c_base + gamma_ha_c + alpha_c_means[th] - beta_c_means[ta])
    λ_c_a = exp(mu_c_base +              alpha_c_means[ta] - beta_c_means[th])

    # Conversion
    logit_q_h = -3.23 + sigma_conv_att * z_att_means[th] - sigma_conv_def * z_def_means[ta]
    logit_q_a = -3.23 + sigma_conv_att * z_att_means[ta] - sigma_conv_def * z_def_means[th]
    q_c_h = logistic(logit_q_h)
    q_c_a = logistic(logit_q_a)

    # Total Lambda
    μ_tot_h = μ_op_h + 0.78 * 0.219 + 0.063 + q_c_h * λ_c_h
    μ_tot_a = μ_op_a + 0.78 * 0.219 + 0.063 + q_c_a * λ_c_a

    # Compute 4-way Score Matrix
    S = compute_4way_score_matrix(μ_op_h, μ_op_a, λ_c_h, λ_c_a, q_c_h, q_c_a)
    sum_prob = sum(S)

    # 1X2 Probabilities
    p_home = sum(S[i, j] for i in 1:size(S,1), j in 1:size(S,2) if i > j)
    p_draw = sum(S[i, i] for i in 1:min(size(S,1), size(S,2)))
    p_away = sum(S[i, j] for i in 1:size(S,1), j in 1:size(S,2) if i < j)

    @printf("\n[%s vs %s] (Actual: %d-%d)\n", r.home_team, r.away_team, r.goals_total_h, r.goals_total_a)
    @printf("  Open-Play μ: (H: %.2f, A: %.2f) | Corners λ: (H: %.2f, A: %.2f) | Conv q: (H: %.2f%%, A: %.2f%%)\n",
            μ_op_h, μ_op_a, λ_c_h, λ_c_a, q_c_h * 100, q_c_a * 100)
    @printf("  Total Goal μ: (H: %.2f, A: %.2f) | Prob Sum Invariant: %.6f\n", μ_tot_h, μ_tot_a, sum_prob)
    @printf("  1X2 Model Probs: Home: %5.2f%% | Draw: %5.2f%% | Away: %5.2f%%\n",
            p_home * 100, p_draw * 100, p_away * 100)
end

println("\n================================================================================")
println("✓ TURING MCMC SMOKE TEST COMPLETE")
println("================================================================================")
