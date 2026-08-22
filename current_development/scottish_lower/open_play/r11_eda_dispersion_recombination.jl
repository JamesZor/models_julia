# ==============================================================================
# r11_eda_dispersion_recombination.jl
#
# Exploratory Data Analysis (EDA):
# 1. Empirical Dispersion (Mean, Variance, Fano Factor = Var/Mean) of:
#    - Gross Goals (Home & Away)
#    - Open-Play Goals (Home & Away)
#    - Penalty Goals (Home & Away)
# 2. Posterior Dispersion Parameters (r_home, r_away) from MCMC Chains:
#    - goals_negbin_ctl (Gross Goals NegBin)
#    - goals_negbin_open_play (Open-Play NegBin)
#    - recomb_negbin_integrated (Recombination NegBin)
# 3. Mathematical Overdispersion Analysis (Var/Mean = 1 + μ/r)
# ==============================================================================

using BayesianFootball
using DataFrames
using Statistics
using Printf
using MCMCChains

# Include loader for open play extraction
include("l01_open_play_feature.jl")

println("="^90)
println("🔬 EDA: DISPERSION & OVERDISPERSION ANALYSIS (GROSS GOALS vs OPEN-PLAY)")
println("="^90)

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower())
df = extract_open_play_match_data(ds)

n_total = nrow(df)
println("\n✓ Total Matches in Dataset: $n_total")

# Add open-play goals and penalty goals
df.home_gross_goals = Int.(df.home_score)
df.away_gross_goals = Int.(df.away_score)

df.home_pen_goals = Int.(df.pen_scored_h)
df.away_pen_goals = Int.(df.pen_scored_a)

df.home_open_goals = Int.(df.y_np_nog_h)
df.away_open_goals = Int.(df.y_np_nog_a)

# ==============================================================================
# SECTION 1: EMPIRICAL DISPERSION METRICS (FULL DATASET & RECENT SEASONS)
# ==============================================================================

function compute_dispersion_metrics(v::Vector{Int}, name::String)
    m = mean(v)
    v_var = var(v)
    fano = v_var / m  # Fano Factor: Var / Mean (Poisson = 1.0)
    std_err_fano = fano * sqrt(2.0 / (length(v) - 1))
    return (
        name = name,
        n = length(v),
        mean = round(m, digits=4),
        var = round(v_var, digits=4),
        std = round(sqrt(v_var), digits=4),
        fano = round(fano, digits=4),
        overdisp_pct = round((fano - 1.0) * 100, digits=2),
        fano_ci_low = round(fano - 1.96 * std_err_fano, digits=4),
        fano_ci_high = round(fano + 1.96 * std_err_fano, digits=4)
    )
end

println("\n" * "="^90)
println("📊 1. EMPIRICAL DISPERSION METRICS (ALL 1,990 MATCHES)")
println("="^90)

metrics_all = [
    compute_dispersion_metrics(df.home_gross_goals, "Gross Home Goals"),
    compute_dispersion_metrics(df.away_gross_goals, "Gross Away Goals"),
    compute_dispersion_metrics(df.home_open_goals, "Open-Play Home Goals"),
    compute_dispersion_metrics(df.away_open_goals, "Open-Play Away Goals"),
    compute_dispersion_metrics(df.home_pen_goals, "Penalty Home Goals"),
    compute_dispersion_metrics(df.away_pen_goals, "Penalty Away Goals"),
    compute_dispersion_metrics(df.home_gross_goals .+ df.away_gross_goals, "Total Gross Goals"),
    compute_dispersion_metrics(df.home_open_goals .+ df.away_open_goals, "Total Open-Play Goals")
]

disp_df_all = DataFrame(metrics_all)
println(disp_df_all)

# Target Test Matches (Seasons 24/25 & 25/26)
df_recent = filter(r -> hasproperty(r, :season) && (r.season == "24/25" || r.season == "25/26"), df)
if nrow(df_recent) == 0
    df_recent = df[end-709:end, :]
end

println("\n" * "="^90)
println("📊 2. EMPIRICAL DISPERSION METRICS (RECENT $(nrow(df_recent)) TEST MATCHES: 24/25 & 25/26)")
println("="^90)

metrics_recent = [
    compute_dispersion_metrics(df_recent.home_gross_goals, "Gross Home Goals"),
    compute_dispersion_metrics(df_recent.away_gross_goals, "Gross Away Goals"),
    compute_dispersion_metrics(df_recent.home_open_goals, "Open-Play Home Goals"),
    compute_dispersion_metrics(df_recent.away_open_goals, "Open-Play Away Goals"),
    compute_dispersion_metrics(df_recent.home_pen_goals, "Penalty Home Goals"),
    compute_dispersion_metrics(df_recent.away_pen_goals, "Penalty Away Goals"),
    compute_dispersion_metrics(df_recent.home_gross_goals .+ df_recent.away_gross_goals, "Total Gross Goals"),
    compute_dispersion_metrics(df_recent.home_open_goals .+ df_recent.away_open_goals, "Total Open-Play Goals")
]

disp_df_recent = DataFrame(metrics_recent)
println(disp_df_recent)

# ==============================================================================
# SECTION 2: TEAM-BY-TEAM FANO FACTORS (BEFORE vs AFTER OPEN-PLAY FILTERING)
# ==============================================================================
println("\n" * "="^90)
println("⚽ 3. TEAM-LEVEL EMPIRICAL FANO FACTORS (GROSS vs OPEN-PLAY)")
println("="^90)

all_teams = sort(unique(vcat(df.home_team, df.away_team)))
team_metrics = []

for team in all_teams
    # Matches played as home or away
    h_m = filter(r -> r.home_team == team, df)
    a_m = filter(r -> r.away_team == team, df)
    
    scored_gross = vcat(h_m.home_gross_goals, a_m.away_gross_goals)
    scored_open  = vcat(h_m.home_open_goals, a_m.away_open_goals)
    
    if length(scored_gross) >= 30
        fano_gross = var(scored_gross) / mean(scored_gross)
        fano_open  = var(scored_open) / mean(scored_open)
        push!(team_metrics, (
            team = String(team),
            n_matches = length(scored_gross),
            mean_gross = round(mean(scored_gross), digits=2),
            fano_gross = round(fano_gross, digits=3),
            mean_open = round(mean(scored_open), digits=2),
            fano_open = round(fano_open, digits=3),
            fano_delta = round(fano_open - fano_gross, digits=3)
        ))
    end
end

team_df = sort(DataFrame(team_metrics), :fano_gross, rev=true)
println(team_df)
println("\n  • Average Team Fano Gross: $(round(mean(team_df.fano_gross), digits=3))")
println("  • Average Team Fano Open : $(round(mean(team_df.fano_open), digits=3))")

# ==============================================================================
# SECTION 3: MCMC POSTERIOR DISPERSION PARAMETERS (r_home, r_away)
# ==============================================================================
println("\n" * "="^90)
println("🔬 4. LEARNED MCMC DISPERSION PARAMETERS ACROSS 40 ROLLING FOLDS")
println("="^90)

# Check folders
grid_dirs = [
    "/root/BayesianFootball/data/scottish_negbin_grid",
    "/root/BayesianFootball/data/scottish_open_play_grid"
]

all_exps = []
for gdir in grid_dirs
    if isdir(gdir)
        append!(all_exps, Experiments.list_experiments(gdir))
    end
end

target_models = [
    "goals_negbin_ctl_hl365_hs2",
    "goals_negbin_open_play_hl365_hs2",
    "recomb_negbin_integrated_hl365_hs2"
]

model_r_stats = []

for mod_name in target_models
    matching = filter(p -> occursin(mod_name, p), all_exps)
    if isempty(matching)
        println("⚠️  Experiment not found for: $mod_name")
        continue
    end
    exp_path = matching[end]
    println("Loading: $exp_path")
    exp_res = Experiments.load_experiments(exp_path)
    
    r_h_all_folds = Float64[]
    r_a_all_folds = Float64[]
    
    for split_res in exp_res.results
        chain = split_res.chain
        chain_names = String.(names(chain))
        
        # Check parameter names
        if "log_r" in chain_names && "delta_r_home" in chain_names
            log_r_arr = vec(Array(chain["log_r"]))
            δ_r_arr   = vec(Array(chain["delta_r_home"]))
            r_h_samples = exp.(log_r_arr .+ δ_r_arr)
            r_a_samples = exp.(log_r_arr)
            append!(r_h_all_folds, r_h_samples)
            append!(r_a_all_folds, r_a_samples)
        elseif Symbol("disp.log_r") in names(chain) && Symbol("disp.δ_r_home") in names(chain)
            log_r_arr = vec(Array(chain[Symbol("disp.log_r")]))
            δ_r_arr   = vec(Array(chain[Symbol("disp.δ_r_home")]))
            r_h_samples = exp.(log_r_arr .+ δ_r_arr)
            r_a_samples = exp.(log_r_arr)
            append!(r_h_all_folds, r_h_samples)
            append!(r_a_all_folds, r_a_samples)
        end
    end
    
    if !isempty(r_h_all_folds)
        mean_rh = mean(r_h_all_folds)
        mean_ra = mean(r_a_all_folds)
        q05_rh, q95_rh = quantile(r_h_all_folds, [0.05, 0.95])
        q05_ra, q95_ra = quantile(r_a_all_folds, [0.05, 0.95])
        
        # Theoretical NegBin Fano Factor = 1 + μ / r
        # Assume typical Scottish goals: μ_home ≈ 1.55, μ_away ≈ 1.25
        implied_fano_h = 1.0 + (1.55 / mean_rh)
        implied_fano_a = 1.0 + (1.25 / mean_ra)
        
        push!(model_r_stats, (
            model = mod_name,
            r_home_mean = round(mean_rh, digits=2),
            r_home_90ci = "($(round(q05_rh, digits=1)) - $(round(q95_rh, digits=1)))",
            r_away_mean = round(mean_ra, digits=2),
            r_away_90ci = "($(round(q05_ra, digits=1)) - $(round(q95_ra, digits=1)))",
            implied_fano_home = round(implied_fano_h, digits=4),
            implied_fano_away = round(implied_fano_a, digits=4),
            overdisp_pct_home = round((implied_fano_h - 1.0) * 100, digits=2),
            overdisp_pct_away = round((implied_fano_a - 1.0) * 100, digits=2)
        ))
    end
end

r_summary_df = DataFrame(model_r_stats)
println(r_summary_df)

println("\n" * "="^90)
println("✓ EDA COMPLETED SUCCESSFULLY!")
println("="^90)
