# current_development/MetaModels/analysis_headless.jl
#
# Run this in the REPL after running r03_meta_framework_runner.jl
# Requires in scope: chain, joined_data, team_names, team_map
#
# NOTE: This framework's θ_t is on the [0,1] probability scale (logistic-transformed).
#       The "good regime" threshold is θ > mean(θ_t) (i.e. above-average trust in L1).

using Statistics
using LogExpFunctions: logistic

println("\n" * "="^70)
println("  META MODEL CHAIN ANALYSIS (Headless)")
println("="^70)

# ===========================================================================
# 1. MCMC HEALTH CHECK
# ===========================================================================
println("\n--- 1. MCMC DIAGNOSTICS ---")

rhats    = MCMCChains.rhat(chain)
ess_df   = MCMCChains.ess(chain)
rhat_vals = rhats[:, :rhat]
ess_vals  = ess_df[:, :ess]

n_bad_rhat = count(x -> !ismissing(x) && x > 1.05, rhat_vals)
n_total    = count(!ismissing, rhat_vals)
min_ess    = round(minimum(skipmissing(ess_vals)), digits=1)
mean_ess   = round(mean(skipmissing(ess_vals)), digits=1)
acc_rate   = round(mean(chain[:acceptance_rate]), digits=3)

println("  R-hat:           $n_bad_rhat / $n_total params > 1.05  (want 0)")
println("  ESS:             min=$min_ess  mean=$mean_ess  (want > 100)")
println("  Acceptance Rate: $acc_rate  (target 0.65–0.80)")

#=
julia> println("  R-hat:           $n_bad_rhat / $n_total params > 1.05  (want 0)")
  R-hat:           0 / 225 params > 1.05  (want 0)

julia> println("  ESS:             min=$min_ess  mean=$mean_ess  (want > 100)")
  ESS:             min=346.9  mean=648.4  (want > 100)

julia> println("  Acceptance Rate: $acc_rate  (target 0.65–0.80)")
  Acceptance Rate: 0.8  (target 0.65–0.80)
=#


# ===========================================================================
# 2. GLOBAL TRUST PARAMETER
# ===========================================================================
println("\n--- 2. GLOBAL TRUST (α_intercept) ---")
α_samples = vec(chain[:α_intercept])
α_mean    = round(mean(α_samples), digits=3)
α_std     = round(std(α_samples), digits=3)
α_q025    = round(quantile(α_samples, 0.025), digits=3)
α_q975    = round(quantile(α_samples, 0.975), digits=3)
θ_global  = round(logistic(mean(α_samples)), digits=3)

println("  α_intercept: mean=$α_mean  std=$α_std  95% CI [$α_q025, $α_q975]")
println("  → Implied global trust in L1: $θ_global")
println("  weights L1 at $(round(θ_global*100, digits=1))% | market at $(round((1-θ_global)*100, digits=1))%")

#=
julia> println("  α_intercept: mean=$α_mean  std=$α_std  95% CI [$α_q025, $α_q975]")
  α_intercept: mean=-1.01  std=0.832  95% CI [-2.55, 0.659]

julia> println("  → Implied global trust in L1: $θ_global")
  → Implied global trust in L1: 0.267

julia> println("  weights L1 at $(round(θ_global*100, digits=1))% | market at $(round((1-θ_global)*100, digits=1))%")
  weights L1 at 26.7% | market at 73.3%
=#


# ===========================================================================
# 3. GRW DYNAMICS
# ===========================================================================
println("\n--- 3. GRW DYNAMICS (σ_GRW) ---")
σ_GRW_samples = vec(chain[Symbol("dyn_θ_logit.σ_GRW")])
σ_mean = round(mean(σ_GRW_samples), digits=4)
σ_std  = round(std(σ_GRW_samples), digits=4)
println("  σ_GRW: mean=$σ_mean  std=$σ_std")
if σ_mean < 0.05
    println("  >> Trust is very STABLE over the entire period.")
elseif σ_mean < 0.15
    println("  >> Moderate regime variation detected.")
else
    println("  >> STRONG regime variation: reliability fluctuates significantly.")
end

#=
julia> σ_mean = round(mean(σ_GRW_samples), digits=4)
0.0972

julia> σ_std  = round(std(σ_GRW_samples), digits=4)
0.1047

julia> println("  σ_GRW: mean=$σ_mean  std=$σ_std")
  σ_GRW: mean=0.0972  std=0.1047

julia> if σ_mean < 0.05
           println("  >> Trust is very STABLE over the entire period.")
       elseif σ_mean < 0.15
           println("  >> Moderate regime variation detected.")
       else
           println("  >> STRONG regime variation: reliability fluctuates significantly.")
       end
  >> Moderate regime variation detected.
=#


# ===========================================================================
# 4. RECONSTRUCT WEEKLY θ_t TRAJECTORY
# ===========================================================================
println("\n--- 4. WEEKLY TRUST TRAJECTORY (θ_t) ---")

n_weeks = maximum(joined_data.W)
σ_GRW_mean_val = mean(σ_GRW_samples)

z_w_means = [mean(vec(chain[Symbol("dyn_θ_logit.z_w[$w]")])) for w in 1:n_weeks]
logit_drift = cumsum(z_w_means .* σ_GRW_mean_val)
logit_drift_centered = logit_drift .- mean(logit_drift)
θ_t = logistic.(mean(α_samples) .+ logit_drift_centered)

best_week  = argmax(θ_t)
worst_week = argmin(θ_t)

println("  Best week  (θ highest): Week $best_week  → θ = $(round(θ_t[best_week], digits=3))")
println("  Worst week (θ lowest):  Week $worst_week → θ = $(round(θ_t[worst_week], digits=3))")
println("  Range: [$(round(minimum(θ_t), digits=3)), $(round(maximum(θ_t), digits=3))]")
println("  Std Dev across weeks:   $(round(std(θ_t), digits=4))")

quarter = max(1, n_weeks ÷ 4)
println("\n  Quarterly summary:")
for q in 1:4
    ws = (q-1)*quarter + 1
    we = min(q*quarter, n_weeks)
    qv = θ_t[ws:we]
    dir = qv[end] > qv[1] ? "↑" : "↓"
    println("  Q$q (wk $(ws)–$(we)): avg θ=$(round(mean(qv), digits=3))  $dir")
end

#=
julia> println("  Best week  (θ highest): Week $best_week  → θ = $(round(θ_t[best_week], digits=3))")
  Best week  (θ highest): Week 18  → θ = 0.293

julia> println("  Worst week (θ lowest):  Week $worst_week → θ = $(round(θ_t[worst_week], digits=3))")
  Worst week (θ lowest):  Week 191 → θ = 0.24

julia> println("  Range: [$(round(minimum(θ_t), digits=3)), $(round(maximum(θ_t), digits=3))]")
  Range: [0.24, 0.293]

julia> println("  Std Dev across weeks:   $(round(std(θ_t), digits=4))")
  Std Dev across weeks:   0.0169

julia> quarter = max(1, n_weeks ÷ 4)
49

julia> println("\n  Quarterly summary:")

  Quarterly summary:

julia> for q in 1:4
           ws = (q-1)*quarter + 1
           we = min(q*quarter, n_weeks)
           qv = θ_t[ws:we]
           dir = qv[end] > qv[1] ? "↑" : "↓"
           println("  Q$q (wk $(ws)–$(we)): avg θ=$(round(mean(qv), digits=3))  $dir")
       end
  Q1 (wk 1–49): avg θ=0.289  ↓
  Q2 (wk 50–98): avg θ=0.275  ↓
  Q3 (wk 99–147): avg θ=0.259  ↓
  Q4 (wk 148–196): avg θ=0.246  ↓
=#


# ===========================================================================
# 5. REGIME DETECTION (mirroring r02_meta_model_real_data.jl logic)
# ===========================================================================
println("\n--- 5. REGIME DETECTION ---")

# Assign each match its weekly θ
joined_data.θ_t = θ_t[joined_data.W]

# Build weekly PnL (requires :pnl column in joined_data — falls back to outcome if missing)
weekly_pnl = zeros(Float64, n_weeks)
if hasproperty(joined_data, :pnl) && !all(ismissing, joined_data.pnl)
    for i in 1:nrow(joined_data)
        w = joined_data.W[i]
        if !ismissing(joined_data.pnl[i])
            weekly_pnl[w] += joined_data.pnl[i]
        end
    end
    has_pnl = true
else
    # Fallback: approximate PnL as (is_winner - prob_fair_close) — edge proxy
    for i in 1:nrow(joined_data)
        w = joined_data.W[i]
        weekly_pnl[w] += Float64(joined_data.is_winner[i]) - joined_data.prob_fair_close[i]
    end
    has_pnl = false
    println("  (No :pnl column found — using outcome-prob as edge proxy)")
end

# Regime threshold: weeks where θ > median(θ_t) = "Good Regime"
θ_threshold  = mean(θ_t)
good_weeks   = findall(x -> x > θ_threshold, θ_t)
bad_weeks    = findall(x -> x <= θ_threshold, θ_t)

corr           = cor(θ_t, weekly_pnl)
avg_pnl_good   = isempty(good_weeks) ? 0.0 : mean(weekly_pnl[good_weeks])
avg_pnl_bad    = isempty(bad_weeks) ? 0.0  : mean(weekly_pnl[bad_weeks])
total_pnl_good = isempty(good_weeks) ? 0.0 : sum(weekly_pnl[good_weeks])
total_pnl_bad  = isempty(bad_weeks) ? 0.0  : sum(weekly_pnl[bad_weeks])

pnl_label = has_pnl ? "PnL" : "Edge Proxy"

println("  Threshold: θ > $(round(θ_threshold, digits=3)) = Good Regime (above-average trust)")
println("  Correlation (θ_t vs Weekly $pnl_label): $(round(corr, digits=3))")
println("  --------------------------------")
println("  Weeks in Good Regime (θ > thresh): $(length(good_weeks))")
println("  Weeks in Bad Regime  (θ ≤ thresh): $(length(bad_weeks))")
println("  --------------------------------")
println("  Total $pnl_label (Good Regime): $(round(total_pnl_good, digits=4))")
println("  Total $pnl_label (Bad Regime):  $(round(total_pnl_bad,  digits=4))")
println("  Avg Weekly $pnl_label (Good Regime): $(round(avg_pnl_good, digits=4))")
println("  Avg Weekly $pnl_label (Bad Regime):  $(round(avg_pnl_bad,  digits=4))")

# Signal quality
if corr > 0.05
    println("\n  >> POSITIVE correlation: higher θ → better $pnl_label weeks. L1 regime signal is VALID.")
elseif corr < -0.05
    println("\n  >> NEGATIVE correlation: this market's bias is inverted (or model needs more data).")
else
    println("\n  >> WEAK correlation: Meta Model is not yet distinguishing good/bad weeks on this market.")
end


#=
julia> corr           = cor(θ_t, weekly_pnl)
-0.05156152130483727

julia> avg_pnl_good   = isempty(good_weeks) ? 0.0 : mean(weekly_pnl[good_weeks])
0.1774882986097532

julia> avg_pnl_bad    = isempty(bad_weeks) ? 0.0  : mean(weekly_pnl[bad_weeks])
0.24436940913127073

julia> total_pnl_good = isempty(good_weeks) ? 0.0 : sum(weekly_pnl[good_weeks])
15.97394687487779

julia> total_pnl_bad  = isempty(bad_weeks) ? 0.0  : sum(weekly_pnl[bad_weeks])
25.903157367914698

julia> pnl_label = has_pnl ? "PnL" : "Edge Proxy"
"Edge Proxy"

julia> println("  Threshold: θ > $(round(θ_threshold, digits=3)) = Good Regime (above-average trust)")
  Threshold: θ > 0.267 = Good Regime (above-average trust)

julia> println("  Correlation (θ_t vs Weekly $pnl_label): $(round(corr, digits=3))")
  Correlation (θ_t vs Weekly Edge Proxy): -0.052

julia> println("  --------------------------------")
  --------------------------------

julia> println("  Weeks in Good Regime (θ > thresh): $(length(good_weeks))")
  Weeks in Good Regime (θ > thresh): 90

julia> println("  Weeks in Bad Regime  (θ ≤ thresh): $(length(bad_weeks))")
  Weeks in Bad Regime  (θ ≤ thresh): 106

julia> println("  --------------------------------")
  --------------------------------

julia> println("  Total $pnl_label (Good Regime): $(round(total_pnl_good, digits=4))")
  Total Edge Proxy (Good Regime): 15.9739

julia> println("  Total $pnl_label (Bad Regime):  $(round(total_pnl_bad,  digits=4))")
  Total Edge Proxy (Bad Regime):  25.9032

julia> println("  Avg Weekly $pnl_label (Good Regime): $(round(avg_pnl_good, digits=4))")
  Avg Weekly Edge Proxy (Good Regime): 0.1775

julia> println("  Avg Weekly $pnl_label (Bad Regime):  $(round(avg_pnl_bad,  digits=4))")
  Avg Weekly Edge Proxy (Bad Regime):  0.2444

julia> # Signal quality
       if corr > 0.05
           println("\n  >> POSITIVE correlation: higher θ → better $pnl_label weeks. L1 regime signal is VALID.")
       elseif corr < -0.05
           println("\n  >> NEGATIVE correlation: this market's bias is inverted (or model needs more data).")
       else
           println("\n  >> WEAK correlation: Meta Model is not yet distinguishing good/bad weeks on this market.")
       end

  >> NEGATIVE correlation: this market's bias is inverted (or model needs more data).
=#

# ===========================================================================
# 6. TEAM BIASES
# ===========================================================================
println("\n--- 6. TEAM BIASES (δ_team) ---")
σ_team_samples = vec(chain[Symbol("δ_team.σ_team")])
σ_team_mean    = round(mean(σ_team_samples), digits=4)
println("  σ_team: mean=$σ_team_mean  (near 0 = no team-specific signal)")

team_results = NamedTuple[]
for (team, idx) in team_map
    z_sym = Symbol("δ_team.z_team[$idx]")
    if z_sym in keys(chain)
        z_s  = vec(chain[z_sym])
        δ_s  = σ_team_samples .* z_s  # actual posterior bias
        push!(team_results, (
            team   = team,
            δ_mean = mean(δ_s),
            δ_std  = std(δ_s),
            δ_q025 = quantile(δ_s, 0.025),
            δ_q975 = quantile(δ_s, 0.975),
        ))
    end
end
sort!(team_results, by=x -> abs(x.δ_mean), rev=true)

println("\n  Top 10 teams by |bias| (▲ = L1 underestimates, ▼ = L1 overestimates):")
println("  " * rpad("Team", 35) * rpad("δ_mean", 10) * rpad("δ_std", 10) * "95% CI")
println("  " * "-"^75)
for t in first(team_results, 10)
    dir = t.δ_mean > 0 ? "▲" : "▼"
    ci  = "[$(round(t.δ_q025, digits=3)), $(round(t.δ_q975, digits=3))]"
    println("  $dir " * rpad(t.team, 33) *
            rpad(string(round(t.δ_mean, digits=4)), 10) *
            rpad(string(round(t.δ_std,  digits=4)), 10) * ci)
end

sig_teams = filter(t -> t.δ_q025 > 0 || t.δ_q975 < 0, team_results)
println("\n  Teams with 95% CI NOT crossing zero (statistically significant bias):")
if isempty(sig_teams)
    println("  (none — all CIs cross zero, team biases are currently weak)")
else
    for t in sig_teams
        dir = t.δ_mean > 0 ? "▲ OVER" : "▼ UNDER"
        println("  $dir  $(t.team): CI [$(round(t.δ_q025, digits=3)), $(round(t.δ_q975, digits=3))]")
    end
end


#=
julia> println("\n  Top 10 teams by |bias| (▲ = L1 underestimates, ▼ = L1 overestimates):")

  Top 10 teams by |bias| (▲ = L1 underestimates, ▼ = L1 overestimates):

julia> println("  " * rpad("Team", 35) * rpad("δ_mean", 10) * rpad("δ_std", 10) * "95% CI")
  Team                               δ_mean    δ_std     95% CI

julia> println("  " * "-"^75)
  ---------------------------------------------------------------------------

julia> for t in first(team_results, 10)
           dir = t.δ_mean > 0 ? "▲" : "▼"
           ci  = "[$(round(t.δ_q025, digits=3)), $(round(t.δ_q975, digits=3))]"
           println("  $dir " * rpad(t.team, 33) *
                   rpad(string(round(t.δ_mean, digits=4)), 10) *
                   rpad(string(round(t.δ_std,  digits=4)), 10) * ci)
       end
  ▼ cove-rangers                     -0.0131   0.1399    [-0.371, 0.229]
  ▼ edinburgh-city-fc                -0.0129   0.1243    [-0.322, 0.209]
  ▲ stenhousemuir                    0.0127    0.1375    [-0.228, 0.346]
  ▲ alloa-athletic                   0.0108    0.1724    [-0.301, 0.414]
  ▲ dunfermline-athletic             0.0104    0.124     [-0.253, 0.319]
  ▲ stranraer                        0.0098    0.1455    [-0.268, 0.369]
  ▼ east-kilbride                    -0.0085   0.1497    [-0.311, 0.277]
  ▼ east-fife                        -0.008    0.1039    [-0.266, 0.205]
  ▲ inverness-caledonian-thistle     0.0077    0.1583    [-0.276, 0.307]
  ▲ kelty-hearts-fc                  0.0056    0.1382    [-0.268, 0.323]

julia> sig_teams = filter(t -> t.δ_q025 > 0 || t.δ_q975 < 0, team_results)
NamedTuple[]

julia> println("\n  Teams with 95% CI NOT crossing zero (statistically significant bias):")

  Teams with 95% CI NOT crossing zero (statistically significant bias):

julia> if isempty(sig_teams)
           println("  (none — all CIs cross zero, team biases are currently weak)")
       else
           for t in sig_teams
               dir = t.δ_mean > 0 ? "▲ OVER" : "▼ UNDER"
               println("  $dir  $(t.team): CI [$(round(t.δ_q025, digits=3)), $(round(t.δ_q975, digits=3))]")
           end
       end
  (none — all CIs cross zero, team biases are currently weak)
=#


# ===========================================================================
# 7. PREDICTIVE FIT (Brier & Log-Loss)
# ===========================================================================
println("\n--- 7. PREDICTIVE FIT ---")

Q_meta   = θ_global .* joined_data.p_L1 .+ (1 - θ_global) .* joined_data.prob_fair_close
Y        = Float64.(joined_data.is_winner)

brier_meta   = mean((Q_meta .- Y).^2)
brier_market = mean((joined_data.prob_fair_close .- Y).^2)
brier_l1     = mean((joined_data.p_L1 .- Y).^2)

clip(v) = clamp.(v, 1e-7, 1.0 - 1e-7)
ll(p, y) = -mean(y .* log.(clip(p)) .+ (1 .- y) .* log.(clip(1 .- p)))

ll_meta   = ll(Q_meta, Y)
ll_market = ll(joined_data.prob_fair_close, Y)
ll_l1     = ll(joined_data.p_L1, Y)

println("  " * rpad("Model", 22) * rpad("Brier Score", 15) * "Log-Loss")
println("  " * "-"^52)
println("  " * rpad("Market (Baseline)", 22) * rpad(string(round(brier_market, digits=5)), 15) * string(round(ll_market, digits=5)))
println("  " * rpad("Layer 1 (Raw)",     22) * rpad(string(round(brier_l1,     digits=5)), 15) * string(round(ll_l1,     digits=5)))
println("  " * rpad("Meta (θ-mixture)",  22) * rpad(string(round(brier_meta,   digits=5)), 15) * string(round(ll_meta,   digits=5)))
println("  Brier improvement vs Market: $(round((brier_market - brier_meta)*10000, digits=2)) basis points")
println("  Brier improvement vs L1:     $(round((brier_l1 - brier_meta)*10000, digits=2)) basis points")

println("\n" * "="^70)
println("  ANALYSIS COMPLETE")
println("="^70)




#=
julia> brier_meta   = mean((Q_meta .- Y).^2)
0.16965207988166556

julia> brier_market = mean((joined_data.prob_fair_close .- Y).^2)
0.16935339054734705

julia> brier_l1     = mean((joined_data.p_L1 .- Y).^2)
0.17128917013347233

julia> clip(v) = clamp.(v, 1e-7, 1.0 - 1e-7)
clip (generic function with 1 method)

julia> ll(p, y) = -mean(y .* log.(clip(p)) .+ (1 .- y) .* log.(clip(1 .- p)))
ll (generic function with 1 method)

julia> ll_meta   = ll(Q_meta, Y)
0.521992022841272

julia> ll_market = ll(joined_data.prob_fair_close, Y)
0.5210380606979795

julia> ll_l1     = ll(joined_data.p_L1, Y)
0.5264221176488142

julia> println("  " * rpad("Model", 22) * rpad("Brier Score", 15) * "Log-Loss")
  Model                 Brier Score    Log-Loss

julia> println("  " * "-"^52)
  ----------------------------------------------------

julia> println("  " * rpad("Market (Baseline)", 22) * rpad(string(round(brier_market, digits=5)), 15) * string(round(ll_market, digits=5)))
  Market (Baseline)     0.16935        0.52104

julia> println("  " * rpad("Layer 1 (Raw)",     22) * rpad(string(round(brier_l1,     digits=5)), 15) * string(round(ll_l1,     digits=5)))
  Layer 1 (Raw)         0.17129        0.52642

julia> println("  " * rpad("Meta (θ-mixture)",  22) * rpad(string(round(brier_meta,   digits=5)), 15) * string(round(ll_meta,   digits=5)))
  Meta (θ-mixture)      0.16965        0.52199

julia> println("  Brier improvement vs Market: $(round((brier_market - brier_meta)*10000, digits=2)) basis points")
  Brier improvement vs Market: -2.99 basis points

julia> println("  Brier improvement vs L1:     $(round((brier_l1 - brier_meta)*10000, digits=2)) basis points")
  Brier improvement vs L1:     16.37 basis points

julia> println("\n" * "="^70)

======================================================================

julia> println("  ANALYSIS COMPLETE")
  ANALYSIS COMPLETE

julia> println("="^70)
======================================================================
=#

