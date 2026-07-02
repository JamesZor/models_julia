#=
r20 — SMOKE runner for the GRADUATED local-intensity SMILE engine.

Proof that l03's `LocalIntensitySmileDoublePoissonModel` has been graduated into `src/` as
`DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel`. This file builds the model FROM SRC
ONLY — it deliberately does NOT `include(...)` any l0X loader (that's the graduation test). It
trains one split on Ireland and runs six end-to-end verdict checks (each prints ✅/❌).

src pieces exercised:
  • Features.MarketSmileFeature + add_feature!            (src/features/extractors/market_extractors.jl)
  • PreGame.DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel  (…/time_decay/outfield_xg_smile_double_poisson.jl)
  • Predictions SmileScoreMatrix O/U path               (src/predictions/score_computation/smile_poisson.jl)

Run after git pull + REPL restart (structs change → Revise won't pick them up):
    include("current_development/split_market_pillar/r20_smoke_src_smile.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning
using ProgressMeter

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Evaluation  = BayesianFootball.Evaluation
const Data        = BayesianFootball.Data

# NB: NO `include` of any l0X loader — the whole point is that src alone builds the model.
verdicts = Tuple{String, Bool}[]
_mark(name, ok) = (push!(verdicts, (name, ok)); println(ok ? "✅ $name" : "❌ $name"))

# ==========================================
# 1. DATA — train pillar from plain ds.odds (SofaScore); eval vs Betfair ds1
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds  = Data.load_datastore_cached(Data.Ireland())
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds1  = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

save_dir = "./data/split_market_dev_area/"
mkpath(save_dir)

# ==========================================
# 2. MODEL — src struct, keeper defaults (smile_weight=0.5, sup=1.0, Kmax=4)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

model = PreGame.DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    smile_feature          = Features.MarketSmileFeature(Kmax = 4),
    market_on              = true,
    supremacy_weight       = 1.0,
    smile_weight           = 0.5,
)
KMAX = model.smile_feature.Kmax
nK   = KMAX + 1

# ==========================================
# CHECK 1 — builds from src alone
# ==========================================
println("\n", "="^60, "\nCHECK 1 — src-only build\n", "="^60)
_mark("1. model is a src PreGame engine (no loader include)",
      model isa PreGame.DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel)

# ==========================================
# CHECK 2 — feature plumbing (required_features + Λ^mkt(K) inversion)
# ==========================================
println("\n", "="^60, "\nCHECK 2 — smile feature plumbing\n", "="^60)
reqs = Features.required_features(model)
has_smile_req = any(f -> f isa Features.MarketSmileFeature, reqs)
_mark("2a. required_features declares MarketSmileFeature", has_smile_req)

# Directly exercise the src extractor (off-AD Poisson-CDF inversion) on matches with odds.
F = Dict{Symbol, Any}()
smile_ids = unique(ds.odds.match_id)
Features.add_feature!(F, Features.MarketSmileFeature(Kmax = KMAX), smile_ids, Dict{Any,Any}(), ds)
logΛ = F[:flat_smile_logΛ]; mask = F[:flat_smile_mask]
# Rows where the full ladder inverted: all strikes present.
full_rows = findall(i -> all(mask[i, :] .> 0.5), 1:size(mask, 1))
finite_ok = !isempty(full_rows) && all(isfinite, logΛ[full_rows, :]) && all(exp.(logΛ[full_rows, :]) .> 0)
# "Ordered sensibly": Λ^mkt(K) rises with strike K (the smile) for the median full-ladder match.
Λ_med = isempty(full_rows) ? Float64[] : exp.(vec(median(logΛ[full_rows, :], dims=1)))
rising = length(Λ_med) == nK && all(diff(Λ_med) .> 0)
_mark("2b. Λ^mkt(K) finite & positive on full-ladder matches (n=$(length(full_rows)))", finite_ok)
_mark("2c. median Λ^mkt(K) rises with strike (the smile)", rising)
println("    median Λ^mkt(K) over strikes 0.5→$(KMAX+0.5): ", round.(Λ_med, digits=3))

# ==========================================
# 3. TRAIN — one split
# ==========================================
println("\n", "="^60, "\nTRAIN — single split (Ireland 2026)\n", "="^60)
task = Experiments.create_experiment_task(
    ds, model, "r20_smoke_src_smile", save_dir;
    target_seasons  = ["2026"],
    history_seasons = 2,
    warmup_period   = 21,
    dynamics_col    = :match_week,
    samples         = 1000,
    warmup          = 500,
    chains          = 4,
    use_queue       = true,
    max_depth       = 10,
)
res = Experiments.run_experiment(task)
Experiments.save_experiment(res)

# ==========================================
# CHECK 3 — convergence on the smile params (RAW chain; check_convergence drops log_φ/σ_smile)
# ==========================================
println("\n", "="^60, "\nCHECK 3 — convergence (raw-chain ess_rhat)\n", "="^60)
raw_chain = res.training_results[1][1]   # MCMCChains.Chains for the (single) split
er = DataFrame(MCMCChains.ess_rhat(raw_chain))
rhat_col = :rhat in propertynames(er) ? :rhat :
           first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
_rhat(p) = (row = er[er.parameters .== Symbol(p), rhat_col]; isempty(row) ? NaN : row[1])

smile_params = vcat(["σ_smile", "σ_sup", "ν_xg"], ["log_φ[$k]" for k in 1:nK])
present = filter(p -> !isnan(_rhat(p)), smile_params)
smile_rhats = [_rhat(p) for p in present]
println("  smile-param R-hat:")
for (p, r) in zip(present, smile_rhats); println("    $(rpad(p, 12)) rhat=$(round(r, digits=4))"); end
max_rhat_all = maximum(skipmissing(replace(er[!, rhat_col], NaN => missing)))
_mark("3a. all smile params found in raw chain (σ_smile, σ_sup, log_φ[1:$nK])",
      length(present) == length(smile_params))
_mark("3b. smile-param R-hat ≤ 1.01", !isempty(smile_rhats) && maximum(smile_rhats) ≤ 1.01)
_mark("3c. global max R-hat ≤ 1.05 (whole model)", max_rhat_all ≤ 1.05)
println("    global max R-hat = $(round(max_rhat_all, digits=4))")

# ==========================================
# CHECK 4 — φ shape sanity (gentle monotone ≈0.93→1.05, every CI crosses 1.0)
# ==========================================
println("\n", "="^60, "\nCHECK 4 — φ(K) shape\n", "="^60)
φ_mean = zeros(nK); φ_lo = zeros(nK); φ_hi = zeros(nK)
for k in 1:nK
    d = exp.(vec(Array(raw_chain[Symbol("log_φ[$k]")])))
    φ_mean[k] = mean(d); φ_lo[k] = quantile(d, 0.05); φ_hi[k] = quantile(d, 0.95)
end
println("  strike K+0.5 | φ mean [5%,95%]")
for k in 1:nK
    println("    $(rpad(string(k-1+0.5), 5)) | $(round(φ_mean[k],digits=3))  [$(round(φ_lo[k],digits=3)), $(round(φ_hi[k],digits=3))]")
end
ci_crosses_1 = all(φ_lo .< 1.0 .< φ_hi)
gentle_range = all(0.80 .< φ_mean .< 1.20)     # known result ≈0.93→1.05; loose smoke bound
_mark("4a. every φ(K) 90% CI crosses 1.0 (flat pregame smile)", ci_crosses_1)
_mark("4b. φ(K) means in a gentle band (0.80–1.20)", gentle_range)

# ==========================================
# CHECK 5 — PPD end-to-end + smile O/U ≠ grid O/U (φ is actually used)
# ==========================================
println("\n", "="^60, "\nCHECK 5 — PPD + smile-vs-grid O/U\n", "="^60)
ppd = Predictions.model_inference(ds1, res)
ppd_ok = ppd isa Predictions.PPD && nrow(ppd) > 0
_mark("5a. model_inference runs, non-empty PPD (no :r ArgumentError)", ppd_ok)

# Direct proof the smile path prices O/U with φ (differs from the grid-only price) for the same draws.
latents = Experiments.extract_oos_predictions(ds1, res)
ou_diffs = Float64[]
for row in eachrow(latents.df)
    params = Predictions.extract_params(model, row)
    S = Predictions.compute_score_matrix(model, params)         # SmileScoreMatrix
    S isa Predictions.SmileScoreMatrix || continue
    for K in 0:(nK-1)
        mkt = Data.MarketOverUnder(K + 0.5)
        smile_p = Predictions.compute_market_probs(S, mkt)               # φ path
        grid_p  = Predictions.compute_market_probs(S.grid, mkt)          # plain grid
        ks = Data.outcomes(mkt)
        push!(ou_diffs, abs(mean(smile_p[ks.over]) - mean(grid_p[ks.over])))
    end
end
max_ou_diff = isempty(ou_diffs) ? 0.0 : maximum(ou_diffs)
_mark("5b. score matrix is SmileScoreMatrix (dedicated dispatch, not plain Poisson)",
      !isempty(ou_diffs))
_mark("5c. smile O/U differs from grid O/U (φ actually priced; max Δ=$(round(max_ou_diff,digits=4)))",
      max_ou_diff > 1e-4)

# ==========================================
# CHECK 6 — LogLoss vs Betfair runs end-to-end (INFORMATIONAL on a single split)
# ==========================================
# NB: a single fold's OOS set is tiny (n≈4 matches on Ireland 2026 fold-0), so the diff_ll
# NUMBER is pure noise and can sit either side of 0 — the meaningful ≈−0.02 reference is a
# FULL-CV figure (r10, ~200+ matches). So check 6 only asserts the eval runs and returns a
# finite value; the number is printed for reference, not gated on the full-CV range.
println("\n", "="^60, "\nCHECK 6 — LogLoss vs Betfair runs (single split; number is informational)\n", "="^60)
diff_ll = NaN; n_obs = 0
try
    ll = Evaluation.evaluate_experiments(Evaluation.LogLoss(), [res], ds1)
    Evaluation.display_summary_metric(ll, :logloss)
    lldf = ll isa DataFrame ? ll : (hasproperty(ll, :df) ? ll.df : DataFrame())
    dcol = first(filter(c -> occursin("diff", lowercase(string(c))), propertynames(lldf)))
    diff_ll = lldf[1, dcol]
    ncol = filter(c -> occursin("n_obs", lowercase(string(c))), propertynames(lldf))
    n_obs = isempty(ncol) ? 0 : lldf[1, first(ncol)]
catch e
    @warn "LogLoss eval failed" exception=(e, catch_backtrace())
end
_mark("6. LogLoss eval runs end-to-end, finite diff_ll", isfinite(diff_ll))
println("    diff_ll (model − market) = $(round(diff_ll, digits=4)) on n_obs=$(n_obs) " *
        "(single-fold noise; full-CV reference ≈ −0.02)")

# ==========================================
# SUMMARY
# ==========================================
println("\n", "="^60, "\nR20 SMOKE SUMMARY\n", "="^60)
for (name, ok) in verdicts; println(ok ? "✅ $name" : "❌ $name"); end
n_pass = count(last, verdicts)
println("\n$(n_pass)/$(length(verdicts)) checks passed.")
println(n_pass == length(verdicts) ?
    ">> GRADUATION VERIFIED: src builds & prices the smile engine end-to-end." :
    ">> SOME CHECKS FAILED — inspect above before recording graduation.")
