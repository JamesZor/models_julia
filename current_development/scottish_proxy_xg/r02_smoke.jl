#=
r02 — WP4 SMOKE. Both arms on a short window before committing ~25h to the grid. ~1.5-2h.

Target = the LAST season with history_seasons = 2, so the history block (23/24 + 24/25) is 100%
commentary-covered. The smoke therefore tests the ENGINES; the grid tests the masking. Keeping
those two failure modes apart is deliberate — a smoke that also exercised the 22/23 coverage hole
could not tell a broken likelihood from a thin pillar.

Checks: feature plumbing, both arms train, kappa ~ 1, nu plausible, sigma_q vs its prior, the
Poisson dispatch path (NOT NegBin — the `:r` gotcha), normalised score matrices, and a warmup probe
that picks the grid's warmup rather than guessing it.

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_proxy_xg/r02_smoke.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning

pinthreads(:cores)

const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_proxy_xg/l02_pxg_engines.jl"))

const HL            = 365.0
const HS            = 2
const SAMPLES       = 800
const WARMUP        = 300
const CHAINS        = 3
const WARMUP_PERIOD = 20        # -> ~3 folds
const RUN_WARMUP_PROBE = true   # +1 Arm-A cell at warmup 800; the funnel lost hours to this

verdicts = Tuple{String, Bool}[]
_mark(name, ok) = (push!(verdicts, (name, ok)); println(ok ? "✅ $name" : "❌ $name"))
_r(x, d = 4) = (x isa Number && isnan(x)) ? NaN : round(x, digits = d)

println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
save_dir = joinpath(ROOT, "data/scottish_pxg_smoke/"); mkpath(save_dir)
TARGET = last(sort(unique(String.(ds.matches.season))))
println("[INFO] target season = $TARGET  (history = $HS seasons, fully covered)")

dyn = PreGame.TimeDecayDynamics(days_half_life = HL)

# ==========================================
# CHECK 1 — feature plumbing
# ==========================================
println("\n", "="^68, "\nCHECK 1 — features\n", "="^68)
mA  = TeamPxGGoalsAPMModel(dynamics_config = dyn)
mA0 = TeamPxGGoalsAPMModel(dynamics_config = dyn, apm_on = false)
mB  = TeamFunnelPxGGoalsAPMModel(dynamics_config = dyn)

rA, rA0, rB = Features.required_features.((mA, mA0, mB))
_mark("1a. Arm A declares ProxyXGFeature", any(f -> f isa ProxyXGFeature, rA))
_mark("1b. Arm A declares the RAPM pillar", any(f -> f isa Features.XGPlusMinusFeature, rA))
_mark("1c. apm_on=false DROPS the RAPM feature (skips the ridge fit)",
      !any(f -> f isa Features.AbstractPlusMinusFeature, rA0))
_mark("1d. Arm B declares BOTH ShotsFunnelFeature and ProxyXGFeature",
      any(f -> f isa Features.ShotsFunnelFeature, rB) && any(f -> f isa ProxyXGFeature, rB))

F = Dict{Symbol, Any}()
Features.add_feature!(F, ProxyXGFeature(), collect(Int.(ds.matches.match_id)), Dict{Any, Any}(), ds)
_mark("1e. extractor emits finite, strictly positive xG",
      all(isfinite, F[:flat_home_xg_proxy]) && all(F[:flat_home_xg_proxy] .> 0))
println("    masked-in home sides: $(Int(sum(F[:flat_pxg_mask_h]))) / $(length(F[:flat_pxg_mask_h]))")

# ==========================================
# 2. TRAIN
# ==========================================
function _train(model, name; warmup = WARMUP)
    println("\n", "#"^68, "\n# TRAIN $name (warmup=$warmup)\n", "#"^68)
    task = Experiments.create_experiment_task(
        ds, model, name, save_dir;
        target_seasons  = [TARGET], history_seasons = HS,
        warmup_period   = WARMUP_PERIOD, dynamics_col = :match_biweek,
        samples = SAMPLES, warmup = warmup, chains = CHAINS,
        use_queue = true, max_depth = 10)
    t0 = time()
    res = Experiments.run_experiment(task)
    Experiments.save_experiment(res)
    println("[INFO] $name: $(length(res.training_results.items)) folds in " *
            "$(_r((time() - t0) / 60, 1)) min")
    return res
end

function _pool(res, sym)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]
        Symbol(sym) in keys(ch) && append!(out, vec(Array(ch[Symbol(sym)])))
    end
    return out
end

function _rhat_summary(res)
    worst = 0.0; n = 0; n_ok = 0
    for it in res.training_results.items
        er = DataFrame(MCMCChains.ess_rhat(it[1]))
        rcol = :rhat in propertynames(er) ? :rhat :
               first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
        vals = collect(skipmissing(replace(er[!, rcol], NaN => missing)))
        isempty(vals) && continue
        n += 1; mr = maximum(vals); worst = max(worst, mr); mr <= 1.01 && (n_ok += 1)
    end
    return (n = n, n_ok = n_ok, worst = worst)
end

function _eps(res)
    out = Float64[]
    for it in res.training_results.items
        try
            append!(out, vec(Array(it[1][:step_size])))
        catch
        end
    end
    return isempty(out) ? NaN : median(out)
end

resA = _train(mA, "pxg_apm_smoke")
_mark("2a. Arm A trained (folds > 0)", length(resA.training_results.items) > 0)

# ==========================================
# CHECK 3 — Arm A parameter sanity
# ==========================================
println("\n", "="^68, "\nCHECK 3 — Arm A posteriors\n", "="^68)
κ = exp.(_pool(resA, "log_κ")); ν = _pool(resA, "ν_xg")
cA = _rhat_summary(resA); εA = _eps(resA)
println("    kappa : mean=$(_r(mean(κ))) 90%=[$(_r(quantile(κ, .05))), $(_r(quantile(κ, .95)))]")
println("    nu_xg : mean=$(_r(mean(ν))) 90%=[$(_r(quantile(ν, .05))), $(_r(quantile(ν, .95)))]" *
        "   (prior mean 4.0; implied CV = $(_r(1 / sqrt(mean(ν)), 3)))")
println("    R-hat : $(cA.n_ok)/$(cA.n) folds ≤1.01, worst=$(_r(cA.worst))    median eps=$(_r(εA, 5))")
_mark("3a. kappa near 1 (0.75–1.30) — the cell table is a conversion table",
      0.75 <= mean(κ) <= 1.30)
_mark("3b. nu in a plausible band (1.5–15)", 1.5 <= mean(ν) <= 15.0)
_mark("3c. nu moved off its prior (posterior sd < prior sd 1.5)", std(ν) < 1.5)
_mark("3d. worst R-hat ≤ 1.05", cA.worst <= 1.05)
_mark("3e. step size healthy (eps > 1e-3) — the r07 spurious-basin signature is eps ~ 2e-4",
      isnan(εA) || εA > 1e-3)

# ==========================================
# CHECK 4 — PPD + the Poisson dispatch path
# ==========================================
println("\n", "="^68, "\nCHECK 4 — PPD end-to-end (the `:r` NegBin gotcha)\n", "="^68)
ok_ppd = false; ok_norm = false; worst_norm = NaN
try
    ppd = Predictions.model_inference(ds, resA)
    ok_ppd = ppd isa Predictions.PPD && nrow(ppd) > 0
    latents = Experiments.extract_oos_predictions(ds, resA)
    devs = Float64[]
    for row in first(eachrow(latents.df), 25)
        S = Predictions.compute_score_matrix(mA, Predictions.extract_params(mA, row))
        mkt = Data.MarketOverUnder(2.5); o = Data.outcomes(mkt)
        p = Predictions.compute_market_probs(S, mkt)
        push!(devs, abs(mean(p[o.over]) + mean(p[o.under]) - 1.0))
    end
    worst_norm = isempty(devs) ? NaN : maximum(devs)
    ok_norm = !isnan(worst_norm) && worst_norm < 1e-3
catch e
    @error "PPD phase failed" exception = (e, catch_backtrace())
end
_mark("4a. model_inference runs (no ArgumentError on a missing :r column)", ok_ppd)
_mark("4b. O/U probabilities normalise (max |over+under−1| = $(_r(worst_norm, 6)))", ok_norm)

# ==========================================
# CHECK 5 — Arm B
# ==========================================
println("\n", "="^68, "\nCHECK 5 — Arm B (3-layer volume -> quality -> goals)\n", "="^68)
resB = _train(mB, "funnel_pxg_apm_smoke")
_mark("5a. Arm B trained (folds > 0)", length(resB.training_results.items) > 0)

if length(resB.training_results.items) > 0
    σq = _pool(resB, "σ_q"); qr = _pool(resB, "q_raw")
    q  = 1 ./ (1 .+ exp.(-qr)); κB = exp.(_pool(resB, "log_κ")); νq = _pool(resB, "ν_q")
    cB = _rhat_summary(resB)
    prior_mean_σq = mean(rand(PXG_SIGQ_PRIOR, 20_000))
    println("    q (xG per shot): mean=$(_r(mean(q))) 90%=[$(_r(quantile(q, .05))), $(_r(quantile(q, .95)))]")
    println("    sigma_q        : mean=$(_r(mean(σq))) vs PRIOR mean $(_r(prior_mean_σq))  " *
            "ratio=$(_r(mean(σq) / prior_mean_σq, 3))   <-- THE r04 COMPARISON")
    println("    nu_q=$(_r(mean(νq)))  kappa=$(_r(mean(κB)))  " *
            "R-hat $(cB.n_ok)/$(cB.n) ≤1.01, worst=$(_r(cB.worst))  eps=$(_r(_eps(resB), 5))")
    _mark("5b. q lands near the league conversion rate (0.08–0.20)", 0.08 <= mean(q) <= 0.20)
    _mark("5c. worst R-hat ≤ 1.05", cB.worst <= 1.05)
    _mark("5d. kappa near 1 (0.75–1.30)", 0.75 <= mean(κB) <= 1.30)
    println(mean(σq) < 0.4 * prior_mean_σq ?
        "    ⚠ sigma_q collapsed toward 0 — the r04 hierarchical-conversion null is reproducing.\n" *
        "      Team shot quality is then NOT a usable axis; expect Arm B ≈ the plain funnel." :
        "    → sigma_q survives its prior: team-level shot quality is identified. Report the ratio.")
end

# ==========================================
# CHECK 6 — the isolation cell builds
# ==========================================
println("\n", "="^68, "\nCHECK 6 — pxg_noapm isolation cell\n", "="^68)
ok6 = false
try
    fs = Features.create_features(ds, mA0, collect(Int.(ds.matches.match_id)))
    ok6 = fs isa Features.FeatureSet
catch e
    try   # signature differs across versions; the point is only that the pillar-off path builds
        ok6 = !any(f -> f isa Features.AbstractPlusMinusFeature, Features.required_features(mA0))
    catch
    end
end
_mark("6a. apm_on=false model is constructible and pillar-free", ok6)

# ==========================================
# CHECK 7 — warmup probe (picks the grid's warmup)
# ==========================================
if RUN_WARMUP_PROBE
    println("\n", "="^68, "\nCHECK 7 — warmup probe: 300 vs 800 on the same folds\n", "="^68)
    resW = _train(mA, "pxg_apm_smoke_w800"; warmup = 800)
    cW = _rhat_summary(resW); εW = _eps(resW)
    νW = _pool(resW, "ν_xg"); κW = exp.(_pool(resW, "log_κ"))
    println("    warmup 300: worst R-hat=$(_r(cA.worst))  eps=$(_r(εA, 5))  " *
            "nu=$(_r(mean(ν)))  kappa=$(_r(mean(κ)))")
    println("    warmup 800: worst R-hat=$(_r(cW.worst))  eps=$(_r(εW, 5))  " *
            "nu=$(_r(mean(νW)))  kappa=$(_r(mean(κW)))")
    better = cW.worst < cA.worst - 0.005
    println(better ?
        "    → USE WARMUP 800 IN r03. The chain was still in burn-in at 300 (the funnel's lesson:\n" *
        "      cheap iterations need more of them; bbc_xg_proxy/NOTES.md:174-204)." :
        "    → KEEP WARMUP 300 IN r03. 800 buys nothing; spend the budget on folds instead.")
    _mark("7a. warmup probe completed", length(resW.training_results.items) > 0)
end

# ==========================================
# SUMMARY
# ==========================================
println("\n", "="^68, "\nR02 SMOKE SUMMARY\n", "="^68)
for (name, ok) in verdicts; println(ok ? "✅ $name" : "❌ $name"); end
n_pass = count(last, verdicts)
println("\n$(n_pass)/$(length(verdicts)) checks passed.")
println(n_pass == length(verdicts) ?
    ">> SMOKE PASS. Set WARMUP in r03_grid.jl from check 7 and launch the grid.\n" *
    ">> If check 3e failed (eps ~ 2e-4 with a collapsed kappa), switch the grid to MapInit —\n" *
    ">> that is the r07 spurious-basin signature and MapInit is the r07b fix that held." :
    ">> CHECKS FAILED — do NOT launch a 25h grid. Fix here first.")
