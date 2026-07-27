#=
r01_eda_calibration.jl

Runner file to:
1. Load `li_smile50` model and the Betfair odds data.
2. Evaluate historical bias on key markets (btts_yes, home, over_25).
3. Compute the Global Bias (all data pooled) vs. Walk-Forward time-decay bias.
4. Compare the parameters and the resulting calibration on the Score Matrix.
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics
using Dates

const Evaluation = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Data = BayesianFootball.Data

# Resolve loader paths from the package root so this runs identically whether the file is
# `include()`d (nested includes are otherwise resolved relative to THIS file's dir) or pasted.
const _ROOT = pkgdir(BayesianFootball)
include(joinpath(_ROOT, "current_development/split_market_pillar/l03_local_intensity_poisson.jl"))
include(joinpath(_ROOT, "current_development/score_matrix_calibration/l01_score_matrix_calibration.jl"))

println("[INFO] Loading Ireland dataset...")
ds = Data.load_datastore_cached(Data.Ireland(); max_age_hours=99999)
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds1 = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

smile_dir = "./data/double_poisson_smile_grid/"
exps = Experiments.list_experiments(smile_dir; data_dir="")
all_results = Experiments.load_experiments(exps)
exp_smile = all_results[findfirst(r -> r.config.name == "li_smile50", all_results)]

println("[INFO] Extracting out-of-sample predictions for li_smile50...")
latents = Experiments.extract_oos_predictions(ds1, exp_smile)
ppd = Predictions.model_inference(latents)

mf = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
select!(mf, :match_id, :market_name, :market_line, :selection, :prob_model)
adf = innerjoin(ds1.odds, mf, on = [:match_id, :market_name, :market_line, :selection])
adf = innerjoin(adf, ds1.matches[:, [:match_id, :match_date]], on=:match_id)
dropmissing!(adf, [:prob_fair_close, :is_winner])
adf.spread = Float64.(adf.prob_model) .- Float64.(adf.prob_fair_close)

# NOTE: adf.selection is a Symbol column (mf.selection from model_inference is Symbol[]).
adf.sel_sym = Symbol.(adf.selection)
match_ids = unique(adf.match_id)

# per-line bias helper: (bias, t) of (model_p − reference)
_biast(model_p, ref) = (b = mean(model_p .- ref);
                        se = std(model_p .- ref) / sqrt(length(ref));
                        (round(b, digits=4), round(b / se, digits=2)))
_gmed(d) = (v = filter(!=(0.0), collect(values(d))); isempty(v) ? 0.0 : round(median(v), digits=4))

selections = [
    :home, :draw, :away, :btts_yes, :btts_no,
    :over_15, :under_15, :over_25, :under_25, :over_35, :under_35,
]

println("\n" * "="^80)
println("1. WHERE DOES THE MODEL SIT — vs MARKET and vs REALITY (per line)")
println("   model = posterior-mean p | market = de-vigged prob_fair_close | actual = win freq")
println("   γ_mkt  centers model→market (strip model−market skew) ; γ_real centers model→reality")
println("="^80)

tbl = DataFrame(line=Symbol[], n=Int[], model=Float64[], market=Float64[], actual=Float64[],
                mdl_mkt=Float64[], t_mm=Float64[], mkt_act=Float64[], t_ka=Float64[],
                mdl_act=Float64[], t_ma=Float64[], g_mkt=Float64[], g_real=Float64[])

for s in selections
    d = adf[adf.sel_sym .== s, :]
    nrow(d) < 5 && continue
    mp = Float64.(d.prob_model); mk = Float64.(d.prob_fair_close); ac = Float64.(d.is_winner)
    (b_mm, t_mm) = _biast(mp, mk)     # model − market
    (b_ka, t_ka) = _biast(mk, ac)     # market − actual (is the MARKET miscalibrated?)
    (b_ma, t_ma) = _biast(mp, ac)     # model − actual
    g_mkt  = fit_global_bias(d; target=:prob_fair_close)
    g_real = fit_global_bias(d; target=:is_winner)
    push!(tbl, (s, nrow(d), round(mean(mp), digits=3), round(mean(mk), digits=3),
                round(mean(ac), digits=3), b_mm, t_mm, b_ka, t_ka, b_ma, t_ma,
                round(g_mkt, digits=3), round(g_real, digits=3)))
end
show(tbl; allrows=true, allcols=true, truncate=0); println()

println("""
READ:
 • mdl_mkt (t) = the model−market skew your philosophy strips; γ_mkt undoes it.
 • mkt_act (t) = how far the MARKET itself is from reality on this line. If ≈0, the
   market is the honest anchor and γ_mkt≈γ_real. If large, the two calibrations DIVERGE
   and "which target" is a real bet, not a formality.
 • mdl_act (t) = residual model miscalibration vs reality; γ_real undoes it.
""")

# ==========================================================================
# 2. TILT VALIDATION on btts_yes — market-target vs reality-target, side by side
# ==========================================================================
println("="^80)
println("2. SCORE MATRIX TILT VALIDATION (btts_yes): does each γ center where it claims?")
println("="^80)

btts = adf[adf.sel_sym .== :btts_yes, :]
g_mkt_g  = fit_global_bias(btts; target=:prob_fair_close)
g_real_g = fit_global_bias(btts; target=:is_winner)
g_mkt_wf  = fit_walk_forward_bias(btts; target=:prob_fair_close, half_life_days=90.0)
g_real_wf = fit_walk_forward_bias(btts; target=:is_winner,       half_life_days=90.0)
println("  γ_mkt(global)=$(round(g_mkt_g,digits=3))  γ_real(global)=$(round(g_real_g,digits=3))" *
        "   |  WF medians: mkt=$(_gmed(g_mkt_wf))  real=$(_gmed(g_real_wf))")

mask_btts = mask_for("BTTS", "", "btts_yes")
_btts_p(P) = mean(sum(P[mask_btts, :], dims=1))
function _tilted(P_raw, γ)
    P = copy(P_raw); tilt_score_matrix!(P, [mask_btts], [γ]); _btts_p(P)
end

results = DataFrame(match_id=eltype(match_ids)[], raw_p=Float64[],
                    mkt_g=Float64[], real_g=Float64[], mkt_wf=Float64[], real_wf=Float64[])
for r in eachrow(latents.df)
    mid = r.match_id
    mid in match_ids || continue
    λh, λa = r.λ_h, r.λ_a
    S = length(λh)
    P_raw = Matrix{Float64}(undef, GG * GG, S)
    for s in 1:S
        g = vec(pdf.(Poisson(λh[s]), 0:GG-1) * pdf.(Poisson(λa[s]), 0:GG-1)')
        P_raw[:, s] = g ./ sum(g)
    end
    push!(results, (mid, _btts_p(P_raw),
                    _tilted(P_raw, g_mkt_g), _tilted(P_raw, g_real_g),
                    _tilted(P_raw, get(g_mkt_wf, mid, 0.0)), _tilted(P_raw, get(g_real_wf, mid, 0.0))))
end

df_eval = innerjoin(results, btts[:, [:match_id, :prob_fair_close, :is_winner]], on=:match_id)
mkt = Float64.(df_eval.prob_fair_close); act = Float64.(df_eval.is_winner)

vtbl = DataFrame(variant=String[], bias_vs_mkt=Float64[], t_mkt=Float64[],
                 bias_vs_real=Float64[], t_real=Float64[])
for (name, col) in [("raw", :raw_p), ("tilt→market (global)", :mkt_g), ("tilt→reality (global)", :real_g),
                    ("tilt→market (WF)", :mkt_wf), ("tilt→reality (WF)", :real_wf)]
    p = df_eval[!, col]
    (bm, tm) = _biast(p, mkt); (br, tr) = _biast(p, act)
    push!(vtbl, (name, bm, tm, br, tr))
end
println("\nBTTS_YES (n=$(nrow(df_eval))):  a tilt that centers on its target drives THAT bias→0")
show(vtbl; allrows=true, allcols=true, truncate=0); println()

println("\nDone.")
