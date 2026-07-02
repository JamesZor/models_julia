#=
r02_basic_calibration.jl

Market-target calibration (the PRODUCTION path, per `calibrate-centre-edge-in-tails`) with the
reality-target γ reported ALONGSIDE as a diagnostic (never applied). Validates, per line, that the
market-target tilt centers the model−market bias, comparing:
  • POOLED γ  — fit on ALL data (in-sample; bias→0 by construction, the reference).
  • WALK-FORWARD γ — fit on ONLY past data w/ 90d half-life (the honest OUT-OF-SAMPLE test).

Marginal-space calibration:  cal_p = logistic(γ_mkt + logit(model_p)).
This is pricing-route agnostic — it acts on `prob_model` directly, so it is identical for
grid-priced markets (1X2/BTTS) and smile-Λ-priced totals (O/U). The 144-grid tilt
(`tilt_score_matrix!`, validated in r01) is only needed to hand a COHERENT JOINT to the Kelly
allocator (r03); note the production tilt is applied PER POSTERIOR DRAW (shifts the whole PPD),
whereas this bias check uses the posterior-mean `prob_model` — fine for systematic centering, and
they differ only by a small Jensen term (logistic is nonlinear). For the smile model, totals are
priced off Λ, so their production tilt is a Λ-shift — deferred to r03.

Server:
  ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"   # broken LanguageServer/JSONRPC dep on server env
  using BayesianFootball
  include("current_development/score_matrix_calibration/r02_basic_calibration.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics
using Dates
using LogExpFunctions: logit, logistic

const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Data        = BayesianFootball.Data

const _ROOT = pkgdir(BayesianFootball)
include(joinpath(_ROOT, "current_development/split_market_pillar/l03_local_intensity_poisson.jl"))
include(joinpath(_ROOT, "current_development/score_matrix_calibration/l01_score_matrix_calibration.jl"))

# ==========================================================================
# 1. DATA + OOS PREDICTIONS  (li_smile50, Ireland, betfair anchor — same as r01)
# ==========================================================================
println("[INFO] Loading Ireland + li_smile50 OOS predictions...")
ds   = Data.load_datastore_cached(Data.Ireland(); max_age_hours=99999)
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds1  = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

exps        = Experiments.list_experiments("./data/double_poisson_smile_grid/"; data_dir="")
all_results = Experiments.load_experiments(exps)
exp_smile   = all_results[findfirst(r -> r.config.name == "li_smile50", all_results)]

latents = Experiments.extract_oos_predictions(ds1, exp_smile)
ppd     = Predictions.model_inference(latents)
mf = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
select!(mf, :match_id, :market_name, :market_line, :selection, :prob_model)
adf = innerjoin(ds1.odds, mf, on = [:match_id, :market_name, :market_line, :selection])
adf = innerjoin(adf, ds1.matches[:, [:match_id, :match_date]], on=:match_id)
dropmissing!(adf, [:prob_fair_close, :is_winner])
adf.sel_sym = Symbol.(adf.selection)

# ==========================================================================
# 2. HELPERS
# ==========================================================================
_biast(p, ref) = (b = mean(p .- ref); se = std(p .- ref) / sqrt(length(ref)); (round(b, digits=4), round(b / se, digits=2)))
_gmed(d)       = (v = filter(!=(0.0), collect(values(d))); isempty(v) ? 0.0 : round(median(v), digits=4))
_cal(p, γ)     = logistic.(γ .+ logit.(clamp.(Float64.(p), 1e-6, 1.0 - 1e-6)))   # γ scalar or per-row vector

selections = [:home, :draw, :away, :btts_yes, :btts_no,
              :over_15, :under_15, :over_25, :under_25, :over_35, :under_35]

# ==========================================================================
# 3. PER-LINE VALIDATION: pooled vs walk-forward market-target tilt
# ==========================================================================
println("\n" * "="^104)
println("MARKET-TARGET CALIBRATION — pooled (in-sample) vs walk-forward (OOS, past-only, 90d half-life)")
println("  bias = cal_p − market.  pooled→0 by construction; WF is the honest OOS test (n_wf = matches w/ γ≠0).")
println("  g_real = reality-target γ (DIAGNOSTIC ONLY, not applied) — big |g_real−g_mkt| ⇒ market≠reality.")
println("="^104)

tbl = DataFrame(line=Symbol[], n=Int[],
                raw_b=Float64[], raw_t=Float64[],
                pool_b=Float64[], pool_t=Float64[],
                n_wf=Int[], wf_b=Float64[], wf_t=Float64[],
                g_mkt=Float64[], g_mkt_wf=Float64[], g_real=Float64[])

for s in selections
    d = adf[adf.sel_sym .== s, :]
    nrow(d) < 5 && continue
    d = sort(d, :match_date)
    mp = Float64.(d.prob_model); mk = Float64.(d.prob_fair_close)

    g_mkt_g  = fit_global_bias(d; target=:prob_fair_close)
    g_mkt_wf = fit_walk_forward_bias(d; target=:prob_fair_close, half_life_days=90.0)
    g_real_g = fit_global_bias(d; target=:is_winner)

    pool_p = _cal(mp, g_mkt_g)
    γ_row  = [get(g_mkt_wf, m, 0.0) for m in d.match_id]
    wf_p   = _cal(mp, γ_row)
    active = γ_row .!= 0.0                                   # OOS matches with a fitted γ

    (rb, rt) = _biast(mp, mk)
    (pb, pt) = _biast(pool_p, mk)
    (wb, wt) = sum(active) >= 3 ? _biast(wf_p[active], mk[active]) : (NaN, NaN)

    push!(tbl, (s, nrow(d), rb, rt, pb, pt, sum(active), wb, wt,
                round(g_mkt_g, digits=3), _gmed(g_mkt_wf), round(g_real_g, digits=3)))
end
show(tbl; allrows=true, allcols=true, truncate=0); println()

println("""

$("="^104)
READ:
 • raw_t = uncorrected model−market skew (|t|≫2 ⇒ a line the calibrator must fix).
 • pool_t ≈ 0 everywhere = the global tilt does its job in-sample (sanity check).
 • wf_t   = does a γ fit on PAST matches still center the CURRENT bias? |wf_t| small & n_wf healthy
            ⇒ the skew is STABLE and the calibrator generalizes. Blown-up wf_t or tiny n_wf ⇒ the
            per-line bias drifts / not enough history — trust pooled less there.
 • g_mkt vs g_real: agree on 1X2/totals (market≈reality); on btts_yes they point OPPOSITE ways
   (g_mkt<0 pulls to market, g_real>0 pushes to realized). We ship g_mkt — the btts realized
   over-rate is left to the per-match deviation edge, not bet in the mean.
$("="^104)
""")

println("Done.  (Production calibrator = market-target γ; reality-target retained as diagnostic only.)")
