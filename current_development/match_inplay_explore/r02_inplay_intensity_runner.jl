#=
r02_inplay_intensity_runner.jl  —  Fit & evaluate the parametric in-play intensity model.

Pipeline:
  l01 panel (market-implied λ_rem + game state)  ->  l02 long-format intensity dataset
    ->  Poisson GLM  λ_inst(t, state, pregame_λ)  ->  held-out evaluation vs baselines.

Headline (Ireland, held-out by match): the model — using ONLY pregame λ + game state + time, no
live odds — beats the pregame-only baseline AND the market's own λ_rem at predicting realized
remaining goals. Score effects: trailing +0.25 / leading −0.24 (log-rate), both highly significant.

Run with threads:  julia --project -t 32   (then pinthreads(:cores))
=#

using Revise
using BayesianFootball
using DataFrames
using Statistics
using ThreadPinning
using ProgressMeter

pinthreads(:cores)

const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

# ==========================================================================
# 1. DATA + PRE-GAME LATENTS + PANEL  (reuse r01 machinery)
# ==========================================================================
include("l01_inplay_inverse.jl")
include("l02_inplay_intensity.jl")

println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())
bf = ds.betfair_odds

saved_files      = Experiments.list_experiments("./data/dixon_coles_ab/", data_dir = "")
res_pre_game     = Experiments.load_experiment(saved_files, 1)
pre_game_latents = Experiments.extract_oos_predictions(ds, res_pre_game)
pg_tbl = DataFrame(match_id = Int.(pre_game_latents.df.match_id),
                   pg_λ_h   = [mean(Float64.(v)) for v in pre_game_latents.df.λ_h],
                   pg_λ_a   = [mean(Float64.(v)) for v in pre_game_latents.df.λ_a])

# Threaded panel build (one market-implied λ_rem trace per match).
function build_panel(bf, ds, pg_tbl; config = Features.DoublePoissonMarketFeature(),
                     bin_minutes = 5.0, staleness = 10.0, min_sel = 6, mtk_max = 130.0)
    ids = unique(subset(bf, :minutes_to_kickoff => ByRow(x -> 0 < x <= mtk_max)).match_id)
    parts = Vector{DataFrame}(undef, length(ids))
    Threads.@threads for k in eachindex(ids)
        local tr
        try
            tr = inplay_lambda_trace(bf, ds, Int(ids[k]), config;
                                     bin_minutes = bin_minutes, staleness = staleness,
                                     min_sel = min_sel, mtk_max = mtk_max)
        catch
            tr = DataFrame()
        end
        parts[k] = tr
    end
    panel = vcat([df for df in parts if nrow(df) > 0]...)
    return leftjoin(panel, pg_tbl, on = :match_id)
end

panel = build_panel(bf, ds, pg_tbl; bin_minutes = 5.0)

# ==========================================================================
# 2. FIT THE INTENSITY MODEL
# ==========================================================================
D = build_intensity_dataset(panel, ds)
println("[INFO] intensity dataset: $(nrow(D)) side-bins over $(length(unique(D.match_id))) matches")

Dtr, Dte = split_by_match(D; frac = 0.75)
model = fit_intensity_model(Dtr)
println("\n[INTENSITY MODEL COEFFICIENTS]"); show(coeftable(model)); println()

# ==========================================================================
# 3. HELD-OUT EVALUATION
# ==========================================================================
y          = Float64.(Dte.rem_goals)
pred_model = predict_intensity(model, Dte)                # state-aware model (no live odds)
pred_pre   = exp.(Dte.log_pregame) .* Dte.rem_frac        # baseline (a): pregame-only, no state
pred_mkt   = clamp.(Dte.mkt_lam, 0, 6)                     # baseline (b): market λ_rem (degenerate-capped)

println("\n[HELD-OUT mean Poisson log-score (higher is better)]")
println("  model (pregame+state+time) : $(round(mean_logscore(y, pred_model), digits=4))")
println("  pregame-only (no state)    : $(round(mean_logscore(y, pred_pre),   digits=4))")
println("  market λ_rem               : $(round(mean_logscore(y, pred_mkt),   digits=4))")

# Where the state effect should bite: non-level bins only.
nonlevel = (Dte.trailing .> 0) .| (Dte.leading .> 0)
println("\n[NON-LEVEL bins only, n=$(sum(nonlevel))]")
println("  model        : $(round(mean_logscore(y[nonlevel], pred_model[nonlevel]), digits=4))")
println("  pregame-only : $(round(mean_logscore(y[nonlevel], pred_pre[nonlevel]),   digits=4))")

# Calibration: predicted vs actual remaining goals by decile of the model prediction.
Dte_eval = copy(Dte)
Dte_eval.pred = pred_model
Dte_eval.decile = min.(9, floor.(Int, 10 .* (sortperm(sortperm(pred_model)) .- 1) ./ length(pred_model)))
calib = combine(groupby(Dte_eval, :decile),
    nrow => :n,
    :pred      => (x -> round(mean(x), digits = 3)) => :pred_mean,
    :rem_goals => (x -> round(mean(x), digits = 3)) => :actual_mean)
sort!(calib, :decile)
println("\n[CALIBRATION (held-out, by model-prediction decile)]"); show(calib, allrows = true); println()

#=
# ==========================================================================
# 4. PLOTS (GLMakie — on the kaimon server call ex(...; mt=true))
# ==========================================================================
using GLMakie

# calibration scatter
function plot_calibration(calib)
    fig = Figure(size = (500, 500))
    ax = Axis(fig[1, 1]; xlabel = "predicted remaining goals", ylabel = "actual",
              title = "In-play intensity model — held-out calibration")
    scatter!(ax, calib.pred_mean, calib.actual_mean)
    lines!(ax, [0, maximum(calib.actual_mean)], [0, maximum(calib.actual_mean)]; color = :grey)
    fig
end
=#
