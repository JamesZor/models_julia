# current_development/betfair_closing_line/r02_clv_model_grid.jl
#
# Runner: Closing-Line-Value (CLV) of the M1/M2/M3 bigChance A/B grid against the
# Betfair last-traded closing line. Reuses the model-agnostic CLV engine in
# l01_clv_eval.jl unchanged — only the targets list is widened to the full
# model-emittable ∩ Betfair-liquid ∩ gradeable selection set.
#
# Grid (saved in ./data/bigchance_ab/):
#   M1  DP_Goals_Market_XG            {goals, market, xG}            (baseline)
#   M2  DP_Goals_Market_BigChance     {goals, market, bigChance}
#   M3  DP_Goals_Market_BigChance_XG  {goals, market, bigChance, xG}
#
# Prior verdict (LogLoss + 1X2 Kelly P&L, 281 matches, Bet365): M2 > M3 ≳ M1.
# This runner asks the more robust forward question: which model best *leads the
# Betfair close*? Run via the kaimon REPL on the server (PPD-only, no MCMC).
#
# Assumes `ds` (Data.Ireland) is already in scope. If not, uncomment the bootstrap.

using DataFrames
using Statistics
using Printf
using Plots
ENV["GKSwstype"] = "nul"     # headless: save figures without a display
gr()

include(joinpath(@__DIR__, "l01_clv_eval.jl"))

const PLOT_DIR = joinpath(@__DIR__, "plots_grid")
isdir(PLOT_DIR) || mkpath(PLOT_DIR)

# ---------------------------------------------------------------------------
# Optional bootstrap (skip if ds already loaded)
# ---------------------------------------------------------------------------
# using Revise, BayesianFootball
# const Data        = BayesianFootball.Data
# const Experiments = BayesianFootball.Experiments
# const Predictions = BayesianFootball.Predictions
# ds = Data.load_datastore_cached(Data.Ireland())

const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions

# Full model-emittable ∩ Betfair-liquid ∩ gradeable set (widened from the r01 five).
# 1X2 + BTTS(yes/no) + OverUnder 0.5…5.5 all join cleanly. Double-Chance is EXCLUDED:
# the model emits market "DoubleChance"/selections :DC_1X/:DC_X2/:DC_12 while Betfair
# uses "DOUBLE_CHANCE"/:dc_home_draw… — both names mismatch (no join) and coverage is
# thin (~437 matches). CorrectScore has no model probability → out of scope.
const TARGETS = [
    :home, :draw, :away,
    :btts_yes, :btts_no,
    :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
    :over_35, :under_35, :over_45, :under_45, :over_55, :under_55,
]

# ===========================================================================
# STEP 1 — Coverage triage of ds.betfair_odds (decides the scope of the study)
# ===========================================================================
println("\n" * "="^70)
println("STEP 1 — Betfair coverage triage (market × line × selection)")
println("="^70)

bf = ds.betfair_odds
cov_bf = combine(groupby(bf, [:market_name, :market_line, :selection]),
    :match_id => (x -> length(unique(x))) => :n_matches,
    nrow => :n_ticks,
    :minutes_to_kickoff => (m -> count(t -> -60 <= t <= 0, m)) => :ticks_near_close)
sort!(cov_bf, [:market_name, :market_line, :selection])
show(cov_bf; allrows = true, allcols = true); println()

#=
ds = Data.Ireland() betdb (:5433): 994 betfair matches, 1_051_125 ticks.
Markets present: 1X2, BTTS, CorrectScore, DOUBLE_CHANCE, OverUnder.
Liquid model-emittable selections (n_matches / ticks_near_close):
  1X2  home 940/14834  draw 939/11416  away 940/13783      → keep
  BTTS yes 928/4330    no   927/3313                        → keep
  O/U  0.5 879/3273   1.5 946/5117   2.5 965/7929  3.5 912/3803  → keep
  O/U  4.5 708/1831   5.5 419/736  (LOCF-heavy tails)        → keep but discount
  DOUBLE_CHANCE 433-440 matches, name-mismatch w/ model      → EXCLUDED
  CorrectScore  no model probability                         → EXCLUDED
=#

# ===========================================================================
# STEP 2 — Per-model inference + tagged CLV panels → grid_panel
# ===========================================================================
println("\n" * "="^70)
println("STEP 2 — Inferring M1/M2/M3 and building tagged CLV panels")
println("="^70)

files = Experiments.list_experiments("./data/bigchance_ab/"; data_dir = "")
idx(tag) = findfirst(f -> occursin(tag, f), files)
specs = [
    ("M1_xG",        idx("DP_Goals_Market_XG_")),
    ("M2_bigChance", idx("DP_Goals_Market_BigChance_2")),   # the _2026… avoids matching _XG
    ("M3_both",      idx("DP_Goals_Market_BigChance_XG_")),
]

function tagged_panel(name, k)
    res = Experiments.load_experiment(files, k)
    ppd = Predictions.model_inference(ds, res)              # DEFAULT_MARKET_CONFIG: all emittable markets
    panel = build_clv_panel(ppd, ds; targets = TARGETS, window_width = 60.0)
    panel.model_name .= name
    return (name = name, res = res, ppd = ppd, panel = panel)
end

runs = [tagged_panel(name, k) for (name, k) in specs]
grid_panel = reduce(vcat, [r.panel for r in runs])

@printf("grid_panel rows: %d  | per-model matches: %s\n", nrow(grid_panel),
    join(["$(r.name)=$(length(unique(r.panel.match_id)))" for r in runs], ", "))

#=
grid_panel rows: 78933  (26311 each)  | per-model matches: M1=275, M2=275, M3=275
Selections that joined+graded: away,btts_no,btts_yes,draw,home,
  over_05,over_15,over_25,over_35,over_45,over_55,under_05,under_15,under_25,under_35,under_45,under_55
=#

# Coverage (LOCF watch). 4.5/5.5 lines are LOCF-heavy near the close — discount them.
cov = coverage_table(grid_panel[grid_panel.model_name .== "M1_xG", :])
println("\n--- Coverage at τ=-5 (n_matches / LOCF-frac / mean-ticks) ---")
show(sort(filter(:horizon => ==(-5.0), cov), :selection); allrows = true, allcols = true); println()

#=
 over_55/under_55 locf_frac 0.36/0.49 ; over_45/under_45 0.26/0.21 — tails are LOCF noise.
 Core 1X2/BTTS/O-U 0.5–3.5 are well covered (locf_frac < 0.13 at close).
=#

# ===========================================================================
# STAGE 1 — Edge: model vs market log-loss / Brier (held-out Betfair fair line)
# ===========================================================================
println("\n" * "="^70)
println("STAGE 1 — Edge by model (does any model beat the Betfair line?)")
println("="^70)

edge_pooled_model  = sort(edge_by_horizon(grid_panel; group = [:model_name]), :diff_ll)
edge_model_horizon = edge_by_horizon(grid_panel; group = [:model_name, :horizon])
show(edge_pooled_model; allrows = true, allcols = true); println()

#=
3×10 (pooled over horizons & 17 selections) — most-negative diff_ll = best:
 model_name    model_ll  market_ll  diff_ll     diff_ll_lo  diff_ll_hi  diff_brier
 M1_xG         0.54714   0.60534   -0.058195   -0.075595   -0.042006    2.16e-5
 M3_both       0.54750   0.60534   -0.057837   -0.075898   -0.042149    2.01e-4
 M2_bigChance  0.54860   0.60534   -0.056744   -0.074574   -0.040133    5.98e-4
All three BEAT the Betfair line on log-loss (CI excludes 0); ordering M1 < M3 < M2.
CAVEAT: diff_brier ≈ 0 — the LL win is concentrated in extreme-prob tails (vig-removal
artifact on thin O/U 0.5 / 4.5 / 5.5), not core probability accuracy. Read Brier as the tie.
=#

# ===========================================================================
# STAGE 2 (HEADLINE) — CLV alpha: which model leads the close?
# ===========================================================================
println("\n" * "="^70)
println("STAGE 2 — CLV alpha by model (HEADLINE)")
println("="^70)

clv_pooled_model   = clv_alpha(grid_panel; group = [:model_name])
clv_model_horizon  = clv_alpha(grid_panel; group = [:model_name, :horizon])
clv_model_sel      = clv_alpha(grid_panel; group = [:model_name, :selection])
show(clv_pooled_model; allrows = true, allcols = true); println()
show(sort(clv_model_horizon, [:horizon, :model_name]); allrows = true, allcols = true); println()

#=
POOLED by model (n=26311 each):  (mean_clv is model-independent by construction)
 model_name    beta       beta_p        hit_rate  hit_p
 M1_xG         0.056336   1.7e-203      0.518528  1.9e-9     ← best
 M3_both       0.054344   1.3e-195      0.516552  8.1e-8
 M2_bigChance  0.047187   1.1e-142      0.509673  1.7e-3     ← worst
Ordering M1 ≳ M3 > M2 — FLIPS the prior LogLoss/P&L verdict (M2 > M3 ≳ M1).

by model × horizon: M1 ≳ M3 > M2 holds at EVERY horizon −1440…−45 (M3 edges M1 only
at −1440). β peaks at τ=−720 (M1 0.114, M2 0.101, M3 0.112; hit ~0.56) and decays to
~0 by the close. At τ=−5 hit-rate dips below 0.5 (model & close have converged — the
residual move is mean-reverting noise, not signal). The tradeable CLV lives at 12h–1.5h.
=#

# Which markets carry each model's CLV? (pooled over horizons)
beta_wide = sort(unstack(select(clv_model_sel, :selection, :model_name, :beta),
    :selection, :model_name, :beta), :selection)
println("\n--- CLV β by selection × model (M2 is weakest on totals/BTTS) ---")
show(beta_wide; allrows = true, allcols = true); println()

#=
Liquid core (ignore over_05/under_45/under_55 — extreme-prob scale artifacts, hit<0.5):
 selection  M1_xG     M2_bigChance  M3_both
 btts_no    0.0517    0.0404        0.0469
 btts_yes   0.0517    0.0406        0.0473
 draw       0.0315    0.0145        0.0341
 over_15    0.0244    0.0043        0.0257    ← M2 ≈ 0
 over_25    0.0367    0.0222        0.0376
 over_35    0.0200    0.0112        0.0211
M2 (bigChance) is WEAKEST exactly on totals (O/U 1.5–3.5) and BTTS — the markets where
bigChance was hypothesised to add edge. M1 and M3 are ~tied; adding bigChance never helps.
=#

# ===========================================================================
# STAGE 3 — Entry-timing P&L (sanity; Stage 2 is the verdict)
# ===========================================================================
println("\n" * "="^70)
println("STAGE 3 — Entry-timing filtered P&L by model")
println("="^70)

for thr in (0.0, 0.02, 0.05)
    pnl = entry_timing_pnl(grid_panel; edge_threshold = thr, group = [:model_name, :horizon])
    pooled = combine(groupby(pnl, :model_name), :n_bets => sum => :total_bets, :roi => mean => :mean_roi)
    @printf("\n--- edge_threshold = %.2f (pooled over horizons) ---\n", thr)
    show(sort(pooled, :mean_roi, rev = true); allrows = true, allcols = true); println()
end

#=
edge_threshold = 0.02 (pooled):  same ordering as CLV
 model_name    total_bets  mean_roi
 M1_xG         8791        0.1305     ← best
 M3_both       8820        0.1268
 M2_bigChance  8691        0.0946     ← worst
Even the noisy flat-stake ROI reproduces M1 ≳ M3 > M2 on the Betfair held-out panel.
=#

# ===========================================================================
# STAGE 4 — Roll spread (microstructure) + per-model PIT calibration
# ===========================================================================
println("\n" * "="^70)
println("STAGE 4 — Roll spread + PIT vs closing line")
println("="^70)

roll = roll_spread(ds.betfair_odds; targets = TARGETS)
show(sort(roll, :selection); allrows = true, allcols = true); println()

pit_summary = DataFrame(model = String[], n = Int[], ks_d = Float64[], ks_p = Float64[])
for r in runs
    pit = pit_calibration(r.ppd, ds; targets = TARGETS)
    push!(pit_summary, (r.name, pit.n, pit.ks_d, pit.ks_p))
    plot_pit(pit.pit; save_path = joinpath(PLOT_DIR, "pit_$(r.name).png"))
end
show(pit_summary; allrows = true, allcols = true); println()

#=
PIT vs Betfair close — all three diverge significantly (KS p ≈ 0):
 model         ks_d      ks_p
 M1_xG         0.1657    1.3e-99
 M2_bigChance  0.1571    1.6e-89    ← marginally closest to the close
 M3_both       0.1660    5.7e-100
Non-uniform PIT = the model takes systematic positions away from the close — that
divergence is the SOURCE of the CLV, largest for M1/M3 (consistent with their higher β).
=#

# ===========================================================================
# Plots + headline
# ===========================================================================
cmp = sort(clv_model_horizon, [:model_name, :horizon])
ph = plot(title = "CLV alpha by model: does the model lead the Betfair close?",
    xlabel = "Minutes to kickoff", ylabel = "β (line move per unit model signal)",
    legend = :topright, titlefontsize = 10, size = (900, 480))
for (g, c) in zip(groupby(cmp, :model_name), [:steelblue, :darkorange, :seagreen])
    plot!(ph, g.horizon, g.beta; ribbon = 1.96 .* g.beta_se, marker = :circle, lw = 2,
        label = g.model_name[1], color = c)
end
hline!(ph, [0.0]; color = :grey, ls = :dash, label = "")
savefig(ph, joinpath(PLOT_DIR, "headline_clv_beta_by_model.png"))

ph2 = plot(title = "Directional hit-rate by model", xlabel = "Minutes to kickoff",
    ylabel = "hit-rate (sign signal == sign move)", legend = :topright,
    titlefontsize = 10, size = (900, 480))
for (g, c) in zip(groupby(cmp, :model_name), [:steelblue, :darkorange, :seagreen])
    plot!(ph2, g.horizon, g.hit_rate; marker = :circle, lw = 2, label = g.model_name[1], color = c)
end
hline!(ph2, [0.5]; color = :grey, ls = :dash, label = "coin-flip")
savefig(ph2, joinpath(PLOT_DIR, "headline_hitrate_by_model.png"))

for r in runs
    p = r.panel
    plot_edge_decay(edge_by_horizon(p; group = [:horizon]); save_path = joinpath(PLOT_DIR, "edge_$(r.name).png"))
    plot_clv(clv_alpha(p; group = [:horizon]); save_path = joinpath(PLOT_DIR, "clv_$(r.name).png"))
    plot_pnl(entry_timing_pnl(p; edge_threshold = 0.02, group = [:horizon]); save_path = joinpath(PLOT_DIR, "pnl_$(r.name).png"))
end

println("\n" * "="^70)
println("HEADLINE")
println("="^70)
best = clv_pooled_model[argmax(clv_pooled_model.beta), :]
@printf("• Best CLV: %s (pooled β = %.4f, hit-rate = %.3f)\n", best.model_name, best.beta, best.hit_rate)
println("• Ordering M1 ≳ M3 > M2 on CLV, log-loss AND filtered ROI — FLIPS the prior")
println("  LogLoss/1X2-P&L verdict (M2 > M3 ≳ M1). bigChance does not lead the close.")
println("• Plots saved to: $(PLOT_DIR)")
