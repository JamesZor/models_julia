# current_development/betfair_closing_line/r01_clv_eval.jl
#
# Runner: edge / entry-timing EDA — Betfair price path vs Bayesian posterior.
# Assumes `ds`, `results_model` (and ideally `ppd`) are already in scope from
# r00_basic_runner.jl. If not, uncomment the bootstrap block below.

using DataFrames
using Statistics
using Printf
using Plots
ENV["GKSwstype"] = "nul"     # headless: save figures without a display
gr()

include(joinpath(@__DIR__, "l01_clv_eval.jl"))

const PLOT_DIR = joinpath(@__DIR__, "plots")
isdir(PLOT_DIR) || mkpath(PLOT_DIR)

# ---------------------------------------------------------------------------
# Optional bootstrap (skip if ds / results_model / ppd already loaded by r00)
# ---------------------------------------------------------------------------
# using Revise, BayesianFootball, ThreadPinning
# const Data        = BayesianFootball.Data
# const Experiments = BayesianFootball.Experiments
# const Predictions = BayesianFootball.Predictions
# ds            = Data.load_datastore_cached(Data.Ireland())
# saved_files   = Experiments.list_experiments("./data/dixon_coles_halflife_grid/"; data_dir="")
# results_model = Experiments.load_experiment(saved_files, 2)   # DCMH_HalfLife_60

if !@isdefined(ppd)
    ppd = BayesianFootball.Predictions.model_inference(ds, results_model)
end

# ===========================================================================
# STAGE 0 — Build the horizon-resolved panel + coverage
# ===========================================================================
println("\n" * "="^70)
println("STAGE 0 — Building horizon-resolved CLV panel")
println("="^70)

panel = build_clv_panel(ppd, ds; window_width = 60.0)
@printf("Panel rows: %d  | matches: %d  | selections: %s\n",
    nrow(panel), length(unique(panel.match_id)), join(string.(unique(panel.selection)), ", "))

cov = coverage_table(panel)
println("\n--- Coverage (matches / LOCF-fraction / mean-ticks per selection × horizon) ---")
show(cov; allrows = true, allcols = true); println()

# ===========================================================================
# STAGE 1 — Edge decay over time (HEADLINE)
# ===========================================================================
println("\n" * "="^70)
println("STAGE 1 — Edge decay (model vs market log-loss / Brier by horizon)")
println("="^70)

edge_pooled = edge_by_horizon(panel; group = [:horizon])
println("\n--- Pooled across the five target selections ---")
show(edge_pooled; allrows = true, allcols = true); println()

edge_by_sel = edge_by_horizon(panel; group = [:selection, :horizon])
println("\n--- Per selection × horizon ---")
show(edge_by_sel; allrows = true, allcols = true); println()

plot_edge_decay(edge_pooled; save_path = joinpath(PLOT_DIR, "stage1_edge_decay.png"))

# Sanity cross-check against the existing LogLoss metric (~ -0.038 for HalfLife_60).
near_close = filter(:horizon => ==(-5.0), edge_pooled)
isempty(near_close) || @printf("\n[sanity] pooled diff_ll at τ=-5: %.4f (cf. closing LogLoss metric ≈ -0.038)\n",
    near_close.diff_ll[1])

# ===========================================================================
# STAGE 2 — Does the model anticipate the closing-line move? (CLV alpha)
# ===========================================================================
println("\n" * "="^70)
println("STAGE 2 — CLV alpha: does model_signal predict the line move?")
println("="^70)

alpha_pooled = clv_alpha(panel; group = [:horizon])
println("\n--- Pooled ---")
show(alpha_pooled; allrows = true, allcols = true); println()

alpha_by_sel = clv_alpha(panel; group = [:selection, :horizon])
println("\n--- Per selection × horizon ---")
show(alpha_by_sel; allrows = true, allcols = true); println()

plot_clv(alpha_pooled; save_path = joinpath(PLOT_DIR, "stage2_clv_alpha.png"))

# ===========================================================================
# STAGE 3 — CLV -> realised P&L by entry horizon
# ===========================================================================
println("\n" * "="^70)
println("STAGE 3 — Entry-timing P&L (filtered bets by horizon)")
println("="^70)

for thr in (0.0, 0.02, 0.05)
    pnl = entry_timing_pnl(panel; edge_threshold = thr, group = [:horizon])
    @printf("\n--- edge_threshold = %.2f ---\n", thr)
    show(pnl; allrows = true, allcols = true); println()
    thr == 0.02 && plot_pnl(pnl; save_path = joinpath(PLOT_DIR, "stage3_pnl_thr0.02.png"))
end

# ===========================================================================
# STAGE 4 — Microstructure rigour + PIT calibration check
# ===========================================================================
println("\n" * "="^70)
println("STAGE 4 — Roll effective spread + PIT calibration")
println("="^70)

roll = roll_spread(ds.betfair_odds)
println("\n--- Roll (1984) effective spread per selection (decimal-odds units) ---")
show(roll; allrows = true, allcols = true); println()

pit = pit_calibration(ppd, ds)
@printf("\nPIT vs closing line: n=%d  KS D=%.4f  p=%.4f\n", pit.n, pit.ks_d, pit.ks_p)
println("--- Central credible-interval coverage (empirical vs nominal) ---")
show(pit.coverage; allrows = true, allcols = true); println()
plot_pit(pit.pit; save_path = joinpath(PLOT_DIR, "stage4_pit.png"))

# ===========================================================================
# Headline summary
# ===========================================================================
println("\n" * "="^70)
println("HEADLINE")
println("="^70)
best_edge  = edge_pooled[argmin(edge_pooled.diff_ll), :]
best_pnl   = let p = entry_timing_pnl(panel; edge_threshold = 0.02); isempty(p) ? nothing : p[argmax(p.roi), :] end
@printf("• Largest model edge (most-negative diff_ll) at τ = %.0f min  (diff_ll = %.4f)\n",
    best_edge.horizon, best_edge.diff_ll)
isnothing(best_pnl) || @printf("• Best ROI entry (edge>0.02) at τ = %.0f min  (ROI = %.3f, n=%d)\n",
    best_pnl.horizon, best_pnl.roi, best_pnl.n_bets)
println("• Plots saved to: $(PLOT_DIR)")
