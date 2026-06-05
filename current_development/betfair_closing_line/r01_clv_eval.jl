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


#=
Panel rows: 8687  | matches: 258  | selections: over_25, under_25, under_15, btts_yes, under_35
=#

cov = coverage_table(panel)

#=
40×5 DataFrame
 Row │ selection  horizon  n_matches  locf_frac  mean_ticks 
     │ Symbol     Float64  Int64      Float64    Float64    
─────┼──────────────────────────────────────────────────────
   1 │ btts_yes   -1440.0         56  0.875        0.196429
   2 │ btts_yes    -720.0        142  0.640845     0.528169
   3 │ btts_yes    -360.0        208  0.5625       0.759615
   4 │ btts_yes    -180.0        228  0.486842     0.903509
   5 │ btts_yes     -90.0        239  0.460251     1.03347
   6 │ btts_yes     -45.0        250  0.276        1.956
   7 │ btts_yes     -20.0        251  0.135458     3.08765
   8 │ btts_yes      -5.0        253  0.055336     4.32016
   9 │ over_25    -1440.0        109  0.899083     0.110092
  10 │ over_25     -720.0        212  0.514151     0.75
  11 │ over_25     -360.0        256  0.492188     1.04688
  12 │ over_25     -180.0        257  0.420233     1.14397
  13 │ over_25      -90.0        257  0.354086     1.32685
  14 │ over_25      -45.0        257  0.120623     2.85992
  15 │ over_25      -20.0        257  0.0350195    4.98833
  16 │ over_25       -5.0        257  0.0116732    6.74708
  17 │ under_15   -1440.0         50  0.92         0.08
  18 │ under_15    -720.0        179  0.625698     0.553073
  19 │ under_15    -360.0        246  0.686992     0.5
  20 │ under_15    -180.0        250  0.6          0.608
  21 │ under_15     -90.0        254  0.645669     0.57874
  22 │ under_15     -45.0        255  0.317647     1.61176
  23 │ under_15     -20.0        256  0.125        2.88672
  24 │ under_15      -5.0        256  0.0703125    3.79688
  25 │ under_25   -1440.0        109  0.889908     0.12844
  26 │ under_25    -720.0        212  0.627358     0.660377
  27 │ under_25    -360.0        256  0.585938     0.785156
  28 │ under_25    -180.0        257  0.48249      0.922179
  29 │ under_25     -90.0        257  0.490272     0.984436
  30 │ under_25     -45.0        257  0.171206     2.41245
  31 │ under_25     -20.0        257  0.0428016    4.22179
  32 │ under_25      -5.0        257  0.0116732    5.84825
  33 │ under_35   -1440.0         14  0.857143     0.142857
  34 │ under_35    -720.0        124  0.637097     0.548387
  35 │ under_35    -360.0        205  0.736585     0.370732
  36 │ under_35    -180.0        236  0.699153     0.427966
  37 │ under_35     -90.0        247  0.643725     0.48583
  38 │ under_35     -45.0        252  0.297619     1.5
  39 │ under_35     -20.0        256  0.109375     2.62109
  40 │ under_35      -5.0        256  0.046875     3.85156
=#

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


#=
8×10 DataFrame
 Row │ horizon  n      model_ll  market_ll  diff_ll      diff_ll_lo  diff_ll_hi    model_brier  market_brier  diff_brier  
     │ Float64  Int64  Float64   Float64    Float64      Float64     Float64       Float64      Float64       Float64     
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ -1440.0    338  0.684351   0.693506  -0.00915454  -0.0226744   0.00485647      0.245477      0.249882  -0.00440514
   2 │  -720.0    869  0.654846   0.661203  -0.0063578   -0.0153208   0.00241989      0.23167       0.234825  -0.00315485
   3 │  -360.0   1171  0.64809    0.656306  -0.0082164   -0.0154523  -0.000735779     0.228548      0.232339  -0.0037912
   4 │  -180.0   1228  0.645896   0.655085  -0.00918932  -0.0166018  -0.00260429      0.227461      0.231604  -0.00414355
   5 │   -90.0   1254  0.644751   0.65363   -0.00887861  -0.0160207  -0.00183408      0.226868      0.230884  -0.00401565
   6 │   -45.0   1271  0.645331   0.654207  -0.0088767   -0.0157887  -0.0018257       0.227117      0.231121  -0.0040039
   7 │   -20.0   1277  0.644764   0.652584  -0.00781965  -0.0145789  -0.000676636     0.226856      0.230418  -0.00356146
   8 │    -5.0   1279  0.644782   0.652123  -0.00734097  -0.0142888  -0.000438393     0.226864      0.230215  -0.00335128
=#


plot_edge_decay(edge_pooled; save_path = joinpath(PLOT_DIR, "stage1_edge_decay.png"));

# Sanity cross-check against the existing LogLoss metric (~ -0.038 for HalfLife_60).
near_close = filter(:horizon => ==(-5.0), edge_pooled)
isempty(near_close) || @printf("\n[sanity] pooled diff_ll at τ=-5: %.4f (cf. closing LogLoss metric ≈ -0.038)\n",
    near_close.diff_ll[1])


#=
[sanity] pooled diff_ll at τ=-5: -0.0073 (cf. closing LogLoss metric ≈ -0.038)
=#


# ===========================================================================
# STAGE 2 — Does the model anticipate the closing-line move? (CLV alpha)
# ===========================================================================
println("\n" * "="^70)
println("STAGE 2 — CLV alpha: does model_signal predict the line move?")
println("="^70)

alpha_pooled = clv_alpha(panel; group = [:horizon])
println("\n--- Pooled ---")
show(alpha_pooled; allrows = true, allcols = true); println()

#=
julia> alpha_pooled = clv_alpha(panel; group = [:horizon])
8×9 DataFrame
 Row │ horizon  n      beta        beta_se     beta_p       hit_rate  hit_n  hit_p        mean_clv     
     │ Float64  Int64  Float64     Float64     Float64      Float64   Int64  Float64      Float64      
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ -1440.0    338  0.0258008   0.0274861   0.348568     0.485207    338  0.624528     -0.00277452
   2 │  -720.0    869  0.096013    0.0150043   2.55639e-10  0.535098    869  0.0417529    -0.00440158
   3 │  -360.0   1171  0.045714    0.0101815   7.83052e-6   0.542272   1171  0.00416609   -0.0040566
   4 │  -180.0   1228  0.0321332   0.00857523  0.000187128  0.544788   1228  0.00185626   -0.00202378
   5 │   -90.0   1254  0.0328728   0.00785354  3.04108e-5   0.549442   1254  0.000508999  -0.00159483
   6 │   -45.0   1271  0.0257814   0.00608425  2.42466e-5   0.561762   1271  1.18148e-5   -0.00085308
   7 │   -20.0   1277  0.0136256   0.00438393  0.00192465   0.521535   1277  0.130726      0.000692393
   8 │    -5.0   1279  0.00334941  0.00298317  0.261747     0.505864   1279  0.69547       0.000690664
=#


alpha_by_sel = clv_alpha(panel; group = [:selection, :horizon])
println("\n--- Per selection × horizon ---")
show(alpha_by_sel; allrows = true, allcols = true); println()

#=
julia> show(alpha_by_sel; allrows = true, allcols = true); println()
40×10 DataFrame
 Row │ selection  horizon  n      beta          beta_se     beta_p       hit_rate  hit_n  hit_p      mean_clv     
     │ Symbol     Float64  Int64  Float64       Float64     Float64      Float64   Int64  Float64    Float64      
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ btts_yes   -1440.0     56  -0.0460886    0.0849886   0.589849     0.375        56  0.0814268  -0.0230417
   2 │ btts_yes    -720.0    142   0.126124     0.0473138   0.00858655   0.478873    142  0.674935   -0.00399952
   3 │ btts_yes    -360.0    208   0.0729179    0.0278287   0.00944119   0.548077    208  0.187571   -0.00136199
   4 │ btts_yes    -180.0    228   0.0672709    0.0224028   0.00297608   0.548246    228  0.164155    0.000830155
   5 │ btts_yes     -90.0    239   0.0496053    0.0192791   0.0106917    0.556485    239  0.0923994   0.0010756
   6 │ btts_yes     -45.0    250   0.0480188    0.0163499   0.00362633   0.576       250  0.0190985   0.0018694
   7 │ btts_yes     -20.0    251   0.0263657    0.0119187   0.0278661    0.553785    251  0.100584    0.00222667
   8 │ btts_yes      -5.0    253   0.00973845   0.00731555  0.184331     0.513834    253  0.706092    0.00184791
   9 │ over_25    -1440.0    109   0.0438425    0.0456383   0.338893     0.53211     109  0.5657     -0.0169367
  10 │ over_25     -720.0    212   0.10319      0.0297846   0.000643746  0.54717     212  0.191796    0.000831107
  11 │ over_25     -360.0    256   0.04286      0.0213029   0.0452828    0.535156    256  0.287989    0.00205054
  12 │ over_25     -180.0    257   0.0290815    0.0179768   0.106958     0.568093    257  0.0337304   0.00279638
  13 │ over_25      -90.0    257   0.0321409    0.0169052   0.0583981    0.55642     257  0.0805063  -0.0010057
  14 │ over_25      -45.0    257   0.0234722    0.0126459   0.064592     0.579767    257  0.0124406   0.000716092
  15 │ over_25      -20.0    257   0.0136247    0.00918638  0.139272     0.509728    257  0.803023    0.00134376
  16 │ over_25       -5.0    257   0.00529088   0.00626476  0.399155     0.51751     257  0.617847    0.0015531
  17 │ under_15   -1440.0     50  -0.0323853    0.0672582   0.632344     0.36         50  0.0649086   0.0211868
  18 │ under_15    -720.0    179   0.0790226    0.0331793   0.0182945    0.536313    179  0.369806   -0.0127986
  19 │ under_15    -360.0    246   0.0474386    0.0223957   0.0351689    0.54065     246  0.225671   -0.0161115
  20 │ under_15    -180.0    250   0.023207     0.0194063   0.232898     0.496       250  0.949588   -0.0107227
  21 │ under_15     -90.0    254   0.023676     0.0179893   0.189332     0.53937     254  0.233132   -0.0081078
  22 │ under_15     -45.0    255   0.0235286    0.0135755   0.0842846    0.54902     255  0.132696   -0.00682128
  23 │ under_15     -20.0    256   0.0138474    0.0100523   0.169562     0.519531    256  0.573861   -0.00268295
  24 │ under_15      -5.0    256   0.000172762  0.00712436  0.980673     0.507812    256  0.851315   -0.00246297
  25 │ under_25   -1440.0    109   0.043842     0.0456383   0.338899     0.53211     109  0.5657      0.0107141
  26 │ under_25    -720.0    212   0.103188     0.0297844   0.000643846  0.54717     212  0.191796   -0.002501
  27 │ under_25    -360.0    256   0.0435569    0.0213077   0.0419667    0.542969    256  0.189248   -0.00223263
  28 │ under_25    -180.0    257   0.0298397    0.0179786   0.0981982    0.571984    257  0.0245392  -0.000790734
  29 │ under_25     -90.0    257   0.0326708    0.0169115   0.0544843    0.560311    257  0.0610861   0.00103474
  30 │ under_25     -45.0    257   0.0237336    0.012655    0.0618754    0.579767    257  0.0124406   8.62282e-5
  31 │ under_25     -20.0    257   0.0138637    0.00919294  0.132772     0.509728    257  0.803023    0.00103695
  32 │ under_25      -5.0    257   0.00529627   0.00627156  0.399187     0.51751     257  0.617847    0.00093536
  33 │ under_35   -1440.0     14   0.100762     0.14225     0.492263     0.642857     14  0.42395    -0.00203813
  34 │ under_35    -720.0    124   0.0646874    0.0374307   0.0864842    0.556452    124  0.242923   -0.00493609
  35 │ under_35    -360.0    205   0.0223175    0.0239288   0.352102     0.546341    205  0.208584   -0.00222894
  36 │ under_35    -180.0    236   0.0133126    0.0204028   0.514728     0.538136    236  0.268427   -0.00215779
  37 │ under_35     -90.0    247   0.0317398    0.0180288   0.0795695    0.534413    247  0.308646   -0.000830235
  38 │ under_35     -45.0    252   0.0145172    0.0142901   0.310664     0.52381     252  0.488426   -7.29572e-5
  39 │ under_35     -20.0    256   0.00234563   0.00963108  0.807778     0.515625    256  0.661833    0.00156361
  40 │ under_35      -5.0    256  -0.00528698   0.00690144  0.444346     0.472656    256  0.416557    0.00158916
=#


plot_clv(alpha_pooled; save_path = joinpath(PLOT_DIR, "stage2_clv_alpha.png"));

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

#=
--- edge_threshold = 0.00 ---
8×6 DataFrame
 Row │ horizon  n_bets  roi       mean_log_growth  hit       avg_odds 
     │ Float64  Int64   Float64   Float64          Float64   Float64  
─────┼────────────────────────────────────────────────────────────────
   1 │ -1440.0     168  0.29131          -2.67677  0.547619   2.3931
   2 │  -720.0     433  0.247255         -2.71979  0.545035   2.43477
   3 │  -360.0     569  0.177732         -2.84562  0.530756   2.41446
   4 │  -180.0     603  0.163351         -2.87697  0.527363   2.41628
   5 │   -90.0     617  0.188566         -2.73997  0.546191   2.39736
   6 │   -45.0     618  0.173979         -2.80579  0.537217   2.41095
   7 │   -20.0     633  0.173074         -2.79584  0.538705   2.41108
   8 │    -5.0     629  0.169685         -2.80666  0.537361   2.40689

--- edge_threshold = 0.02 ---
8×6 DataFrame
 Row │ horizon  n_bets  roi       mean_log_growth  hit       avg_odds 
     │ Float64  Int64   Float64   Float64          Float64   Float64  
─────┼────────────────────────────────────────────────────────────────
   1 │ -1440.0     128  0.281797         -2.73796  0.539062   2.43496
   2 │  -720.0     322  0.283759         -2.67211  0.549689   2.47022
   3 │  -360.0     424  0.261155         -2.63582  0.556604   2.44702
   4 │  -180.0     439  0.26983          -2.60759  0.560364   2.44227
   5 │   -90.0     461  0.268231         -2.5984   0.561822   2.44612
   6 │   -45.0     463  0.259169         -2.61995  0.559395   2.45755
   7 │   -20.0     475  0.23904          -2.65306  0.555789   2.44767
   8 │    -5.0     475  0.227374         -2.71471  0.547368   2.45189

--- edge_threshold = 0.05 ---
8×6 DataFrame
 Row │ horizon  n_bets  roi       mean_log_growth  hit       avg_odds 
     │ Float64  Int64   Float64   Float64          Float64   Float64  
─────┼────────────────────────────────────────────────────────────────
   1 │ -1440.0      71  0.392958         -2.52664  0.56338    2.50634
   2 │  -720.0     184  0.311988         -2.69945  0.543478   2.59432
   3 │  -360.0     229  0.349833         -2.46095  0.576419   2.51042
   4 │  -180.0     237  0.398143         -2.3741   0.586498   2.52135
   5 │   -90.0     244  0.380749         -2.41262  0.581967   2.51032
   6 │   -45.0     251  0.37757          -2.44269  0.577689   2.54114
   7 │   -20.0     252  0.307044         -2.59394  0.559524   2.55283
   8 │    -5.0     246  0.285885         -2.70258  0.544715   2.56668
=#


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
