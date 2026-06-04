# current_development/order_book/r03_entry_timing_eda.jl
# Entry-timing EDA for OverUnder 2.5 markets
# Assumes ob DataFrame is in scope (from r01). Re-include loaders if needed.

using Plots, Printf, Statistics, DataFrames
plotly() # Switch to interactive HTML backend

# Prevents plotting engine from trying to open a local window
ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "l01_order_book.jl"))
include(joinpath(@__DIR__, "l02_ob_plots.jl"))
include(joinpath(@__DIR__, "l03_ob_features.jl"))

PLOTS_DIR = joinpath(@__DIR__, "plots")
mkpath(PLOTS_DIR) # Ensure directory exists before saving
@async serve(dir=PLOTS_DIR, port=8080)
# ---------------------------------------------------------------------------
# 1. Build feature DataFrame
# ---------------------------------------------------------------------------

obf = add_ob_features(ob)
add_entry_criteria!(obf;
    spread_thresh   = 2.0,    # % — adjust to your cost tolerance
    depth_thresh    = 500.0,  # £ total depth at L1–L3
    obi_stab_thresh = 0.15,
    obi_max         = 0.4,
)

println("Features added. Columns: ", names(obf))

# ---------------------------------------------------------------------------
# 2. Aggregate stats table — OU 2.5
# ---------------------------------------------------------------------------

stats = ou25_aggregate_stats(obf; time_window = (-360.0, 0.0))


#=
julia> stats = ou25_aggregate_stats(obf; time_window = (-360.0, 0.0))
14×14 DataFrame
 Row │ time_bucket  selection  n      med_spread_pct  q25_spread_pct  q75_spread_pct  med_depth  q25_depth  q75_depth  med_OBI      med_OBI_stab  jump_freq  OBI_fwd_cor  entry_pct 
     │ Float64      Symbol     Int64  Float64         Float64         Float64         Float64    Float64    Float64    Float64      Float64       Float64    Float64      Float64   
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │      -360.0  over_25       11         2.15054         1.83486         2.57468    25156.0    12613.0    30583.5  -0.252463      NaN          0.0        0.122326    0.0
   2 │      -330.0  over_25       14         2.32841         2.03046         2.73973    29495.0    15252.2    36311.0  -0.321583        0.123004   0.0       -0.00625611  0.0
   3 │      -150.0  over_25       20         1.83486         1.75881         2.76498    47779.0    18734.8    54122.5  -0.0674244       0.144602   0.1        0.381676    0.05
   4 │      -120.0  over_25       41         1.75881         1.51899         2.55754    45109.0    32392.0    56514.0  -0.159096        0.136202   0.219512   0.654514    0.268293
   5 │       -90.0  over_25       65         2.06186         1.51134         2.64317    34387.0    20020.0    55416.0  -0.1446          0.175187   0.107692   0.193781    0.153846
   6 │       -60.0  over_25       85         2.43038         1.6             2.8436     42724.0    19143.0    60453.0  -0.112857        0.21019    0.176471   0.234763    0.0588235
   7 │       -30.0  over_25       97         2.45399         1.5748          2.8436     50481.0    33499.0    70634.0   0.00621102      0.18328    0.103093   0.115305    0.0824742
   8 │      -360.0  under_25      11         1.61725         1.55676         3.06513    15643.0    14346.5    25424.0   0.444671      NaN          0.0       -0.0534734   0.0
   9 │      -330.0  under_25      14         1.80311         1.61944         2.91703    22198.5    14848.8    28676.2   0.444229        0.242047   0.0        0.0390033   0.0
  10 │      -150.0  under_25      20         2.48139         1.62602         3.53794    39857.5    18551.5    53878.0   0.220492        0.23847    0.1        0.524945    0.0
  11 │      -120.0  under_25      41         2.24719         1.61725         2.92683    56150.0    38326.0    63542.0   0.0561183       0.162609   0.170732  -0.0889918   0.146341
  12 │       -90.0  under_25      65         2.16216         1.96078         2.92683    44472.0    21018.0    62377.0  -0.0242499       0.112716   0.107692   0.0991416   0.169231
  13 │       -60.0  under_25      85         2.62467         1.9802          2.95567    36142.0    21809.0    63727.0   0.121983        0.157912   0.211765   0.175339    0.164706
  14 │       -30.0  under_25      97         2.23464         1.90476         2.849      40931.0    26601.0    74212.0   0.170393        0.139213   0.123711   0.0609087   0.134021
=#


println("\n" * "="^95)
println("  OU 2.5 ENTRY TIMING STATISTICS  —  30-minute buckets, pre-kickoff")
println("="^95)

for sel in unique(stats.selection)
    s = filter(r -> r.selection == sel, stats)
    println("\n  Selection: $(sel)")
    println("  " * "-"^91)
    @printf("  %-10s %5s  %16s  %16s  %8s  %9s  %7s  %8s  %7s\n",
        "bucket", "n", "spread%(med/IQR)", "depth£(med/IQR)", "OBI_med",
        "OBI_stab", "jump%", "OBI→r", "entry%")
    println("  " * "-"^91)
    for r in eachrow(s)
        @printf("  %-10.0f %5d  %5.2f (%4.2f–%4.2f)  %6.0f (%5.0f–%6.0f)  %+6.3f  %8.3f  %5.1f%%  %+7.3f  %5.1f%%\n",
            r.time_bucket, r.n,
            r.med_spread_pct, r.q25_spread_pct, r.q75_spread_pct,
            r.med_depth, r.q25_depth, r.q75_depth,
            r.med_OBI,
            isnan(r.med_OBI_stab) ? 0.0 : r.med_OBI_stab,
            r.jump_freq * 100,
            isnan(r.OBI_fwd_cor)  ? 0.0 : r.OBI_fwd_cor,
            r.entry_pct * 100)
    end
end


#=
Selection: over_25
  -------------------------------------------------------------------------------------------
  bucket         n  spread%(med/IQR)   depth£(med/IQR)   OBI_med   OBI_stab    jump%     OBI→r   entry%
  -------------------------------------------------------------------------------------------
  -360          11   2.15 (1.83–2.57)   25156 (12613– 30584)  -0.252     0.000    0.0%   +0.122    0.0%
  -330          14   2.33 (2.03–2.74)   29495 (15252– 36311)  -0.322     0.123    0.0%   -0.006    0.0%
  -150          20   1.83 (1.76–2.76)   47779 (18735– 54122)  -0.067     0.145   10.0%   +0.382    5.0%
  -120          41   1.76 (1.52–2.56)   45109 (32392– 56514)  -0.159     0.136   22.0%   +0.655   26.8%
  -90           65   2.06 (1.51–2.64)   34387 (20020– 55416)  -0.145     0.175   10.8%   +0.194   15.4%
  -60           85   2.43 (1.60–2.84)   42724 (19143– 60453)  -0.113     0.210   17.6%   +0.235    5.9%
  -30           97   2.45 (1.57–2.84)   50481 (33499– 70634)  +0.006     0.183   10.3%   +0.115    8.2%

  Selection: under_25
  -------------------------------------------------------------------------------------------
  bucket         n  spread%(med/IQR)   depth£(med/IQR)   OBI_med   OBI_stab    jump%     OBI→r   entry%
  -------------------------------------------------------------------------------------------
  -360          11   1.62 (1.56–3.07)   15643 (14346– 25424)  +0.445     0.000    0.0%   -0.053    0.0%
  -330          14   1.80 (1.62–2.92)   22198 (14849– 28676)  +0.444     0.242    0.0%   +0.039    0.0%
  -150          20   2.48 (1.63–3.54)   39858 (18552– 53878)  +0.220     0.238   10.0%   +0.525    0.0%
  -120          41   2.25 (1.62–2.93)   56150 (38326– 63542)  +0.056     0.163   17.1%   -0.089   14.6%
  -90           65   2.16 (1.96–2.93)   44472 (21018– 62377)  -0.024     0.113   10.8%   +0.099   16.9%
  -60           85   2.62 (1.98–2.96)   36142 (21809– 63727)  +0.122     0.158   21.2%   +0.175   16.5%
  -30           97   2.23 (1.90–2.85)   40931 (26601– 74212)  +0.170     0.139   12.4%   +0.061   13.4%
=#

println("\n  Columns: bucket=mins_to_ko, spread%(med / IQR), depth£(med / IQR),")
println("  OBI_med=order book imbalance, OBI_stab=book churn std,")
println("  jump%=% snaps with >2% price move, OBI→r=corr(OBI_t, mid_{t+5}-mid_t),")
println("  entry%=% snaps meeting ALL entry criteria")
println("="^95)

# ---------------------------------------------------------------------------
# 3.  Derived OU2.5 slice for plots
# ---------------------------------------------------------------------------

buckets   = sort(unique(stats.time_bucket))
over_s    = filter(r -> r.selection == :over_25,  stats)
under_s   = filter(r -> r.selection == :under_25, stats)

# Guard: sort both by bucket
sort!(over_s,  :time_bucket)
sort!(under_s, :time_bucket)

# ---------------------------------------------------------------------------
# Plot 1: Spread % over time — when is cost acceptable?
# ---------------------------------------------------------------------------

p1 = plot(;
    title      = "OU 2.5 — Bid-Ask Spread % over Time",
    xlabel     = "Minutes to Kickoff",
    ylabel     = "Spread % of Mid",
    legend     = :topright,
    size       = (900, 400),
    titlefontsize = 10,
);
hline!(p1, [2.0]; color=:grey, linestyle=:dash, linewidth=1, label="2% threshold");

for (s_df, color, label) in [
    (over_s,  :royalblue, "over 2.5"),
    (under_s, :firebrick, "under 2.5"),
]
    isempty(s_df) && continue
    x   = s_df.time_bucket
    med = s_df.med_spread_pct
    lo  = med .- s_df.q25_spread_pct
    hi  = s_df.q75_spread_pct .- med
    plot!(p1, x, med;
        ribbon    = (lo, hi),
        fillalpha = 0.15,
        color     = color,
        linewidth = 2,
        label     = label,
    );
end
vline!(p1, [0.0]; color=:black, linestyle=:dash, linewidth=1, label="kickoff");
savefig(p1, joinpath(PLOTS_DIR, "ou25_spread_pct.html"));
# display(p1) # Removed to prevent headless GUI crashes

# ---------------------------------------------------------------------------
# Plot 2: Market depth over time — when can you fill a stake?
# ---------------------------------------------------------------------------

p2 = plot(;
    title      = "OU 2.5 — Available Depth (L1–L3) over Time",
    xlabel     = "Minutes to Kickoff",
    ylabel     = "Total Depth £",
    legend     = :topleft,
    size       = (900, 400),
    titlefontsize = 10,
);
for thresh in [200.0, 500.0, 1000.0]
    hline!(p2, [thresh]; color=:grey, linestyle=:dot, linewidth=1,
        label = thresh == 200.0 ? "depth thresholds" : "")
end

for (s_df, color, label) in [
    (over_s,  :royalblue, "over 2.5"),
    (under_s, :firebrick, "under 2.5"),
]
    isempty(s_df) && continue
    x   = s_df.time_bucket
    med = s_df.med_depth
    lo  = med .- s_df.q25_depth
    hi  = s_df.q75_depth .- med
    plot!(p2, x, med;
        ribbon    = (lo, hi),
        fillalpha = 0.12,
        color     = color,
        linewidth = 2,
        label     = label,
    );
end
vline!(p2, [0.0]; color=:black, linestyle=:dash, linewidth=1, label="kickoff");
savefig(p2, joinpath(PLOTS_DIR, "ou25_depth.html"));
# display(p2) # Removed to prevent headless GUI crashes

# ---------------------------------------------------------------------------
# Plot 3: Price jump frequency — find the lineup announcement window
# ---------------------------------------------------------------------------

p3 = plot(;
    title     = "OU 2.5 — Price Jump Frequency (|Δmid/5m| > 2% of mid)",
    xlabel    = "Minutes to Kickoff",
    ylabel    = "% Snapshots with Price Jump",
    legend    = :topright,
    size      = (900, 380),
    titlefontsize = 10,
);
for (s_df, color, label) in [
    (over_s,  :royalblue, "over 2.5"),
    (under_s, :firebrick, "under 2.5"),
]
    isempty(s_df) && continue
    bar!(p3, s_df.time_bucket, s_df.jump_freq .* 100;
        color     = color,
        alpha     = 0.6,
        label     = label,
        bar_width = 25,
    );
end
vline!(p3, [0.0]; color=:black, linestyle=:dash, linewidth=1, label="kickoff");
savefig(p3, joinpath(PLOTS_DIR, "ou25_jump_frequency.html"));
# display(p3) # Removed to prevent headless GUI crashes

# ---------------------------------------------------------------------------
# Plot 4: OBI predictive correlation — when does OBI actually signal direction?
# ---------------------------------------------------------------------------

p4 = plot(;
    title     = "OU 2.5 — OBI Predictive Correlation (corr(OBI_t, mid_{t+5} - mid_t))",
    xlabel    = "Minutes to Kickoff",
    ylabel    = "Pearson r",
    legend    = :topright,
    ylims     = (-0.5, 0.5),
    size      = (900, 380),
    titlefontsize = 10,
);
hline!(p4, [0.0]; color=:grey, linestyle=:dot, linewidth=1, label="");
hline!(p4, [0.1, -0.1]; color=:grey, linestyle=:dash, linewidth=1, label="±0.1 reference");

for (s_df, color, label) in [
    (over_s,  :royalblue, "over 2.5"),
    (under_s, :firebrick, "under 2.5"),
]
    isempty(s_df) && continue
    valid = .!isnan.(s_df.OBI_fwd_cor)
    plot!(p4, s_df.time_bucket[valid], s_df.OBI_fwd_cor[valid];
        color     = color,
        linewidth = 2,
        marker    = :circle,
        markersize = 4,
        label     = label,
    );
end
vline!(p4, [0.0]; color=:black, linestyle=:dash, linewidth=1, label="kickoff");
savefig(p4, joinpath(PLOTS_DIR, "ou25_obi_predictive_cor.html"));
# display(p4) # Removed to prevent headless GUI crashes

# ---------------------------------------------------------------------------
# Plot 5: Entry opportunity % — the "when can I enter" summary
# ---------------------------------------------------------------------------

p5 = plot(;
    title     = "OU 2.5 — % Snapshots Meeting All Entry Criteria\n(spread<2%, depth>£500, OBI_stab<0.15, |OBI|<0.4)",
    xlabel    = "Minutes to Kickoff",
    ylabel    = "% Valid Entry Snapshots",
    legend    = :topright,
    size      = (900, 400),
    titlefontsize = 9,
);
for (s_df, color, label) in [
    (over_s,  :royalblue, "over 2.5"),
    (under_s, :firebrick, "under 2.5"),
]
    isempty(s_df) && continue
    bar!(p5, s_df.time_bucket, s_df.entry_pct .* 100;
        color     = color,
        alpha     = 0.65,
        label     = label,
        bar_width = 25,
    );
end
vline!(p5, [0.0]; color=:black, linestyle=:dash, linewidth=1, label="kickoff");
savefig(p5, joinpath(PLOTS_DIR, "ou25_entry_opportunity.html"));
# display(p5) # Removed to prevent headless GUI crashes

# ---------------------------------------------------------------------------
# Combined summary figure (all 5 panels)
# ---------------------------------------------------------------------------

fig_combined = plot(p1, p2, p3, p4, p5;
    layout = (5, 1),
    size   = (950, 1800),
    left_margin  = 6Plots.mm,
    bottom_margin = 3Plots.mm,
);
savefig(fig_combined, joinpath(PLOTS_DIR, "ou25_entry_timing_summary.html"));
println("\nAll interactive HTML plots saved to: $PLOTS_DIR")

# ---------------------------------------------------------------------------
# 5. Web Server Setup (Headless Plot Viewer)
# ---------------------------------------------------------------------------

"""
    serve_plots(; port=8080)

Starts a local web server to serve the generated HTML plots. 
Run this, then on your laptop run `ssh -L 8080:localhost:8080 user@server`
and open `http://localhost:8080` in your browser.
"""
function serve_plots(; port=8080)
    println("\n=== Starting Headless Plot Server ===")
    println("To view these interactive plots:")
    println("1. On your physical laptop terminal, run:")
    println("   ssh -L \$port:localhost:\$port $(ENV["USER"])@$(gethostname())")
    println("2. Open your web browser to:")
    println("   http://localhost:\$port")
    println("=====================================\n")
    
    try
        @eval Main import LiveServer
        Main.LiveServer.serve(dir=PLOTS_DIR, port=port)
    catch
        println("Error: LiveServer.jl is not installed.")
        println("Please run:  using Pkg; Pkg.add(\"LiveServer\")")
        println("And then call serve_plots() again.")
    end
end

# ---------------------------------------------------------------------------
# 4. Headline numbers (printed last — easy to read after charts load)
# ---------------------------------------------------------------------------

println("\n=== HEADLINE NUMBERS: OU 2.5 ===")

# Efficient market window: where spread_pct < 2% AND depth > £500
efficient = filter(r -> r.med_spread_pct < 2.0 && r.med_depth > 500.0, stats)
if !isempty(efficient)
    window_start = minimum(efficient.time_bucket)
    println("  Market efficient window (spread<2%, depth>£500): from $(Int(window_start)) mins")
else
    println("  Market never crosses both efficiency thresholds in this dataset.")
end

# Peak jump window (most lineup news)
jump_window = stats[argmax(stats.jump_freq), :]
println("  Peak price-jump bucket: $(Int(jump_window.time_bucket)) mins  ($(round(jump_window.jump_freq*100, digits=1))% of snapshots, sel=$(jump_window.selection))")

# Best OBI predictive window
obi_valid = filter(r -> !isnan(r.OBI_fwd_cor), stats)
if !isempty(obi_valid)
    best_obi = obi_valid[argmax(abs.(obi_valid.OBI_fwd_cor)), :]
    println("  Strongest OBI signal: $(Int(best_obi.time_bucket)) mins  (r=$(round(best_obi.OBI_fwd_cor, digits=3)), sel=$(best_obi.selection))")
end

# Best entry window (highest entry_pct)
best_entry = stats[argmax(stats.entry_pct), :]
println("  Best entry opportunity bucket: $(Int(best_entry.time_bucket)) mins  ($(round(best_entry.entry_pct*100, digits=1))% of snapshots qualify, sel=$(best_entry.selection))")
