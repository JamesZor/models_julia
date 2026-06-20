#=
r01_inplay_runner.jl  —  Runner for the in-play market-implied λ decay study.

Pipeline (see l01_inplay_inverse.jl for the maths):
  in-play betfair ticks -> wall-clock binning + LOCF -> per-market vig strip
    -> score-conditioned inversion (remaining-λ) + naive baseline
    -> μ(t) = λ_rem * 90/(90 - t_m)  (detrended per-90 instantaneous rate)
    -> game-state (goal_diff, red cards) decay analysis vs the pre-game λ.

Validated in REPL on Ireland: kickoff baseline total μ ≈ 2.4 (home>away), and a
clear trailing-team-pushes signal in μ by goal difference.

Run with threads:  julia --project -t 32   (then pinthreads(:cores))
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using ThreadPinning
using ProgressMeter

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

# ==========================================================================
# 1. DATA + PRE-GAME LATENTS  (mirrors r00_basic_runner.jl)
# ==========================================================================
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())
bf = ds.betfair_odds

save_dir = "./data/dixon_coles_ab/"
saved_files  = Experiments.list_experiments(save_dir, data_dir = "")
res_pre_game = Experiments.load_experiment(saved_files, 1)
pre_game_latents = Experiments.extract_oos_predictions(ds, res_pre_game)

# Pre-game posterior-mean λ per match (full-match double-Poisson rates).
function pregame_lambda_table(pg)
    df = pg.df
    DataFrame(
        match_id   = Int.(df.match_id),
        pg_λ_h     = [mean(Float64.(v)) for v in df.λ_h],
        pg_λ_a     = [mean(Float64.(v)) for v in df.λ_a],
    )
end
pg_tbl = pregame_lambda_table(pre_game_latents)

# ==========================================================================
# 2. LOADER
# ==========================================================================
include("l01_inplay_inverse.jl")

# ==========================================================================
# 3. BUILD THE FULL IN-PLAY PANEL  (threaded over matches)
# ==========================================================================
"""
    build_panel(bf, ds; config, bin_minutes, kwargs...) -> DataFrame

Run `inplay_lambda_trace` over every match that has in-play ticks, threaded.
"""
function build_panel(bf, ds; config = Features.DoublePoissonMarketFeature(),
                     bin_minutes = 5.0, staleness = 10.0, min_sel = 6, mtk_max = 130.0)
    ids = unique(subset(bf, :minutes_to_kickoff => ByRow(x -> 0 < x <= mtk_max)).match_id)
    parts = Vector{DataFrame}(undef, length(ids))
    p = Progress(length(ids); desc = "in-play fits ")
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
        next!(p)
    end
    finish!(p)
    panel = vcat([df for df in parts if nrow(df) > 0]...)
    return leftjoin(panel, pg_tbl, on = :match_id)
end

# bin_minutes = 5.0 is a good default for Irish liquidity; try 2.0 / 3.0 too.
panel = build_panel(bf, ds; bin_minutes = 5.0)
println("[INFO] panel rows = $(nrow(panel)) over $(length(unique(panel.match_id))) matches")

# Keep only well-identified bins inside the playing window for analysis.
# - residual < 0.06: the inversion actually matched the market.
# - λ_rem < 6: drop the ~0.2% degenerate Nelder-Mead blow-ups (λ -> 100s) that
#   occur when the market is nearly settled (few binding constraints). These pass
#   the residual filter but are nonsense, so we cap remaining goals at a sane 6.
# - μ analysis uses median (robust) and only bins with a finite μ (t_m < 80).
robust_median(x) = round(median(x), digits = 2)
clean = subset(panel,
    :residual => ByRow(x -> !isnan(x) && x < 0.06),
    :t_m      => ByRow(x -> 1 <= x <= 88),
    :λ_rem_h  => ByRow(x -> x < 6),
    :λ_rem_a  => ByRow(x -> x < 6),
)
μclean = subset(clean, :μ_h => ByRow(x -> !isnan(x)))   # bins with finite per-90 μ

# ==========================================================================
# 4. ANALYSIS
# ==========================================================================

# --- (a) Sanity: μ at the first playable bin should track the pre-game λ ----
# Validated on Ireland: corr ≈ 0.64 (home) / 0.63 (away) vs the independent
# pre-game Bayesian double-Poisson model.
first_bins = combine(groupby(subset(μclean, :t_m => ByRow(x -> x <= 12),
                                            :gh  => ByRow(==(0)),
                                            :ga  => ByRow(==(0))),
                              :match_id),
    :μ_h => mean => :μ_h0, :μ_a => mean => :μ_a0,
    :pg_λ_h => first => :pg_λ_h, :pg_λ_a => first => :pg_λ_a)
dropmissing!(first_bins)
corr_h = cor(first_bins.μ_h0, first_bins.pg_λ_h)
corr_a = cor(first_bins.μ_a0, first_bins.pg_λ_a)
println("[SANITY] corr(μ_h@KO, pregame λ_h) = $(round(corr_h, digits=2)), " *
        "corr(μ_a@KO, pregame λ_a) = $(round(corr_a, digits=2))")



#=
13 rows omitted

julia> corr_h = cor(first_bins.μ_h0, first_bins.pg_λ_h)
0.770739096597603

julia> corr_a = cor(first_bins.μ_a0, first_bins.pg_λ_a)
0.7060126411075571

julia> println("[SANITY] corr(μ_h@KO, pregame λ_h) = $(round(corr_h, digits=2)), " *
               "corr(μ_a@KO, pregame λ_a) = $(round(corr_a, digits=2))")
[SANITY] corr(μ_h@KO, pregame λ_h) = 0.77, corr(μ_a@KO, pregame λ_a) = 0.71
=#


# --- (b) Decay by game state: median μ as a function of (minute bucket, goal_diff)
μclean.t_bucket  = 10 .* fld.(μclean.t_m, 10)           # 0,10,20,... decade buckets
μclean.gd_bucket = clamp.(μclean.goal_diff, -2, 2)
decay_by_state = combine(groupby(μclean, [:t_bucket, :gd_bucket]),
    nrow => :n,
    :μ_h => robust_median => :μ_home,
    :μ_a => robust_median => :μ_away)
sort!(decay_by_state, [:gd_bucket, :t_bucket])

# Headline: median μ by goal difference (collapsed over time).
by_goal_diff = combine(groupby(μclean, :gd_bucket),
    nrow => :n, :μ_h => robust_median => :μ_home, :μ_a => robust_median => :μ_away)
sort!(by_goal_diff, :gd_bucket)

# --- (c) Red-card effect: man-advantage on remaining rates ------------------
μclean.man_adv = μclean.away_reds .- μclean.home_reds    # +ve => home has more men
redcard_effect = combine(groupby(subset(μclean, :man_adv => ByRow(!=(0))), :man_adv),
    nrow => :n,
    :μ_h => robust_median => :μ_home,
    :μ_a => robust_median => :μ_away)
sort!(redcard_effect, :man_adv)

# --- (d) Conditioned vs naive: how much does score-conditioning matter? ------
clean.cond_minus_naive_h = clean.λ_rem_h .- clean.λ_naive_h
clean.cond_minus_naive_a = clean.λ_rem_a .- clean.λ_naive_a

println("[MEDIAN μ by goal difference]"); show(by_goal_diff, allrows = true); println()
println("[RED CARD man-advantage effect]"); show(redcard_effect, allrows = true); println()
println("[DECAY by goal_diff x minute]"); show(decay_by_state, allrows = true); println()

#=
# ==========================================================================
# 5. PLOTS (GLMakie — on the kaimon server call ex(...; mt=true))
# ==========================================================================
using GLMakie

# (i) single-match λ_rem / μ trace
function plot_match_trace(panel, mid)
    d = sort(subset(panel, :match_id => ByRow(==(mid)), :t_m => ByRow(x -> x <= 90)), :t_m)
    fig = Figure(size = (900, 500))
    ax1 = Axis(fig[1, 1]; xlabel = "match minute", ylabel = "remaining λ", title = "match $mid")
    lines!(ax1, d.t_m, d.λ_rem_h; label = "home"); lines!(ax1, d.t_m, d.λ_rem_a; label = "away")
    axislegend(ax1)
    ax2 = Axis(fig[1, 2]; xlabel = "match minute", ylabel = "μ (per-90 instantaneous)")
    lines!(ax2, d.t_m, d.μ_h; label = "home"); lines!(ax2, d.t_m, d.μ_a; label = "away")
    axislegend(ax2)
    fig
end

# (ii) aggregate decay curves by goal difference
function plot_decay_by_state(decay_by_state)
    fig = Figure(size = (900, 500))
    ax = Axis(fig[1, 1]; xlabel = "minute bucket", ylabel = "mean μ_home",
              title = "Home remaining rate by goal difference")
    for g in sort(unique(decay_by_state.gd_bucket))
        d = sort(subset(decay_by_state, :gd_bucket => ByRow(==(g))), :t_bucket)
        lines!(ax, d.t_bucket, d.μ_home; label = "GD=$g")
    end
    axislegend(ax)
    fig
end
=#
