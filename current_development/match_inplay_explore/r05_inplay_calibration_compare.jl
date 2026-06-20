#=
r05_inplay_calibration_compare.jl  —  Compare game-state model variants by CALIBRATION (Over/Under).

Variants (l03 config.game_state): :none | :linear | :hier_replace | :hier_addon.
Each fit on the 75% match-train split; evaluated on held-out matches by:
  - Over/Under predictive calibration (reliability, ECE, Brier, log-loss) — PRIMARY,
  - calibration BY GAME STATE (per goal-diff bucket) — "which states each variant gets right",
  - held-out elpd on the count target.

Odds come from the PROJECT pipeline (posterior-preserving), not hand-rolled:
  build remaining-goals ScoreMatrix from model μ  ->  compute_market_probs(S, MarketOverUnder(L − T))
  (src/predictions/score_computation/poisson.jl + market_inference/over_under.jl). The in-play tweak
  is the line shift by the current total T.

Run with threads:  julia --project -t 16   (pinthreads(:cores))
=#

using Revise
using BayesianFootball
using DataFrames, Distributions, Turing, Statistics, LinearAlgebra, Random, MCMCChains
using ThreadPinning
pinthreads(:cores)

const Data        = BayesianFootball.Data
const Samplers    = BayesianFootball.Samplers
const Predictions = BayesianFootball.Predictions
const Experiments = BayesianFootball.Experiments
const Features    = BayesianFootball.Features

include("l01_inplay_inverse.jl")
include("l02_inplay_intensity.jl")
include("l03_inplay_turing.jl")

# ==========================================================================
# 1. DATA + PANEL + INPUTS  (reuse l01/l02/l03)
# ==========================================================================
ds = Data.load_datastore_cached(Data.Ireland())
bf = ds.betfair_odds
saved_files      = Experiments.list_experiments("./data/dixon_coles_ab/", data_dir = "")
res_pre_game     = Experiments.load_experiment(saved_files, 1)
pre_game_latents = Experiments.extract_oos_predictions(ds, res_pre_game)
pg_tbl = DataFrame(match_id = Int.(pre_game_latents.df.match_id),
                   pg_λ_h   = [mean(Float64.(v)) for v in pre_game_latents.df.λ_h],
                   pg_λ_a   = [mean(Float64.(v)) for v in pre_game_latents.df.λ_a])

function build_panel(bf, ds, pg_tbl; config = Features.DoublePoissonMarketFeature(),
                     bin_minutes = 5.0, staleness = 10.0, min_sel = 6, mtk_max = 130.0)
    ids = unique(subset(bf, :minutes_to_kickoff => ByRow(x -> 0 < x <= mtk_max)).match_id)
    parts = Vector{DataFrame}(undef, length(ids))
    Threads.@threads for k in eachindex(ids)
        local tr
        try; tr = inplay_lambda_trace(bf, ds, Int(ids[k]), config; bin_minutes=bin_minutes,
                                      staleness=staleness, min_sel=min_sel, mtk_max=mtk_max)
        catch; tr = DataFrame(); end
        parts[k] = tr
    end
    leftjoin(vcat([df for df in parts if nrow(df) > 0]...), pg_tbl, on = :match_id)
end

panel  = build_panel(bf, ds, pg_tbl; bin_minutes = 5.0)
inp    = build_intensity_inputs(panel, ds)
ms     = shuffle(MersenneTwister(1), unique(inp.match_id)); cut = round(Int, 0.75 * length(ms))
tr_ids, te_ids = Set(ms[1:cut]), Set(ms[cut+1:end])
inp_tr, inp_te = subset_inputs(inp, tr_ids), subset_inputs(inp, te_ids)

# ==========================================================================
# 2. FIT THE GAME-STATE VARIANTS
# ==========================================================================
variants = ["none"         => InPlayIntensityConfig(game_state = :none),
            "linear"       => InPlayIntensityConfig(game_state = :linear),
            "hier_replace" => InPlayIntensityConfig(game_state = :hier_replace),
            "hier_addon"   => InPlayIntensityConfig(game_state = :hier_addon)]
nuts = Samplers.NUTSConfig(n_samples = 1000, n_warmup = 500, n_chains = 4, show_progress = false)
chains = Dict{String,Any}()
for (nm, cfg) in variants
    println("[SAMPLING] $nm"); chains[nm] = (Samplers.run_sampler(make_model(inp_tr, cfg), nuts), cfg)
end

# ==========================================================================
# 3. OVER/UNDER PREDICTIVE PROBABILITIES via the project pipeline
# ==========================================================================
# posterior-mean parameters per config (global coefs are tight; team effects off in this grid)
function extract_params(chain, config)
    ᾱ = mean(_chainvec(chain, :α))
    β̄ = vec(mean(_chainmat(chain, :β, length(active_cols(config))), dims = 1))
    δ_gs = has_gs_hier(config) ?
        vec(mean(_chainmat(chain, :z_gs, N_GAME_STATES) .* _chainvec(chain, :σ_gs), dims = 1)) :
        zeros(N_GAME_STATES)
    return (; ᾱ, β̄, δ_gs)
end

"Posterior-mean remaining-goal MEAN for one side under a config."
function predict_mu_side(config, pr, xc, xs, t_m, is_home, gds, man_adv, logpg)
    rf  = max((90.0 - t_m) / 90.0, 0.05)
    xf  = [t_m, t_m^2, Float64(is_home), Float64(gds < 0), Float64(gds > 0), Float64(man_adv), logpg]
    xall = (xf .- xc) ./ xs
    lp  = pr.ᾱ + dot(pr.β̄, xall[active_cols(config)]) + log(rf)
    has_gs_hier(config) && (lp += pr.δ_gs[clamp(gds, -3, 3) + 4])
    return exp(clamp(lp, -20.0, 20.0))
end

build_score_matrix(μh, μa; G = 13) = begin
    S = zeros(G, G, 1); g = 0:(G-1)
    ph = pdf.(Poisson(μh), g); pa = pdf.(Poisson(μa), g)
    @inbounds for j in 1:G, i in 1:G; S[i, j, 1] = ph[i] * pa[j]; end
    Predictions.ScoreMatrix(S)
end

# the dict keys follow the SHIFTED line; grab the 'over' entry (1 sample) robustly
_over_value(d) = (k = first(kk for kk in keys(d) if startswith(String(kk), "over")); d[k][1])

"Model P(final total > L) at a bin, via compute_market_probs with the in-play line shift (L − T)."
function model_over_prob(config, pr, xc, xs, t_m, gh, ga, hr, ar, pg_h, pg_a, L)
    μh = predict_mu_side(config, pr, xc, xs, t_m, 1, gh - ga,  ar - hr, log(pg_h))
    μa = predict_mu_side(config, pr, xc, xs, t_m, 0, ga - gh,  hr - ar, log(pg_a))
    Ls = L - (gh + ga)
    Ls <= -0.5 && return 1.0                     # already over
    return _over_value(Predictions.compute_market_probs(build_score_matrix(μh, μa), Data.MarketOverUnder(Ls)))
end

# Build the OU evaluation set over held-out matches.
function build_ou_eval(chains_entry, panel, ds, te_ids, inp; lines = (1.5, 2.5, 3.5))
    chain, config = chains_entry
    pr = extract_params(chain, config)
    fin = Dict(Int(r.match_id) => (Int(r.home_score), Int(r.away_score))
               for r in eachrow(ds.matches) if !ismissing(r.home_score))
    recs = NamedTuple[]
    for r in eachrow(subset(panel, :match_id => ByRow(m -> m in te_ids)))
        (ismissing(r.pg_λ_h) || r.t_m > 80 || r.residual >= 0.08) && continue
        haskey(fin, r.match_id) || continue
        fh, fa = fin[r.match_id]; T = r.gh + r.ga
        for L in lines
            p_over = model_over_prob(config, pr, inp.x_center, inp.x_scale, r.t_m,
                                     r.gh, r.ga, r.home_reds, r.away_reds, r.pg_λ_h, r.pg_λ_a, L)
            push!(recs, (L = L, gd_bucket = clamp(r.goal_diff, -3, 3), model_p = p_over, won = (fh + fa) > L))
        end
    end
    DataFrame(recs)
end

# ==========================================================================
# 4. CALIBRATION METRICS
# ==========================================================================
brier(p, y)   = mean((p .- y).^2)
logloss(p, y) = -mean(y .* log.(clamp.(p, 1e-9, 1)) .+ (1 .- y) .* log.(clamp.(1 .- p, 1e-9, 1)))
function ece(p, y; nb = 10)
    e = 0.0; N = length(p)
    for b in 0:nb-1
        idx = findall(x -> b/nb <= x < (b+1)/nb || (b == nb-1 && x == 1.0), p)
        isempty(idx) && continue
        e += (length(idx)/N) * abs(mean(p[idx]) - mean(y[idx]))
    end
    return e
end

summary_row(nm, E) = (variant = nm, n = nrow(E),
    ECE = round(ece(E.model_p, Float64.(E.won)), digits = 4),
    Brier = round(brier(E.model_p, Float64.(E.won)), digits = 4),
    LogLoss = round(logloss(E.model_p, Float64.(E.won)), digits = 4))

# Driver (run after chains are fit):
#   evals = Dict(nm => build_ou_eval(chains[nm], panel, ds, te_ids, inp) for nm in keys(chains))
#   comp  = DataFrame([summary_row(nm, evals[nm]) for nm in keys(evals)]); sort!(comp, :ECE)
#   show(comp, allrows=true)
#   # per game-state calibration for the winner:
#   best = comp.variant[1]
#   gs = combine(groupby(evals[best], :gd_bucket), nrow=>:n,
#                :model_p=>(x->round(mean(x),digits=3))=>:pred, :won=>(x->round(mean(x),digits=3))=>:actual)
