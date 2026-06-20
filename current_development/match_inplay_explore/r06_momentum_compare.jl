#=
r06_momentum_compare.jl  —  Does a causal SofaScore-momentum covariate help the in-play model?

Fits the in-play intensity model WITH vs WITHOUT the causal momentum covariate (l06), on the same
held-out split, and compares: β_mom posterior, held-out count elpd, and Over/Under calibration.

Validated finding (Ireland): β_mom ≈ +0.14 (90% CI [0.09, 0.18], ×1.15 per SD), held-out count elpd
−1.0753 → −1.0689 (improves), OU calibration roughly a wash (ECE 0.062→0.059, Brier/LogLoss ~flat).

Run with threads:  julia --project -t 16  (pinthreads(:cores))
=#

using Revise, BayesianFootball
using DataFrames, Distributions, Turing, Statistics, LinearAlgebra, Random, MCMCChains
using ThreadPinning; pinthreads(:cores)

const Data        = BayesianFootball.Data
const Samplers    = BayesianFootball.Samplers
const Predictions = BayesianFootball.Predictions
const Experiments = BayesianFootball.Experiments
const Features    = BayesianFootball.Features

include("l01_inplay_inverse.jl")
include("l02_inplay_intensity.jl")
include("l03_inplay_turing.jl")
include("l06_momentum_feature.jl")

# ---- data + panel + momentum lookup + inputs ----
ds = Data.load_datastore_cached(Data.Ireland()); bf = ds.betfair_odds
saved = Experiments.list_experiments("./data/dixon_coles_ab/", data_dir = "")
pg = Experiments.extract_oos_predictions(ds, Experiments.load_experiment(saved, 1))
pg_tbl = DataFrame(match_id = Int.(pg.df.match_id),
                   pg_λ_h = [mean(Float64.(v)) for v in pg.df.λ_h],
                   pg_λ_a = [mean(Float64.(v)) for v in pg.df.λ_a])
function build_panel(bf, ds, pg_tbl; bin_minutes = 5.0, staleness = 10.0, min_sel = 6, mtk_max = 130.0)
    ids = unique(subset(bf, :minutes_to_kickoff => ByRow(x -> 0 < x <= mtk_max)).match_id)
    parts = Vector{DataFrame}(undef, length(ids))
    Threads.@threads for k in eachindex(ids)
        local tr
        try; tr = inplay_lambda_trace(bf, ds, Int(ids[k]); bin_minutes=bin_minutes, staleness=staleness, min_sel=min_sel, mtk_max=mtk_max)
        catch; tr = DataFrame(); end
        parts[k] = tr
    end
    leftjoin(vcat([d for d in parts if nrow(d) > 0]...), pg_tbl, on = :match_id)
end
panel      = build_panel(bf, ds, pg_tbl)
mom_lookup = build_momentum_lookup(Data.tournament_ids(Data.Ireland()))
inp        = build_intensity_inputs(panel, ds; mom_lookup = mom_lookup)
ms = shuffle(MersenneTwister(1), unique(inp.match_id)); cut = round(Int, 0.75 * length(ms))
inp_tr, inp_te = subset_inputs(inp, Set(ms[1:cut])), subset_inputs(inp, Set(ms[cut+1:end]))
te_ids = Set(ms[cut+1:end])

# ---- fit with vs without momentum (game_state = :linear) ----
nuts = Samplers.NUTSConfig(n_samples = 1000, n_warmup = 500, n_chains = 4, show_progress = false)
cfg_no  = InPlayIntensityConfig(game_state = :linear, use_momentum = false)
cfg_mom = InPlayIntensityConfig(game_state = :linear, use_momentum = true)
ch_no  = Samplers.run_sampler(make_model(inp_tr, cfg_no),  nuts)
ch_mom = Samplers.run_sampler(make_model(inp_tr, cfg_mom), nuts)

βm = _chainvec(ch_mom, :β_mom)
println("β_mom mean=$(round(mean(βm),digits=4))  90% CI=$(round.(quantile(βm,[0.05,0.95]),digits=4))  ×/SD=$(round(exp(mean(βm)),digits=3))")
println("held-out count elpd:  without=$(round(held_out_elpd(ch_no, inp_te, cfg_no).per_obs,digits=4))  " *
        "with=$(round(held_out_elpd(ch_mom, inp_te, cfg_mom).per_obs,digits=4))")

# ---- Over/Under calibration with vs without (project pipeline + in-play line shift) ----
build_score_matrix(μh, μa; G = 13) = begin
    S = zeros(G, G, 1); g = 0:(G-1); ph = pdf.(Poisson(μh), g); pa = pdf.(Poisson(μa), g)
    @inbounds for j in 1:G, i in 1:G; S[i, j, 1] = ph[i] * pa[j]; end
    Predictions.ScoreMatrix(S)
end
_over_value(d) = (k = first(kk for kk in keys(d) if startswith(String(kk), "over")); d[k][1])
function extract_params(chain, config)
    (ᾱ = mean(_chainvec(chain, :α)),
     β̄ = vec(mean(_chainmat(chain, :β, length(active_cols(config))), dims = 1)),
     β̄m = config.use_momentum ? mean(_chainvec(chain, :β_mom)) : 0.0)
end
function predict_mu_side(config, pr, xc, xs, t_m, is_home, gds, man_adv, logpg, mz)
    rf = max((90.0 - t_m) / 90.0, 0.05)
    xf = [t_m, t_m^2, Float64(is_home), Float64(gds < 0), Float64(gds > 0), Float64(man_adv), logpg]
    lp = pr.ᾱ + dot(pr.β̄, ((xf .- xc) ./ xs)[active_cols(config)]) + pr.β̄m * mz + log(rf)
    exp(clamp(lp, -20.0, 20.0))
end
fin = Dict(Int(r.match_id) => (Int(r.home_score), Int(r.away_score)) for r in eachrow(ds.matches) if !ismissing(r.home_score))
function build_ou_eval(chain, config; lines = (1.5, 2.5, 3.5))
    pr = extract_params(chain, config); recs = NamedTuple[]
    mz(mid, t, ih) = config.use_momentum ? (row_net_momentum(mom_lookup, mid, t, ih) - inp.mom_center) / inp.mom_scale : 0.0
    for r in eachrow(subset(panel, :match_id => ByRow(m -> m in te_ids)))
        (ismissing(r.pg_λ_h) || r.t_m > 80 || r.residual >= 0.08 || !haskey(fin, r.match_id)) && continue
        fh, fa = fin[r.match_id]
        μh = predict_mu_side(config, pr, inp.x_center, inp.x_scale, r.t_m, 1, r.gh - r.ga, r.away_reds - r.home_reds, log(r.pg_λ_h), mz(r.match_id, r.t_m, 1))
        μa = predict_mu_side(config, pr, inp.x_center, inp.x_scale, r.t_m, 0, r.ga - r.gh, r.home_reds - r.away_reds, log(r.pg_λ_a), mz(r.match_id, r.t_m, 0))
        for L in lines
            Ls = L - (r.gh + r.ga)
            p = Ls <= -0.5 ? 1.0 : _over_value(Predictions.compute_market_probs(build_score_matrix(μh, μa), Data.MarketOverUnder(Ls)))
            push!(recs, (model_p = p, won = (fh + fa) > L))
        end
    end
    DataFrame(recs)
end
ece(p, y; nb = 10) = begin
    e = 0.0; N = length(p)
    for b in 0:nb-1
        idx = findall(x -> b/nb <= x < (b+1)/nb || (b == nb-1 && x == 1.0), p); isempty(idx) && continue
        e += (length(idx)/N) * abs(mean(p[idx]) - mean(y[idx]))
    end; e
end
brier(p, y)   = mean((p .- y).^2)
logloss(p, y) = -mean(y .* log.(clamp.(p, 1e-9, 1)) .+ (1 .- y) .* log.(clamp.(1 .- p, 1e-9, 1)))
for (nm, ch, cfg) in (("without", ch_no, cfg_no), ("with", ch_mom, cfg_mom))
    E = build_ou_eval(ch, cfg); y = Float64.(E.won); p = Float64.(E.model_p)
    println("OU $nm:  ECE=$(round(ece(p,y),digits=4))  Brier=$(round(brier(p,y),digits=4))  LogLoss=$(round(logloss(p,y),digits=4))")
end
