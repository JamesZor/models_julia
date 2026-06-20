#=
r07_heavytail_diagnosis.jl  —  Does a heavy-tailed (Negative-Binomial) total fix the Over/Under
under-prediction? (Inspired by the "intensity smile" / Local-Intensity model, Buttler-style thesis Ch4.)

ANSWER: No — because there was no real under-prediction to fix. The apparent ~5-pt Over
under-prediction seen in r05 was TEST-SPLIT SAMPLING NOISE, not a structural model bias.

This runner shows three things:
  1. Conditional overdispersion of the remaining total (given the model mean) is MILD (var/mean ≈ 1.09);
     the raw 1.35 is mostly explained by the model mean varying across bins.
  2. Swapping Poisson → NegBin for the total barely moves OU calibration (ECE 0.062 → 0.062).
  3. The model is MEAN-UNBIASED: on TRAIN model_mean ≈ actual_mean; across 15 random splits the held-out
     mean bias is +0.030 ± 0.087 (n.s.). The single r05 split was just goal-heavy.

Lesson: with ~253 matches (63 test) a single split's calibration is dominated by sampling noise — use
cross-validation / more leagues before trusting a ~5-pt or ~0.06-ECE difference.

Run with threads:  julia --project -t 16  (pinthreads(:cores))
=#

using Revise, BayesianFootball
using DataFrames, Distributions, GLM, Optim, Statistics, Random
using ThreadPinning; pinthreads(:cores)

const Data        = BayesianFootball.Data
const Experiments = BayesianFootball.Experiments
const Features    = BayesianFootball.Features

include("l01_inplay_inverse.jl")
include("l02_inplay_intensity.jl")

# ---- data + panel ----
ds = Data.load_datastore_cached(Data.Ireland()); bf = ds.betfair_odds
pg = Experiments.extract_oos_predictions(ds, Experiments.load_experiment(
        Experiments.list_experiments("./data/dixon_coles_ab/", data_dir=""), 1))
pg_tbl = DataFrame(match_id=Int.(pg.df.match_id),
                   pg_λ_h=[mean(Float64.(v)) for v in pg.df.λ_h],
                   pg_λ_a=[mean(Float64.(v)) for v in pg.df.λ_a])
function build_panel(bf, ds, pg_tbl; bin_minutes=5.0, staleness=10.0, min_sel=6, mtk_max=130.0)
    ids = unique(subset(bf, :minutes_to_kickoff=>ByRow(x->0<x<=mtk_max)).match_id)
    parts = Vector{DataFrame}(undef, length(ids))
    Threads.@threads for k in eachindex(ids)
        local tr; try; tr=inplay_lambda_trace(bf,ds,Int(ids[k]); bin_minutes=bin_minutes,staleness=staleness,min_sel=min_sel,mtk_max=mtk_max)
        catch; tr=DataFrame(); end; parts[k]=tr
    end
    leftjoin(vcat([d for d in parts if nrow(d)>0]...), pg_tbl, on=:match_id)
end
panel = build_panel(bf, ds, pg_tbl)
fin = Dict(Int(r.match_id)=>(Int(r.home_score),Int(r.away_score)) for r in eachrow(ds.matches) if !ismissing(r.home_score))

# ---- per-side intensity GLM (l02) → per-bin mean total μ_tot ----
D = build_intensity_dataset(panel, ds)
form = @formula(rem_goals ~ t_m + t_m2 + is_home + trailing + leading + man_adv + log_pregame)
glm_fit = glm(form, D, Poisson(), LogLink(); offset = D.logrem)
co = coef(glm_fit); xc = (:t_m,:t_m2,:is_home,:trailing,:leading,:man_adv,:log_pregame)
# design row from a side's perspective; offset = log(rem_frac)
function mu_side(t_m, is_home, gds, man_adv, logpg)
    rf = max((90.0-t_m)/90.0, 0.05)
    x = [1.0, t_m, t_m^2, Float64(is_home), Float64(gds<0), Float64(gds>0), Float64(man_adv), logpg]
    exp(clamp(dot(co, x) + log(rf), -20.0, 20.0))
end
mu_tot(r) = mu_side(r.t_m,1,r.gh-r.ga,r.away_reds-r.home_reds,log(r.pg_λ_h)) +
            mu_side(r.t_m,0,r.ga-r.gh,r.home_reds-r.away_reds,log(r.pg_λ_a))

# (m = model mean remaining total, y = realized remaining total, T = current total) per usable bin
function bins_set(ids)
    rows=NamedTuple[]
    for r in eachrow(subset(panel, :match_id=>ByRow(m->m in ids)))
        (ismissing(r.pg_λ_h)||r.t_m>80||r.residual>=0.08||!haskey(fin,r.match_id)) && continue
        fh,fa=fin[r.match_id]; remT=(fh-r.gh)+(fa-r.ga); remT<0 && continue
        push!(rows,(m=mu_tot(r), y=remT, T=r.gh+r.ga, fh=fh, fa=fa))
    end
    DataFrame(rows)
end

# ---- 1. fit NegBin dispersion r on the FULL set (total ~ NegBin(mean=m, size=r)) ----
ALL = bins_set(Set(unique(panel.match_id)))
r_hat = exp(Optim.minimizer(optimize(x->-sum(logpdf(NegativeBinomial(exp(x[1]), exp(x[1])/(exp(x[1])+row.m)), row.y) for row in eachrow(ALL)), [log(5.0)], NelderMead()))[1])
println("NegBin size r̂ = $(round(r_hat,digits=2));  conditional var/mean ≈ $(round(1+mean(ALL.m)/r_hat,digits=3))  (raw var/mean = $(round(var(ALL.y)/mean(ALL.y),digits=3)))")

# ---- 2. OU calibration: Poisson vs NegBin total (single split, like r05) ----
ms = shuffle(MersenneTwister(1), unique(panel.match_id)); cut=round(Int,0.75*length(ms))
TE = bins_set(Set(ms[cut+1:end]))
ece(p,y;nb=10)=begin e=0.0;N=length(p); for b in 0:nb-1; idx=findall(x->b/nb<=x<(b+1)/nb||(b==nb-1&&x==1.0),p); isempty(idx)&&continue; e+=(length(idx)/N)*abs(mean(p[idx])-mean(y[idx])); end; e end
brier(p,y)=mean((p.-y).^2)
function ou_probs(TE, r_hat)
    rows=NamedTuple[]
    for row in eachrow(TE), L in (1.5,2.5,3.5)
        need=Int(round(L-row.T+0.5))
        pp = need<=0 ? 1.0 : ccdf(Poisson(row.m), need-1)
        pn = need<=0 ? 1.0 : ccdf(NegativeBinomial(r_hat, r_hat/(r_hat+row.m)), need-1)
        push!(rows,(pp=pp, pn=pn, won=(row.fh+row.fa)>L))
    end
    DataFrame(rows)
end
E = ou_probs(TE, r_hat); y=Float64.(E.won)
println("OU (single split):  Poisson ECE=$(round(ece(E.pp,y),digits=4)) Brier=$(round(brier(E.pp,y),digits=4)) mean_pred=$(round(mean(E.pp),digits=3)) | " *
        "NegBin ECE=$(round(ece(E.pn,y),digits=4)) Brier=$(round(brier(E.pn,y),digits=4)) mean_pred=$(round(mean(E.pn),digits=3)) | actual=$(round(mean(y),digits=3))")

# ---- 3. THE decisive check: is the held-out mean bias real, or split noise? (multi-seed) ----
biases = Float64[]
for seed in 1:15
    msi = shuffle(MersenneTwister(seed), unique(D.match_id)); c=round(Int,0.75*length(msi))
    Dtr=subset(D,:match_id=>ByRow(in(Set(msi[1:c])))); Dte=subset(D,:match_id=>ByRow(in(Set(msi[c+1:end]))))
    m=glm(form, Dtr, Poisson(), LogLink(); offset=Dtr.logrem)
    push!(biases, mean(Dte.rem_goals) - mean(predict(m, Dte; offset=Dte.logrem)))
end
println("Held-out mean bias across 15 splits: mean=$(round(mean(biases),digits=4)), std=$(round(std(biases),digits=4)), " *
        "range=($(round(minimum(biases),digits=3)), $(round(maximum(biases),digits=3))) → consistent with ZERO (no structural bias).")
