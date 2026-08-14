# =============================================================================
# r06 — λ-SPREAD DIAGNOSTIC + PRIOR-BINDINGNESS SWEEP
#
# WHY. r04/r05 closed the *within-match* count law: given a well-informed mean, both Ireland
# leagues sit at Poisson (D = 0.98 / 0.86). So WP2's "1X2 dispersion is half the market's" must
# live in how far λ moves fixture to fixture. This runner measures that directly.
#
# PRECONDITION. This is a zero-sampling script. It reads the chains r04 and r05 already left in
# the session — `sn4_slots` / `sn5_slots` (rung 3 = the flat-dispersion rung, 6 chains each) and
# `sn4_fs` / `sn5_fs`. Run r04 and r05 first, in this REPL. Nothing here refits anything.
#
# TWO QUESTIONS, deliberately kept apart:
#
#   (1) WHICH PRIORS ARE ACTUALLY BINDING?  For every scalar the engine samples, compare the
#       posterior against its own prior: `ratio = post_sd/prior_sd` (how much the data tightened
#       it) and `shift = (post_mean − prior_mean)/prior_sd` (how far the data dragged it). A prior
#       is only worth loosening if it is BOTH tight (ratio → 1) and pushing (|shift| large). A
#       parameter with ratio 0.3 is data-dominated; widening its prior changes nothing.
#
#   (2) IS THE SPREAD ACTUALLY TOO NARROW?  Reconstruct the model's own supremacy and total at
#       posterior-mean parameters and compare their cross-fixture sd against the market's.
#
#       The bare ratio sd_model/sd_mkt is NOT the test. A predictor that correlates ρ with the
#       truth SHOULD have sd = ρ·sd_truth — that is what shrinking to the right degree looks like.
#       So the number that matters is
#
#           over = sd_model / (ρ · sd_mkt)
#
#       over ≈ 1  → correctly shrunk; the deficit is ρ (skill), not spread, and widening priors
#                   would push it past optimal and hurt calibration.
#       over > 1  → spread without the skill to justify it — noise dressed as conviction.
#
# CAVEATS, stated up front. In-sample, on the single r04/r05 fold (biweek 12), at posterior-MEAN
# parameters, evaluated unweighted across all training rows even though the fit is time-decay
# weighted to an effective N ≈ 60. The `over` identity treats the market as truth. This diagnoses
# spread calibration against the market; it is not an OOS accuracy measurement. The honest OOS
# version reads WP2/WP10's stored fold predictions.
# =============================================================================

using Statistics, DataFrames
using Turing: MCMCChains

const PG = BayesianFootball.Models.PreGame

# Engine's own market gate — mirrors l01 line 274 and the shipped engine's line 201 verbatim.
_r6_mok(x) = !ismissing(x) && (xf = Float64(x); !isnan(xf) && 0.02 < xf < 20.0)

_r6_chain(slots, rung) = cat([c for c in slots[rung] if isa(c, MCMCChains.Chains)]...; dims = 3)

# ---------------------------------------------------------------------------
# (1) PRIOR BINDINGNESS
# ---------------------------------------------------------------------------
# Half-normal moments for the truncated(Normal(0, s), lower=0) hyperpriors: mean = s·√(2/π),
# sd = s·√(1 − 2/π). Using the *truncated* moments rather than s itself is what makes `shift`
# comparable across the plain-normal and half-normal rows.
_r6_hn(s) = (s * sqrt(2 / pi), s * sqrt(1 - 2 / pi))

const R6_PRIORS = Dict(
    "p_dyn.w_Outfield_att" => (0.08, 0.05),   # Normal(0.08, 0.05)
    "p_dyn.w_Outfield_def" => (-0.08, 0.05),  # Normal(-0.08, 0.05)
    "p_dyn.w_G_att"        => (0.0, 0.2),
    "p_dyn.w_G_def"        => (0.0, 0.2),
    "ha.γ_base"            => (0.2, 0.2),
    "ha.σ_γ"               => _r6_hn(0.1),
    "kap.κ_base"           => (1.0, 0.2),
    "kap.σ_κ"              => _r6_hn(0.1),
    "σ_sup"                => (0.10, 0.10),   # truncated below at 0.02; barely binds
    "σ_smile"              => (0.15, 0.10),
)

const R6_ORDER = ["p_dyn.w_Outfield_att", "p_dyn.w_Outfield_def", "p_dyn.w_G_att", "p_dyn.w_G_def",
                  "ha.γ_base", "ha.σ_γ", "kap.κ_base", "kap.σ_κ", "σ_sup", "σ_smile"]

function r6_prior_table(slots, rung = 3)
    ch = _r6_chain(slots, rung)
    DataFrame([begin
        v = vec(Array(ch[Symbol(p)]))
        pm, ps = R6_PRIORS[p]
        (param = p, prior_m = pm, prior_sd = ps, post_m = mean(v), post_sd = std(v),
         ratio = std(v) / ps, shift_sd = (mean(v) - pm) / ps)
    end for p in R6_ORDER])
end

# ---------------------------------------------------------------------------
# (2) λ SPREAD vs THE MARKET
# ---------------------------------------------------------------------------
# Rebuilds λ_h/λ_a from the flat feature vectors exactly as the @model does (l01:172-179), at
# posterior-mean parameters. The intercept cancels out of supremacy but NOT out of the total, so
# it is carried here; likewise kappa, which is a per-team multiplier on the rate itself.
function r6_spread(slots, fs, mdl; rung = 3)
    ch = _r6_chain(slots, rung)
    d  = fs.data
    n_teams, n_seasons = Int(d[:n_teams]), Int(d[:n_seasons])

    ha = vec(mean(PG.extract_home_advantage(ch, mdl.homeadvantage_config, n_teams), dims = 1))
    kp = vec(mean(PG.extract_kappa(ch, mdl.kappa_config, n_teams), dims = 1))
    pd = PG.extract_dynamics(ch, mdl.player_dynamics_config, "p_dyn", n_teams)
    it = PG.extract_interception(ch, mdl.interception_config, n_seasons)
    μb, δm = vec(mean(it.μ_base, dims = 1)), vec(mean(it.δ_month, dims = 1))

    wGa, wGd = mean(pd.w_G_att), mean(pd.w_G_def)
    wOa, wOd = mean(pd.w_Outfield_att), mean(pd.w_Outfield_def)
    br = mdl.player_ratings_feature.tracker.prior_mean

    hi, ai = Vector{Int}(d[:flat_home_ids]), Vector{Int}(d[:flat_away_ids])
    si, mi = Vector{Int}(d[:season_indices]), Vector{Int}(d[:flat_months])

    hGc = Float64.(d[:flat_home_G_rating]) .- br
    hOc = (Float64.(d[:flat_home_D_rating]) .+ Float64.(d[:flat_home_M_rating]) .+
           Float64.(d[:flat_home_F_rating])) .- 10br
    aGc = Float64.(d[:flat_away_G_rating]) .- br
    aOc = (Float64.(d[:flat_away_D_rating]) .+ Float64.(d[:flat_away_M_rating]) .+
           Float64.(d[:flat_away_F_rating])) .- 10br

    att_h = wGa .* hGc .+ wOa .* hOc;  def_h = wGd .* hGc .+ wOd .* hOc
    att_a = wGa .* aGc .+ wOa .* aOc;  def_a = wGd .* aGc .+ wOd .* aOc

    μv = μb[si] .+ δm[mi]
    λh = kp[hi] .* exp.(μv .+ ha[hi] .+ att_h .+ def_a)
    λa = kp[ai] .* exp.(μv            .+ att_a .+ def_h)

    mh, ma = d[:flat_market_λ_home], d[:flat_market_λ_away]
    k = findall(i -> _r6_mok(mh[i]) && _r6_mok(ma[i]), eachindex(mh))
    mkh, mka = Float64.(mh[k]), Float64.(ma[k])

    sup_mod = log.(λh[k]) .- log.(λa[k]);  sup_mkt = log.(mkh) .- log.(mka)
    tot_mod = log.(λh[k] .+ λa[k]);        tot_mkt = log.(mkh .+ mka)

    stat(a, b) = begin
        ρ = cor(a, b)
        (sd_model = std(a), sd_mkt = std(b), ratio = std(a) / std(b), ρ = ρ,
         sd_optimal = ρ * std(b), over = std(a) / (ρ * std(b)))
    end

    (n = length(k), coverage = length(k) / length(mh),
     supremacy = stat(sup_mod, sup_mkt),
     totals    = stat(tot_mod, tot_mkt),
     mean_total_model = mean(λh[k] .+ λa[k]), mean_total_mkt = mean(mkh .+ mka))
end

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
r6_prior_79  = r6_prior_table(sn4_slots)
r6_prior_718 = r6_prior_table(sn5_slots)
r6_sp_79     = r6_spread(sn4_slots, sn4_fs, sn4_model(sn4_rungs[3].disp))
r6_sp_718    = r6_spread(sn5_slots, sn5_fs, sn5_model(sn5_rungs[3].disp))

open(joinpath(@__DIR__, "r06_out.txt"), "w") do io
    println(io, "r06 — λ-spread diagnostic (in-sample, posterior-mean, r04/r05 fold)\n")
    for (tag, t) in (("79", r6_prior_79), ("718", r6_prior_718))
        println(io, "==== prior bindingness — $tag ====")
        show(io, MIME"text/plain"(), t); println(io, "\n")
    end
    for (tag, s) in (("79", r6_sp_79), ("718", r6_sp_718))
        println(io, "==== λ spread vs market — $tag ====")
        println(io, "  n = $(s.n)  coverage = $(round(s.coverage, digits=4))")
        for (nm, q) in (("supremacy", s.supremacy), ("totals", s.totals))
            println(io, "  $nm: sd_model $(round(q.sd_model, digits=4))  sd_mkt $(round(q.sd_mkt, digits=4))",
                        "  ratio $(round(q.ratio, digits=3))  ρ $(round(q.ρ, digits=3))",
                        "  sd_opt $(round(q.sd_optimal, digits=4))  OVER $(round(q.over, digits=3))")
        end
        println(io, "  mean total: model $(round(s.mean_total_model, digits=3)) vs market $(round(s.mean_total_mkt, digits=3))\n")
    end
end
