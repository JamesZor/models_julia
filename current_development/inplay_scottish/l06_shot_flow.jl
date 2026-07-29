#=
l06_shot_flow.jl — WP-C loader: MVP-1, shots as the NHPP and goals as Poisson thinning.

    Shots ~ Poisson(λ_s(t)·Δt)
    log λ_s = α_s + log(pregame λ_s_side) + β_s·z + γ_tr·trailing + γ_ld·leading
              + γ_man·man_adv + δ_time[bin]
    Goals | Shots ~ Binomial(Shots, p2)

WHY THIS MIGHT BEAT THE INCUMBENT. The incumbent's game-state term was a clean null on
Scottish data (`state − time` paired t = 0.51, against Ireland's 3.15) — but it was measured
on ~1.4 goals per team-match. Pregame, the funnel resolved team strength 1.8× better than
goals-only precisely by counting shots instead, and there are ~9.7 shots per team-match here
(r04b: λ_s_h 9.72, p2 0.149). **7× the counts.** So the incumbent null may be a resolution
failure rather than a real absence, and this is the instrument that can tell the difference.

WHY THE COMPOSER STILL WORKS UNCHANGED. Poisson thinning preserves the goals marginal:
if shots are Poisson(Λ_s) and each converts independently with probability p2, goals are
Poisson(p2·Λ_s). So `l02`'s kernel simply becomes `p2 · K_s_side` and every market prices
exactly as before — no new score-matrix type, no new market code.

The pregame offset is `λ_s_h` / `λ_s_a` straight from the funnel latents (`l05`), NOT a
refit; and `p2` likewise. Requires l01 (config, CV harness), l04 (clock + exposure) and
l05 (`assemble_matches`) to be included first.
=#

using Turing, Distributions, DataFrames, Statistics, MCMCChains, GLM, Random

# ---------------------------------------------------------------------------
# 1. Slice dataset — shots as the counting process
# ---------------------------------------------------------------------------

"""
    build_shot_slices(mseqs; Δt = 5.0, Tend = 90.0, state_p2 = false) -> DataFrame

One row per match × side × slice, mirroring `build_slices_bbc` column-for-column so the
same `SPEC_FORMULAS` / `cv_race` / `paired_diff` harness runs unchanged — except that `y`
counts SHOTS and `log_pg` is the pregame SHOT intensity.

`y_goals` is carried alongside for the thinning likelihood and the conversion checks; it is
not a regressor.

Game state (`trailing` / `leading` / `man_adv`) is built from events strictly BEFORE the
slice start, identical to `l01.build_slices`. That discipline is what keeps the covariates
causal; any leak here manufactures edge.
"""
function build_shot_slices(mseqs; Δt::Float64 = 5.0, Tend::Float64 = 90.0)
    edges = collect(0.0:Δt:Tend); nb = length(edges) - 1
    n = 2 * nb * length(mseqs)
    y = Vector{Int}(undef, n); yg = Vector{Int}(undef, n); off = Vector{Float64}(undef, n)
    z = similar(off); lpg = similar(off); lpgg = similar(off)
    tr = similar(off); ld = similar(off); man = similar(off)
    gsi = Vector{Int}(undef, n); tix = Vector{Int}(undef, n)
    mid = Vector{Int}(undef, n); ishome = Vector{Bool}(undef, n)
    team = Vector{String}(undef, n); p2v = similar(off)
    k = 0
    for ms in mseqs
        (ms.shots === nothing || ms.λsh === nothing) && continue
        expo = bin_exposure(ms.at1, ms.at2; Δt = Δt, Tend = Tend)
        for b in 1:nb
            lo, hi = edges[b], edges[b+1]; tmid = (lo + hi) / 2
            gh = count(g ->  g.home && g.t < lo, ms.goals)
            ga = count(g -> !g.home && g.t < lo, ms.goals)
            rh = count(c ->  c.home && c.t < lo, ms.reds)
            ra = count(c -> !c.home && c.t < lo, ms.reds)
            gd = gh - ga
            sh_h = count(s ->  s.home && lo <= s.t < hi, ms.shots)
            sh_a = count(s -> !s.home && lo <= s.t < hi, ms.shots)
            gl_h = count(g ->  g.home && lo <= g.t < hi, ms.goals)
            gl_a = count(g -> !g.home && lo <= g.t < hi, ms.goals)
            for (h, sc, gc, pgs, pgg, gds, mans, tm) in
                    ((true,  sh_h, gl_h, ms.λsh, ms.pgh,  gd, ra - rh, ms.home),
                     (false, sh_a, gl_a, ms.λsa, ms.pga, -gd, rh - ra, ms.away))
                k += 1
                y[k] = sc; yg[k] = gc; off[k] = log(expo[b]); z[k] = (tmid - 45) / 45
                lpg[k] = log(pgs); lpgg[k] = log(pgg)
                tr[k] = Float64(gds < 0); ld[k] = Float64(gds > 0)
                man[k] = Float64(mans); gsi[k] = clamp(gds, -3, 3) + 4; tix[k] = b
                mid[k] = ms.mid; ishome[k] = h; team[k] = tm; p2v[k] = ms.p2
            end
        end
    end
    DataFrame(y = y[1:k], y_goals = yg[1:k], off = off[1:k], z = z[1:k],
              log_pg = lpg[1:k], log_pg_goals = lpgg[1:k],
              trailing = tr[1:k], leading = ld[1:k], man_adv = man[1:k],
              gs_idx = gsi[1:k], time_idx = tix[1:k], match_id = mid[1:k],
              is_home = ishome[1:k], team = team[1:k], p2 = p2v[1:k])
end

# ---------------------------------------------------------------------------
# 2. Turing model — shot intensity, optionally with state-dependent conversion
# ---------------------------------------------------------------------------

Base.@kwdef struct ShotFlowConfig
    hier_time::Bool  = true
    state_p2::Bool   = false      # let conversion, not just volume, shift with game state
    Δt::Float64      = 5.0
    Tend::Float64    = 90.0
    σ_α::Float64     = 2.0
    σ_β::Float64     = 1.0
    man_prior_μ::Float64 = log(1.2)
    man_prior_σ::Float64 = 0.3
    σ_time_prior::ContinuousUnivariateDistribution = truncated(Normal(0, 0.5); lower = 0.0)
end

@model function shotflow_intensity(y, y_goals, offset, z, log_pg, trailing, leading,
                                   man_adv, time_idx, logit_p2, n_timebins,
                                   config::ShotFlowConfig)
    α     ~ Normal(0, config.σ_α)
    β     ~ Normal(0, config.σ_β)
    γ_tr  ~ Normal(0, 0.5)
    γ_ld  ~ Normal(0, 0.5)
    γ_man ~ Normal(config.man_prior_μ, config.man_prior_σ)
    logλ = α .+ log_pg .+ β .* z .+ γ_tr .* trailing .+ γ_ld .* leading .+
           γ_man .* man_adv
    if config.hier_time
        σ_time ~ config.σ_time_prior
        z_time ~ filldist(Normal(0, 1), n_timebins)
        logλ = logλ .+ view(z_time .* σ_time, time_idx)
    end
    μ = exp.(clamp.(logλ .+ offset, -20.0, 20.0)) .+ 1e-6
    @addlogprob! sum(logpdf.(Poisson.(μ), y))

    # Conversion. The pregame p2 enters as a logit offset so the fitted terms are pure
    # DEVIATIONS from what the funnel already believes — a null here is then a statement
    # that in-play conversion does not move, not that conversion is unmodelled.
    if config.state_p2
        κ_0  ~ Normal(0, 0.5)
        κ_tr ~ Normal(0, 0.5)
        κ_ld ~ Normal(0, 0.5)
        lp = logit_p2 .+ κ_0 .+ κ_tr .* trailing .+ κ_ld .* leading
        p = 1 ./ (1 .+ exp.(.-clamp.(lp, -20.0, 20.0)))
        @addlogprob! sum(logpdf.(Binomial.(y, clamp.(p, 1e-6, 1 - 1e-6)), y_goals))
    end
end

function make_shotflow_model(df::DataFrame, c::ShotFlowConfig)
    lp2 = log.(clamp.(df.p2, 1e-6, 1 - 1e-6) ./ (1 .- clamp.(df.p2, 1e-6, 1 - 1e-6)))
    shotflow_intensity(df.y, df.y_goals, df.off, df.z, df.log_pg, df.trailing, df.leading,
                       df.man_adv, df.time_idx, lp2, Int(cld(c.Tend, c.Δt)), c)
end

# ---------------------------------------------------------------------------
# 3. Remaining GOAL intensity via thinning
# ---------------------------------------------------------------------------

"""
    remaining_shot_intensity(chain, c; pg_s_h, pg_s_a, gh, ga, reds_h, reds_a, t_now,
                             at1, at2) -> (Λ_s_h, Λ_s_a)   per posterior draw

Integrated remaining SHOT intensity per side. Multiply by `p2` for the goal intensity —
Poisson thinning keeps the goals marginal Poisson, so `l02`'s composer takes
`K_side = p2 · K_s_side` and needs no other change.

Exposure uses the match's own stoppage (`at1`, `at2`), matching `bin_exposure`.
"""
function remaining_shot_intensity(chain, c::ShotFlowConfig; pg_s_h, pg_s_a, gh, ga,
                                  reds_h = 0, reds_a = 0, t_now,
                                  at1 = L04_DEFAULT_AT1, at2 = L04_DEFAULT_AT2)
    αv = _cv(chain, :α); βv = _cv(chain, :β)
    gtr = _cv(chain, :γ_tr); gld = _cv(chain, :γ_ld); gman = _cv(chain, :γ_man)
    edges = collect(0.0:c.Δt:c.Tend); nb = length(edges) - 1
    expo = bin_exposure(at1, at2; Δt = c.Δt, Tend = c.Tend)
    zt = _has(chain, "z_time") ? (_cm(chain, :z_time, nb) .* _cv(chain, :σ_time)) : nothing
    gd = gh - ga; man_h = Float64(reds_a - reds_h); man_a = -man_h
    Λh = zeros(length(αv)); Λa = zeros(length(αv))
    for b in 1:nb
        lo, hi = edges[b], edges[b+1]; hi <= t_now && continue
        tmid = (lo + hi) / 2; zc = (tmid - 45) / 45
        # scale this bin's exposure by the fraction of it still to be played
        dt = expo[b] * (hi - max(lo, t_now)) / (hi - lo)
        bh = αv .+ log(pg_s_h) .+ βv .* zc .+ gtr .* (gd < 0) .+ gld .* (gd > 0) .+ gman .* man_h
        ba = αv .+ log(pg_s_a) .+ βv .* zc .+ gtr .* (gd > 0) .+ gld .* (gd < 0) .+ gman .* man_a
        if zt !== nothing; bh = bh .+ zt[:, b]; ba = ba .+ zt[:, b]; end
        Λh = Λh .+ exp.(bh) .* dt; Λa = Λa .+ exp.(ba) .* dt
    end
    return Λh, Λa
end

"""
    shot_checkpoint_bias(mseqs, chain, c, p2_of; checkpoints) -> DataFrame

The incumbent's headline diagnostic, re-run on the thinned goal intensity: predicted
remaining GOALS vs realised, at 60/75/85′. Incumbent scored +0.020 / +0.014 / +0.003.
"""
function shot_checkpoint_bias(mseqs, chain, c::ShotFlowConfig;
                              checkpoints = (60.0, 75.0, 85.0))
    rows = DataFrame(t0 = Float64[], pred = Float64[], real = Float64[])
    for ms in mseqs, t0 in checkpoints
        (ms.shots === nothing || ms.λsh === nothing) && continue
        gh = count(g ->  g.home && g.t < t0, ms.goals)
        ga = count(g -> !g.home && g.t < t0, ms.goals)
        rh = count(cc ->  cc.home && cc.t < t0, ms.reds)
        ra = count(cc -> !cc.home && cc.t < t0, ms.reds)
        Λh, Λa = remaining_shot_intensity(chain, c; pg_s_h = ms.λsh, pg_s_a = ms.λsa,
                                          gh = gh, ga = ga, reds_h = rh, reds_a = ra,
                                          t_now = t0, at1 = ms.at1, at2 = ms.at2)
        push!(rows, (t0, ms.p2 * (mean(Λh) + mean(Λa)),
                     Float64(count(g -> g.t >= t0, ms.goals))))
    end
    combine(groupby(rows, :t0), nrow => :n,
            [:pred, :real] => ((p, r) -> mean(p .- r)) => :bias,
            [:pred, :real] => ((p, r) -> std(p .- r) / sqrt(length(p))) => :se)
end
