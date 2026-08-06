#=
l07_nowcast.jl — WP-D loader: MVP-2, condition remaining GOAL intensity on shots so far.

    log λ(t) = [incumbent terms] + γ_sf · surplus(t)
    surplus(t) = (shots_so_far(t) − E[shots by t]) / E[shots by t]

with `E[shots by t]` taken from the PREGAME shot intensity, so the term measures in-match
DEVIATION only — team strength is already absorbed by the pregame offset and must not be
re-learnt here.

Unlike MVP-1 this keeps goals as the counting process; shots enter purely as a covariate.
So it is the incumbent plus exactly one term, which is what makes Gate D a clean test.

CAUSALITY. `shots_so_far` counts strictly BEFORE the slice start (`s.t < lo`), identical to
how `l01.build_slices` builds `man_adv`. Any leak here manufactures edge, and this repo has
already been bitten once by an as-of fill that produced e^2.44/match ≈ ×10^161.

WHY E[shots by t] CAN BE SPREAD OVER EXPOSURE. WP-C measured the shot time-profile as
nearly flat — `t_time` 1.04 at shot resolution, β = 0.048 ± 0.040, σ_time = 0.079 — so
apportioning the pregame shot total by cumulative EXPOSURE (which already carries per-match
stoppage via `bin_exposure`) is accurate to a few percent. That is a finding of this stream
being reused, not an assumption.

THE PRE-REGISTERED CONFOUND. A positive γ_sf may mean the model is learning in-match
information, OR it may just be correcting a pregame λ that was mis-estimated for this match
— shots-above-expectation is exactly what a too-low pregame λ looks like. The two are
distinguishable: if it is a pregame correction, freeing the coefficient on `log_pg` (which
lets the model rescale the pregame rate directly) should absorb it and γ_sf should collapse.
`free_pg` in the config exists for that test.

Requires l01, l04, l05 and l06 (for `bin_exposure`) to be included first.
=#

using Turing, Distributions, DataFrames, Statistics, MCMCChains, GLM, Random

# ---------------------------------------------------------------------------
# 1. Surplus
# ---------------------------------------------------------------------------

# Below this many expected shots the ratio is dominated by its own denominator (at t=0 it
# is 0/0), so the surplus is held at 0 — "no information yet" — rather than exploding.
const SURPLUS_MIN_EXPECTED = 1.0
# Cap leverage: a side with 4× its expected shots is already an extreme, and beyond that
# the ratio adds leverage rather than information.
const SURPLUS_CLAMP = (-1.0, 3.0)

"""
    expected_shots_by(λ_s, expo, b) -> Float64

Pregame-expected shots for one side by the START of bin `b`, apportioning the pregame shot
total `λ_s` across the match by cumulative exposure.
"""
function expected_shots_by(λ_s::Float64, expo::Vector{Float64}, b::Int)
    tot = sum(expo)
    tot <= 0 && return 0.0
    return λ_s * sum(view(expo, 1:(b - 1))) / tot
end

"""
    surplus_of(shots_before, expected) -> Float64

`(observed − expected) / expected`, floored and clamped as documented above.
"""
function surplus_of(shots_before::Int, expected::Float64)
    expected < SURPLUS_MIN_EXPECTED && return 0.0
    return clamp((shots_before - expected) / expected, SURPLUS_CLAMP...)
end

# ---------------------------------------------------------------------------
# 2. Slices — the incumbent's goal slices plus the surplus covariate
# ---------------------------------------------------------------------------

"""
    build_nowcast_slices(mseqs; Δt = 5.0, Tend = 90.0) -> DataFrame

`build_slices_bbc` plus three columns:

  * `surplus`     — own shots vs own pregame expectation, before the slice
  * `surplus_opp` — the opponent's, which is a different signal: a side being OUT-shot may
                    face more chances rather than create fewer
  * `shots_before` / `exp_shots` — kept raw for diagnostics and the leak audit

Every count is taken strictly before `lo`.
"""
function build_nowcast_slices(mseqs; Δt::Float64 = 5.0, Tend::Float64 = 90.0)
    base = build_slices_bbc(mseqs; Δt = Δt, Tend = Tend)
    edges = collect(0.0:Δt:Tend); nb = length(edges) - 1
    n = nrow(base)
    sur = zeros(n); sur_o = zeros(n); sb = zeros(Int, n); es = zeros(n)
    k = 0
    for ms in mseqs
        expo = bin_exposure(ms.at1, ms.at2; Δt = Δt, Tend = Tend)
        haveshots = ms.shots !== nothing && ms.λsh !== nothing
        for b in 1:nb
            lo = edges[b]
            if haveshots
                sh_h = count(s ->  s.home && s.t < lo, ms.shots)
                sh_a = count(s -> !s.home && s.t < lo, ms.shots)
                eh = expected_shots_by(ms.λsh, expo, b)
                ea = expected_shots_by(ms.λsa, expo, b)
            else
                sh_h = sh_a = 0; eh = ea = 0.0
            end
            for (h, own, opp, e_own, e_opp) in ((true,  sh_h, sh_a, eh, ea),
                                                (false, sh_a, sh_h, ea, eh))
                k += 1
                sur[k]   = surplus_of(own, e_own)
                sur_o[k] = surplus_of(opp, e_opp)
                sb[k] = own; es[k] = e_own
            end
        end
    end
    base.surplus = sur; base.surplus_opp = sur_o
    base.shots_before = sb; base.exp_shots = es
    return base
end

# The nowcast specs, added to l01's SPEC_FORMULAS for the CV race.
const NOWCAST_FORMULAS = Dict(
    :nowcast     => @formula(y ~ 1 + log_pg + z + trailing + leading + man_adv + surplus),
    :nowcast_opp => @formula(y ~ 1 + log_pg + z + trailing + leading + man_adv +
                                 surplus + surplus_opp),
    # Pregame rate FIXED at 1 (as the Turing model has it) rather than free. If γ_sf is
    # really a pregame correction, it should be BIGGER here, where it has more to correct.
    :nowcast_fixed_pg => @formula(y ~ 1 + z + trailing + leading + man_adv + surplus),
    :full_fixed_pg    => @formula(y ~ 1 + z + trailing + leading + man_adv),
)

"CV fold evaluation against an arbitrary formula dict (l01's `eval_fold` is closed over its own)."
function eval_fold_f(df, train_mask, formula, offset_col::Symbol)
    tr = df[train_mask, :]; te = df[.!train_mask, :]
    m = glm(formula, tr, Poisson(), LogLink(); offset = tr[!, offset_col])
    μ = GLM.predict(m, te; offset = te[!, offset_col])
    mean(logpdf.(Poisson.(max.(μ, 1e-9)), te.y))
end

"""
    nowcast_cv(df; k = 4, repeats = 5, seed = 42) -> DataFrame

Repeated k-fold by MATCH over l01's specs plus the nowcast specs. The `*_fixed_pg` specs
carry `log_pg` inside the offset so the pregame rate is genuinely fixed at 1.
"""
function nowcast_cv(df::DataFrame; k::Int = 4, repeats::Int = 5, seed::Int = 42)
    df = copy(df)
    df.off_pg = df.off .+ df.log_pg          # offset with the pregame rate folded in
    specs = merge(Dict(k2 => v for (k2, v) in SPEC_FORMULAS), NOWCAST_FORMULAS)
    fixed = Set([:nowcast_fixed_pg, :full_fixed_pg])
    mids = unique(df.match_id); rng = MersenneTwister(seed)
    rows = DataFrame(spec = Symbol[], rep = Int[], fold = Int[], loglik = Float64[])
    for rep in 1:repeats
        perm = shuffle(rng, mids)
        folds = [Set(perm[i:k:end]) for i in 1:k]
        for (fi, f) in enumerate(folds)
            train_mask = [!(m in f) for m in df.match_id]
            for (s, form) in specs
                push!(rows, (s, rep, fi,
                    eval_fold_f(df, train_mask, form, s in fixed ? :off_pg : :off)))
            end
        end
    end
    return rows
end

# ---------------------------------------------------------------------------
# 2b. MATCH-CLUSTERED CV — the harness correction this stream needs
# ---------------------------------------------------------------------------
#
# `l01.paired_diff` differences the 20 (repeat, fold) means and divides by their SD. That
# treats 20 numbers as independent observations when (a) each fold mean already averages
# ~137 held-out matches, so its variance is ~1/137 of the match-level variance, and (b) the
# 5 repeats re-score THE SAME 550 matches. The independent unit is the MATCH.
#
# Measured inflation on WP-D: `nowcast_opp − full` scores t = 3.17 fold-paired and
# **t = 1.07 match-clustered** — a factor of ~3, larger than the ~1.9–2× this repo had
# previously measured elsewhere. Every fold-paired t in this stream (including the
# incumbent's published 6.29 / 0.51 / 8.85) is overstated by roughly this factor. It does
# not flip any published sign, but magnitudes must be read accordingly.
#
# Use `match_clustered_cv` + `mc_diff` for anything that has to survive a significance
# claim; `cv_race` + `paired_diff` remain fine as a fast spec-screening workhorse.

"""
    match_clustered_cv(df, specs, fixed_pg_specs; k = 4, repeats = 5, seed = 42)
        -> Dict{Symbol, Dict{Int, Float64}}

Per-spec, per-MATCH mean held-out log-likelihood, averaged over repeats. Folds are by match,
so a match is only ever scored by a model that never saw it.
"""
function match_clustered_cv(df::DataFrame, specs::Dict, fixed_pg_specs::Set;
                            k::Int = 4, repeats::Int = 5, seed::Int = 42)
    df = copy(df); df.off_pg = df.off .+ df.log_pg
    mids = unique(df.match_id); rng = MersenneTwister(seed)
    acc = Dict(s => Dict{Int, Vector{Float64}}() for s in keys(specs))
    for rep in 1:repeats
        perm = shuffle(rng, mids); folds = [Set(perm[i:k:end]) for i in 1:k]
        for f in folds
            tm = [!(m in f) for m in df.match_id]
            tr = df[tm, :]; te = df[.!tm, :]
            for (s, form) in specs
                oc = s in fixed_pg_specs ? :off_pg : :off
                m = glm(form, tr, Poisson(), LogLink(); offset = tr[!, oc])
                μ = GLM.predict(m, te; offset = te[!, oc])
                ll = logpdf.(Poisson.(max.(μ, 1e-9)), te.y)
                for g in groupby(DataFrame(mid = te.match_id, ll = ll), :mid)
                    push!(get!(acc[s], g.mid[1], Float64[]), mean(g.ll))
                end
            end
        end
    end
    return Dict(s => Dict(m => mean(v) for (m, v) in d) for (s, d) in acc)
end

"Paired spec difference with the MATCH as the unit of independence."
function mc_diff(mc::Dict, a::Symbol, b::Symbol)
    ms = collect(keys(mc[a]))
    d = [mc[a][m] - mc[b][m] for m in ms]
    (mean = mean(d), se = std(d) / sqrt(length(d)),
     t = mean(d) / (std(d) / sqrt(length(d))), n = length(d))
end

# ---------------------------------------------------------------------------
# 3. Turing model
# ---------------------------------------------------------------------------

Base.@kwdef struct NowcastConfig
    hier_time::Bool = true
    use_opp::Bool   = false     # add the opponent-surplus term
    free_pg::Bool   = false     # CONFOUND TEST: free the pregame rate coefficient
    Δt::Float64     = 5.0
    Tend::Float64   = 90.0
    σ_α::Float64    = 2.0
    σ_β::Float64    = 1.0
    man_prior_μ::Float64 = log(1.2)
    man_prior_σ::Float64 = 0.3
    σ_sf::Float64   = 0.5
    σ_time_prior::ContinuousUnivariateDistribution = truncated(Normal(0, 0.5); lower = 0.0)
end

@model function nowcast_intensity(y, offset, z, log_pg, trailing, leading, man_adv,
                                  surplus, surplus_opp, time_idx, n_timebins,
                                  config::NowcastConfig)
    α     ~ Normal(0, config.σ_α)
    β     ~ Normal(0, config.σ_β)
    γ_tr  ~ Normal(0, 0.5)
    γ_ld  ~ Normal(0, 0.5)
    γ_man ~ Normal(config.man_prior_μ, config.man_prior_σ)
    γ_sf  ~ Normal(0, config.σ_sf)
    # ρ is the pregame-rate exponent: 1 = the cut-Bayes offset, free = let the in-play
    # module rescale the pregame belief. Centred on 1 so "fixed" is the null.
    ρ = if config.free_pg
        ρ_ ~ Normal(1.0, 0.3); ρ_
    else
        1.0
    end
    logλ = α .+ ρ .* log_pg .+ β .* z .+ γ_tr .* trailing .+ γ_ld .* leading .+
           γ_man .* man_adv .+ γ_sf .* surplus
    if config.use_opp
        γ_sfo ~ Normal(0, config.σ_sf)
        logλ = logλ .+ γ_sfo .* surplus_opp
    end
    if config.hier_time
        σ_time ~ config.σ_time_prior
        z_time ~ filldist(Normal(0, 1), n_timebins)
        logλ = logλ .+ view(z_time .* σ_time, time_idx)
    end
    μ = exp.(clamp.(logλ .+ offset, -20.0, 20.0)) .+ 1e-6
    @addlogprob! sum(logpdf.(Poisson.(μ), y))
end

make_nowcast_model(df::DataFrame, c::NowcastConfig) =
    nowcast_intensity(df.y, df.off, df.z, df.log_pg, df.trailing, df.leading, df.man_adv,
                      df.surplus, df.surplus_opp, df.time_idx,
                      Int(cld(c.Tend, c.Δt)), c)

# ---------------------------------------------------------------------------
# 4. Remaining intensity + calibration
# ---------------------------------------------------------------------------

"""
    remaining_intensity_nowcast(chain, c; pg_h, pg_a, λs_h, λs_a, shots_h, shots_a,
                                gh, ga, reds_h, reds_a, t_now, at1, at2) -> (Λ_h, Λ_a)

As `l01.remaining_intensity`, plus the surplus term evaluated at `t_now` and held fixed
over the remainder — the state is what is known NOW, exactly as the score and man count are.
"""
function remaining_intensity_nowcast(chain, c::NowcastConfig; pg_h, pg_a, λs_h, λs_a,
                                     shots_h, shots_a, gh, ga, reds_h = 0, reds_a = 0,
                                     t_now, at1 = L04_DEFAULT_AT1, at2 = L04_DEFAULT_AT2)
    αv = _cv(chain, :α); βv = _cv(chain, :β)
    gtr = _cv(chain, :γ_tr); gld = _cv(chain, :γ_ld); gman = _cv(chain, :γ_man)
    gsf = _cv(chain, :γ_sf)
    ρv = _has(chain, "ρ_") ? _cv(chain, :ρ_) : ones(length(αv))
    edges = collect(0.0:c.Δt:c.Tend); nb = length(edges) - 1
    expo = bin_exposure(at1, at2; Δt = c.Δt, Tend = c.Tend)
    zt = _has(chain, "z_time") ? (_cm(chain, :z_time, nb) .* _cv(chain, :σ_time)) : nothing
    b_now = clamp(Int(fld(t_now, c.Δt)) + 1, 1, nb)
    s_h = surplus_of(shots_h, expected_shots_by(λs_h, expo, b_now))
    s_a = surplus_of(shots_a, expected_shots_by(λs_a, expo, b_now))
    gd = gh - ga; man_h = Float64(reds_a - reds_h); man_a = -man_h
    Λh = zeros(length(αv)); Λa = zeros(length(αv))
    for b in 1:nb
        lo, hi = edges[b], edges[b+1]; hi <= t_now && continue
        tmid = (lo + hi) / 2; zc = (tmid - 45) / 45
        dt = expo[b] * (hi - max(lo, t_now)) / (hi - lo)
        bh = αv .+ ρv .* log(pg_h) .+ βv .* zc .+ gtr .* (gd < 0) .+ gld .* (gd > 0) .+
             gman .* man_h .+ gsf .* s_h
        ba = αv .+ ρv .* log(pg_a) .+ βv .* zc .+ gtr .* (gd > 0) .+ gld .* (gd < 0) .+
             gman .* man_a .+ gsf .* s_a
        if zt !== nothing; bh = bh .+ zt[:, b]; ba = ba .+ zt[:, b]; end
        Λh = Λh .+ exp.(bh) .* dt; Λa = Λa .+ exp.(ba) .* dt
    end
    return Λh, Λa
end

"""
    nowcast_checkpoint_bias(mseqs, chain, c; checkpoints) -> DataFrame

Gate D's second half. A significant γ_sf with FLAT calibration means the term is absorbing
something else, so this must move together with the CV result.

`by_surplus` splits the bias by surplus tercile: if the term carries real information the
bias should be flat ACROSS terciles, not merely zero on average.
"""
function nowcast_checkpoint_bias(mseqs, chain, c::NowcastConfig;
                                 checkpoints = (60.0, 75.0, 85.0))
    rows = DataFrame(t0 = Float64[], pred = Float64[], real = Float64[], surplus = Float64[])
    for ms in mseqs, t0 in checkpoints
        (ms.shots === nothing || ms.λsh === nothing) && continue
        gh = count(g ->  g.home && g.t < t0, ms.goals)
        ga = count(g -> !g.home && g.t < t0, ms.goals)
        rh = count(x ->  x.home && x.t < t0, ms.reds)
        ra = count(x -> !x.home && x.t < t0, ms.reds)
        sh = count(s ->  s.home && s.t < t0, ms.shots)
        sa = count(s -> !s.home && s.t < t0, ms.shots)
        Λh, Λa = remaining_intensity_nowcast(chain, c; pg_h = ms.pgh, pg_a = ms.pga,
            λs_h = ms.λsh, λs_a = ms.λsa, shots_h = sh, shots_a = sa,
            gh = gh, ga = ga, reds_h = rh, reds_a = ra, t_now = t0,
            at1 = ms.at1, at2 = ms.at2)
        expo = bin_exposure(ms.at1, ms.at2; Δt = c.Δt, Tend = c.Tend)
        b_now = clamp(Int(fld(t0, c.Δt)) + 1, 1, length(expo))
        stot = surplus_of(sh + sa, expected_shots_by(ms.λsh + ms.λsa, expo, b_now))
        push!(rows, (t0, mean(Λh) + mean(Λa),
                     Float64(count(g -> g.t >= t0, ms.goals)), stot))
    end
    overall = combine(groupby(rows, :t0), nrow => :n,
        [:pred, :real] => ((p, r) -> mean(p .- r)) => :bias,
        [:pred, :real] => ((p, r) -> std(p .- r) / sqrt(length(p))) => :se)
    rows.tercile = let q = quantile(rows.surplus, [1/3, 2/3])
        [s <= q[1] ? "low" : s <= q[2] ? "mid" : "high" for s in rows.surplus]
    end
    by_surplus = combine(groupby(rows, [:t0, :tercile]), nrow => :n,
        [:pred, :real] => ((p, r) -> mean(p .- r)) => :bias,
        [:pred, :real] => ((p, r) -> std(p .- r) / sqrt(length(p))) => :se)
    return overall, sort(by_surplus, [:t0, :tercile]), rows
end
