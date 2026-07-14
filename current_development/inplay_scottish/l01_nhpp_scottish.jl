#=
l01_nhpp_scottish.jl — WP2 loader: Bayesian NHPP in-play intensity for 56/57.

Port of ../match_inplay_explore/l08_nhpp_turing.jl (Ireland-validated form) with two
extensions and one league-specific convention:
  + man_adv covariate (signed red-card advantage; 305 Scottish matches carry a red).
    Prior γ_man ~ Normal(log 1.2, 0.3) — centred on the market-implied Vecer/Kopriva/
    Ichiba 2009 opponent multiplier ×1.2, which matches the Ireland fit (+0.18). See
    RESEARCH.md §2.
  + expected_remaining_x returns PER-DRAW integrated remaining intensity (Λ_h, Λ_a)
    separately — the WP3 compose-posteriors entry point.
  ~ Stoppage clamping (r00 finding): this feed puts ALL stoppage goals at exactly
    mm=45 / mm=90 with added_time=0. With Δt=5 and Tend=95 the mm=90 mass lands in the
    [90,95) slice, whose 5-min exposure proxies league-mean H2 stoppage; mm=45 H1
    stoppage mass lands in [45,50) together with early-H2 goals — conflated, but the
    SAME clock convention holds at inference time, so fair value is consistent.

Likelihood (discretised NHPP == Poisson regression on Δt slices; exact as Δt→0):
  per (match × side × slice):  y ~ Poisson(exp(logλ + log Δt)),
  logλ = α + log pgλ_side + β·z(t) + γ_tr·trailing + γ_ld·leading + γ_man·man_adv
         [+ δ_time[bin]] [+ δ_state[gd bucket]]
AD-guide compliant: broadcast + view, config branches on constants, clamp + 1e-6.
=#

using Turing, Distributions, DataFrames, Statistics, LinearAlgebra, MCMCChains
using GLM, Random
const Samplers = BayesianFootball.Samplers

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
Base.@kwdef struct NHPPXConfig
    hier_time::Bool  = true
    hier_state::Bool = false      # Ireland verdict: linear state suffices
    Δt::Float64      = 5.0
    Tend::Float64    = 95.0       # mm=90-clamped stoppage mass lives in [90,95)
    σ_α::Float64     = 2.0
    σ_β::Float64     = 1.0
    man_prior_μ::Float64 = log(1.2)   # Vecer market-implied anchor (RESEARCH.md §2)
    man_prior_σ::Float64 = 0.3
    σ_time_prior::ContinuousUnivariateDistribution = truncated(Normal(0, 0.5); lower = 0.0)
    σ_state_prior::ContinuousUnivariateDistribution = truncated(Normal(0, 0.5); lower = 0.0)
end

# ---------------------------------------------------------------------------
# 2. Incident extraction (minimal copies of match_inplay_explore/l01 helpers)
# ---------------------------------------------------------------------------

_inc_minute(r)::Int = Int(r.time) +
    ((ismissing(r.added_time) || r.added_time == 999) ? 0 : Int(r.added_time))

"Goals of one match as sorted (t::Float64, home::Bool) tuples."
function goals_of(ds, mid)::Vector{@NamedTuple{t::Float64, home::Bool}}
    g = subset(ds.incidents, :match_id => ByRow(==(mid)),
                              :incident_type => ByRow(==("goal")))
    out = [(t = Float64(_inc_minute(r)), home = Bool(r.is_home)) for r in eachrow(g)
           if !ismissing(r.is_home)]
    sort!(out, by = x -> x.t)
    return out
end

"Red cards (straight red / second yellow) as sorted (t, home) tuples."
function reds_of(ds, mid)::Vector{@NamedTuple{t::Float64, home::Bool}}
    c = subset(ds.incidents, :match_id => ByRow(==(mid)),
                              :incident_type => ByRow(==("card")))
    isred(x) = !ismissing(x) && (occursin("red", lowercase(String(x))) ||
                                 occursin("yellowred", lowercase(String(x))))
    out = [(t = Float64(_inc_minute(r)), home = Bool(r.is_home)) for r in eachrow(c)
           if isred(r.incident_class) && !ismissing(r.is_home)]
    sort!(out, by = x -> x.t)
    return out
end

# ---------------------------------------------------------------------------
# 3. Assembly: DataStore + pregame latents -> per-match sequences
# ---------------------------------------------------------------------------

"""
    assemble_nhpp_matches(ds, latents_df, train_pairs) -> Vector{NamedTuple}

One entry per usable match: (mid, pgh, pga, goals, reds, home, away, tournament_id,
season). `train_pairs` :: Set of (tournament_id, season) allowed in training
(56's incident-hole seasons excluded). Pregame λ = posterior MEAN of the OOS draws
(the multiplier model conditions on it as an offset; per-draw pairing happens at
inference in WP3, not in training — cut-posterior convention, RESEARCH.md §3).
"""
function assemble_nhpp_matches(ds, latents_df::DataFrame, train_pairs)
    inc_mids = Set(unique(ds.incidents.match_id))
    lat = Dict(r.match_id => (mean(r.λ_h), mean(r.λ_a)) for r in eachrow(latents_df))
    out = NamedTuple[]
    for r in eachrow(ds.matches)
        (r.tournament_id, r.season) in train_pairs || continue
        haskey(lat, r.match_id) || continue
        r.match_id in inc_mids || continue
        pgh, pga = lat[r.match_id]
        (isfinite(pgh) && isfinite(pga) && pgh > 0 && pga > 0) || continue
        push!(out, (mid = Int(r.match_id), pgh = pgh, pga = pga,
                    goals = goals_of(ds, r.match_id), reds = reds_of(ds, r.match_id),
                    home = String(r.home_team), away = String(r.away_team),
                    tournament_id = Int(r.tournament_id), season = String(r.season)))
    end
    return out
end

# ---------------------------------------------------------------------------
# 4. Slice dataset (long format, one row per match × side × slice)
# ---------------------------------------------------------------------------

function build_slices(mseqs; Δt = 5.0, Tend = 95.0)
    edges = collect(0.0:Δt:Tend); nb = length(edges) - 1
    n = 2 * nb * length(mseqs)
    y = Vector{Int}(undef, n); off = Vector{Float64}(undef, n)
    z = similar(off); lpg = similar(off); tr = similar(off); ld = similar(off)
    man = similar(off); gsi = Vector{Int}(undef, n); tix = Vector{Int}(undef, n)
    mid = Vector{Int}(undef, n); ishome = Vector{Bool}(undef, n)
    k = 0
    for ms in mseqs
        for b in 1:nb
            lo, hi = edges[b], edges[b+1]; tmid = (lo + hi) / 2
            gh = count(g ->  g.home && g.t < lo, ms.goals)
            ga = count(g -> !g.home && g.t < lo, ms.goals)
            yh = count(g ->  g.home && lo <= g.t < hi, ms.goals)
            ya = count(g -> !g.home && lo <= g.t < hi, ms.goals)
            rh = count(c ->  c.home && c.t < lo, ms.reds)
            ra = count(c -> !c.home && c.t < lo, ms.reds)
            gd = gh - ga
            for (h, yc, pg, gds, mans) in ((true,  yh, ms.pgh,  gd, ra - rh),
                                           (false, ya, ms.pga, -gd, rh - ra))
                k += 1
                y[k] = yc; off[k] = log(Δt); z[k] = (tmid - 45) / 45
                lpg[k] = log(pg); tr[k] = Float64(gds < 0); ld[k] = Float64(gds > 0)
                man[k] = Float64(mans); gsi[k] = clamp(gds, -3, 3) + 4; tix[k] = b
                mid[k] = ms.mid; ishome[k] = h
            end
        end
    end
    DataFrame(y = y, off = off, z = z, log_pg = lpg, trailing = tr, leading = ld,
              man_adv = man, gs_idx = gsi, time_idx = tix, match_id = mid,
              is_home = ishome)
end

# ---------------------------------------------------------------------------
# 5. Turing model
# ---------------------------------------------------------------------------

@model function nhppx_intensity(y, offset, z, log_pg, trailing, leading, man_adv,
                                gs_idx, time_idx, n_states, n_timebins, config::NHPPXConfig)
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
    if config.hier_state
        σ_gs ~ config.σ_state_prior
        z_gs ~ filldist(Normal(0, 1), n_states)
        logλ = logλ .+ view(z_gs .* σ_gs, gs_idx)
    end
    μ = exp.(clamp.(logλ .+ offset, -20.0, 20.0)) .+ 1e-6
    @addlogprob! sum(logpdf.(Poisson.(μ), y))
end

make_nhppx_model(df::DataFrame, c::NHPPXConfig) =
    nhppx_intensity(df.y, df.off, df.z, df.log_pg, df.trailing, df.leading, df.man_adv,
                    df.gs_idx, df.time_idx, 7, Int(cld(c.Tend, c.Δt)), c)

# ---------------------------------------------------------------------------
# 6. GLM CV harness (fast workhorse for spec races; l07 pattern)
# ---------------------------------------------------------------------------

const SPEC_FORMULAS = Dict(
    :pg_only => @formula(y ~ 1 + log_pg),
    :time    => @formula(y ~ 1 + log_pg + z),
    :state   => @formula(y ~ 1 + log_pg + z + trailing + leading),
    :full    => @formula(y ~ 1 + log_pg + z + trailing + leading + man_adv),
)

function eval_fold(df, train_mask, spec::Symbol)
    tr = df[train_mask, :]; te = df[.!train_mask, :]
    m = glm(SPEC_FORMULAS[spec], tr, Poisson(), LogLink(); offset = tr.off)
    μ = GLM.predict(m, te; offset = te.off)
    mean(logpdf.(Poisson.(max.(μ, 1e-9)), te.y))
end

"Repeated k-fold (by MATCH) mean held-out per-row loglik for each spec."
function cv_race(df; specs = collect(keys(SPEC_FORMULAS)), k = 4, repeats = 5, seed = 42)
    mids = unique(df.match_id); rng = MersenneTwister(seed)
    rows = DataFrame(spec = Symbol[], rep = Int[], fold = Int[], loglik = Float64[])
    for rep in 1:repeats
        perm = shuffle(rng, mids)
        folds = [Set(perm[i:k:end]) for i in 1:k]
        for (fi, f) in enumerate(folds)
            train_mask = [!(m in f) for m in df.match_id]
            for s in specs
                push!(rows, (s, rep, fi, eval_fold(df, train_mask, s)))
            end
        end
    end
    return rows
end

"Paired per-fold difference a − b with a t-stat (l07 convention)."
function paired_diff(cv::DataFrame, a::Symbol, b::Symbol)
    da = sort(subset(cv, :spec => ByRow(==(a))), [:rep, :fold]).loglik
    db = sort(subset(cv, :spec => ByRow(==(b))), [:rep, :fold]).loglik
    d = da .- db
    (mean = mean(d), se = std(d) / sqrt(length(d)), t = mean(d) / (std(d) / sqrt(length(d))))
end

# ---------------------------------------------------------------------------
# 7. Per-draw remaining intensity (WP3 compose-posteriors entry point)
# ---------------------------------------------------------------------------

_cv(ch, s) = vec(Array(ch[s]))
_cm(ch, base, k) = reduce(hcat, _cv(ch, Symbol("$base[$i]")) for i in 1:k)
_has(ch, s) = any(occursin.(s, string.(MCMCChains.names(ch, :parameters))))

"""
    remaining_intensity(chain, c; pg_h, pg_a, gh, ga, reds_h, reds_a, t_now)
        -> (Λ_h::Vector, Λ_a::Vector)   one entry per posterior draw

Integrated remaining intensity per side from t_now to Tend, holding the current
score / man count fixed. In WP3, pg_h/pg_a vary per pregame draw (paired draws).
"""
function remaining_intensity(chain, c::NHPPXConfig; pg_h, pg_a, gh, ga,
                             reds_h = 0, reds_a = 0, t_now, Tend = c.Tend)
    αv = _cv(chain, :α); βv = _cv(chain, :β)
    gtr = _cv(chain, :γ_tr); gld = _cv(chain, :γ_ld); gman = _cv(chain, :γ_man)
    edges = collect(0.0:c.Δt:Tend); nb = length(edges) - 1
    zt = _has(chain, "z_time") ? (_cm(chain, :z_time, nb) .* _cv(chain, :σ_time)) : nothing
    zg = _has(chain, "z_gs")   ? (_cm(chain, :z_gs, 7)    .* _cv(chain, :σ_gs))   : nothing
    gd = gh - ga; man_h = Float64(reds_a - reds_h); man_a = -man_h
    Λh = zeros(length(αv)); Λa = zeros(length(αv))
    for b in 1:nb
        lo, hi = edges[b], edges[b+1]; hi <= t_now && continue
        tmid = (lo + hi) / 2; zc = (tmid - 45) / 45; dt = hi - max(lo, t_now)
        bh = αv .+ log(pg_h) .+ βv .* zc .+ gtr .* (gd < 0) .+ gld .* (gd > 0) .+ gman .* man_h
        ba = αv .+ log(pg_a) .+ βv .* zc .+ gtr .* (gd > 0) .+ gld .* (gd < 0) .+ gman .* man_a
        if zt !== nothing; bh = bh .+ zt[:, b]; ba = ba .+ zt[:, b]; end
        if zg !== nothing
            bh = bh .+ zg[:, clamp(gd, -3, 3) + 4]; ba = ba .+ zg[:, clamp(-gd, -3, 3) + 4]
        end
        Λh = Λh .+ exp.(bh) .* dt; Λa = Λa .+ exp.(ba) .* dt
    end
    return Λh, Λa
end
