#=
l05_pregame_source.jl — WP-B loader: a pluggable pregame posterior source.

`r01`/`r02` read their pregame λ draws from one hard-coded serialized file,
`data/scottish_decay_grid/latents_hl365_hs2.jls`. That artifact is a dead end — it carries
only `λ_h` / `λ_a`, and the grid that produced it has been superseded by the funnel family.
The funnel engines additionally emit **`λ_s_h`, `λ_s_a` (pregame SHOT intensities) and
`p2` (conversion)**, which are exactly the offsets a shot-flow in-play model needs, so the
in-play stack has to be able to swap pregame engines without editing the runners.

Two sources implement `pregame_draws`:

  * `ExperimentSource`  — a real `ExperimentResults` directory, via
    `Experiments.load_experiment` → `extract_oos_predictions`. Carries the shot latents.
  * `LatentsFileSource` — the legacy `.jls`, kept ONLY so the incumbent's published r01–r03
    numbers stay reproducible. Its `λ_s_h` / `λ_s_a` / `p2` are `nothing`, and anything
    needing them must say so rather than silently substituting.

`extract_oos_predictions` re-creates every feature set and walks every split, so it is
minutes, not seconds. Results are cached to `out/pregame_<name>.jls`; delete that file to
force a re-extract.

Requires `l01_nhpp_scottish.jl` (for `goals_of` / `reds_of`) and `l04_bbc_timeline.jl`
(for the stoppage defaults) to be included first.

⚠ The DataStore passed in must have ALL NINE fields — `extract_oos_predictions` on an APM
arm reads `ds.bbc_events` to rebuild the plus-minus ratings, and the 7-argument DataStore
idiom drops it, zeroing the ratings with no error. `load_datastore_cached` is fine.
=#

using DataFrames, Serialization, Statistics

const Exp = BayesianFootball.Experiments

abstract type AbstractPregameSource end

"""
    ExperimentSource(path[, name])

An `ExperimentResults` directory, e.g.
`data/experiments/plus_minus/funnel_apm_xg_20260728_213335`.
"""
struct ExperimentSource <: AbstractPregameSource
    path::String
    name::String
end
ExperimentSource(path::String) = ExperimentSource(path, basename(rstrip(path, '/')))

"""
    LatentsFileSource(path[, name])

The legacy serialized latents DataFrame (`λ_h`, `λ_a` only). Present so the incumbent arm
stays byte-reproducible against its published numbers; not a candidate for new work.
"""
struct LatentsFileSource <: AbstractPregameSource
    path::String
    name::String
end
LatentsFileSource(path::String) = LatentsFileSource(path, "latents_file")

source_name(s::AbstractPregameSource) = s.name

# The known artifacts, so runners name an engine rather than a timestamp. `funnel_apm_xg`
# and `funnel_winner` are the monthly-dynamics arms — the pair the APM stream measured as a
# tie (t = −0.57). The `plus_minus_biweek/` reruns are a different fold cadence; do not mix.
const PREGAME_ROOT = joinpath(dirname(dirname(@__DIR__)), "data", "experiments", "plus_minus")

"""
    known_source(engine; root = PREGAME_ROOT) -> ExperimentSource

Resolve an engine name to its most recent artifact directory (timestamps sort lexically).
"""
function known_source(engine::String; root::String = PREGAME_ROOT)
    isdir(root) || error("pregame artifact root not found: $root")
    c = filter(d -> startswith(d, engine * "_"), readdir(root))
    isempty(c) && error("no artifact for '$engine' under $root")
    return ExperimentSource(joinpath(root, maximum(c)), engine)
end

# ---------------------------------------------------------------------------
# 2. The interface
# ---------------------------------------------------------------------------

const PregameDraws = Dict{Int, @NamedTuple{
    λ_h::Vector{Float64}, λ_a::Vector{Float64},
    λ_s_h::Union{Nothing, Vector{Float64}}, λ_s_a::Union{Nothing, Vector{Float64}},
    p2::Union{Nothing, Vector{Float64}}}}

_vec(x) = x === nothing ? nothing : Vector{Float64}(collect(x))

"""
    pregame_draws(src, ds; cache_dir = "out", refresh = false) -> PregameDraws

`match_id => (λ_h, λ_a, λ_s_h, λ_s_a, p2)`, each a vector of posterior draws (OOS only).
The three shot fields are `nothing` for sources that do not carry them.
"""
function pregame_draws(src::ExperimentSource, ds;
                       cache_dir::String = joinpath(@__DIR__, "out"),
                       refresh::Bool = false)
    mkpath(cache_dir)
    cf = joinpath(cache_dir, "pregame_$(src.name).jls")
    (!refresh && isfile(cf)) && return deserialize(cf)::PregameDraws

    exp = Exp.load_experiment(src.path)
    df = Exp.extract_oos_predictions(ds, exp).df
    out = PregameDraws()
    hasshots = all(c -> hasproperty(df, c), (:λ_s_h, :λ_s_a, :p2))
    for r in eachrow(df)
        out[Int(r.match_id)] = (
            λ_h = _vec(r.λ_h), λ_a = _vec(r.λ_a),
            λ_s_h = hasshots ? _vec(r.λ_s_h) : nothing,
            λ_s_a = hasshots ? _vec(r.λ_s_a) : nothing,
            p2    = hasshots ? _vec(r.p2)    : nothing)
    end
    serialize(cf, out)
    return out
end

function pregame_draws(src::LatentsFileSource, ds; kwargs...)
    df = deserialize(abspath(src.path))
    out = PregameDraws()
    for r in eachrow(df)
        out[Int(r.match_id)] = (λ_h = _vec(r.λ_h), λ_a = _vec(r.λ_a),
                                λ_s_h = nothing, λ_s_a = nothing, p2 = nothing)
    end
    return out
end

"True when the source carries the pregame shot intensities MVP-1/2/3 need."
has_shot_latents(d::PregameDraws) =
    !isempty(d) && first(values(d)).λ_s_h !== nothing

# ---------------------------------------------------------------------------
# 3. Assembly — the single entry point every downstream WP uses
# ---------------------------------------------------------------------------

"""
    assemble_matches(ds, draws, train_pairs; seqs = nothing, require_incidents = true,
                     reconcile = true) -> Vector{NamedTuple}

Generalises `l01.assemble_nhpp_matches` along two axes the race needs:

  * **event source** — `seqs = nothing` reads goals/reds from `ds.incidents` on `l01`'s
    clock (race arm 0a); passing the `build_event_seqs` dict reads them from BBC instead,
    and additionally supplies `shots`, `corners` and per-match stoppage (arms 0b, 1–3).
  * **pregame source** — any `AbstractPregameSource`, carrying full draw vectors rather
    than the posterior mean, so the composer can pair draws.

Each entry: `(mid, pgh, pga, λh_draws, λa_draws, λsh, λsa, p2, goals, reds, shots,
corners, at1, at2, home, away, tournament_id, season)`.

`pgh` / `pga` are posterior MEANS — the multiplier model conditions on the pregame rate as
a fixed offset while training (cut-posterior convention, RESEARCH.md §3); per-draw pairing
happens at inference, not here.

`reconcile = true` drops matches whose BBC goal counts disagree with the final score. Gate
A found exactly 2 such matches in 1,070, both single-match feed defects.
"""
function assemble_matches(ds, draws::PregameDraws, train_pairs;
                          seqs = nothing, require_incidents::Bool = true,
                          reconcile::Bool = true)
    inc_mids = Set(unique(ds.incidents.match_id))
    out = NamedTuple[]
    for r in eachrow(ds.matches)
        (r.tournament_id, r.season) in train_pairs || continue
        mid = Int(r.match_id)
        haskey(draws, mid) || continue
        (require_incidents && !(r.match_id in inc_mids)) && continue

        d = draws[mid]
        pgh = mean(d.λ_h); pga = mean(d.λ_a)
        (isfinite(pgh) && isfinite(pga) && pgh > 0 && pga > 0) || continue

        if seqs === nothing
            g = goals_of(ds, r.match_id); rd = reds_of(ds, r.match_id)   # from l01
            sh = nothing; co = nothing; at1 = L04_DEFAULT_AT1; at2 = L04_DEFAULT_AT2
        else
            haskey(seqs, mid) || continue
            s = seqs[mid]
            g, rd, sh, co, at1, at2 = s.goals, s.reds, s.shots, s.corners, s.at1, s.at2
            if reconcile && !ismissing(r.home_score) && !ismissing(r.away_score)
                (count(x -> x.home, g) == r.home_score &&
                 count(x -> !x.home, g) == r.away_score) || continue
            end
        end

        push!(out, (mid = mid, pgh = pgh, pga = pga,
                    λh_draws = d.λ_h, λa_draws = d.λ_a,
                    λsh = d.λ_s_h === nothing ? nothing : mean(d.λ_s_h),
                    λsa = d.λ_s_a === nothing ? nothing : mean(d.λ_s_a),
                    p2  = d.p2   === nothing ? nothing : mean(d.p2),
                    goals = g, reds = rd, shots = sh, corners = co,
                    at1 = at1, at2 = at2,
                    home = String(r.home_team), away = String(r.away_team),
                    tournament_id = Int(r.tournament_id), season = String(r.season)))
    end
    return out
end

"Coverage report for a pregame source against the DataStore."
function draws_qa(draws::PregameDraws, ds)
    mids = Set(keys(draws))
    m = subset(ds.matches, :match_id => ByRow(x -> x in mids))
    byseason = sort(combine(groupby(m, [:tournament_id, :season]), nrow => :matches),
                    [:tournament_id, :season])
    λh = [mean(v.λ_h) for v in values(draws)]; λa = [mean(v.λ_a) for v in values(draws)]
    nd = length(first(values(draws)).λ_h)
    sh = has_shot_latents(draws)
    (matches = length(draws), draws_per_match = nd, by_season = byseason,
     λ_h = (mean = mean(λh), min = minimum(λh), max = maximum(λh)),
     λ_a = (mean = mean(λa), min = minimum(λa), max = maximum(λa)),
     has_shot_latents = sh,
     λ_s_h = sh ? mean(mean(v.λ_s_h) for v in values(draws)) : NaN,
     λ_s_a = sh ? mean(mean(v.λ_s_a) for v in values(draws)) : NaN,
     p2 = sh ? mean(mean(v.p2) for v in values(draws)) : NaN)
end
