# src/features/plus_minus/plus_minus.jl
#
# Regularized adjusted plus-minus (RAPM) player ratings, ported from
# current_development/plus_minus_ratings/ (WP0-WP7, green-lit 2026-07-23).
#
# WHY THIS EXISTS: Scottish League One (56) and League Two (57) have ZERO SofaScore player ratings,
# so the Ireland-family player engines cannot run there. RAPM builds a rating out of what those
# leagues DO have — lineups, incidents and BBC commentary. It was validated on tiers 54/55, where
# the SofaScore rating exists as a yardstick, using only lower-league-available features, so
# whatever passed there transfers verbatim to 56/57.
#
# THE VERDICT IT RESTS ON (`y_shots`, `w_SIM = 0`, λ = 1000, half-life 730d):
#   * split-half reliability 0.669 vs the SofaScore rating's 0.660 — PASS
#   * match-outcome retrodiction better than the SofaScore-fed model on both held-out seasons
#   * season-to-season stability ~0.29 against the base paper's own 0.35
#   * top-20 lists overlap 9-10/20 against a chance expectation of 1.2 (7-8x enrichment)
# with three honest limits: the SofaScore bar is low, the margin over a no-information floor is
# ~2%, and cross-tier transfer is thin (rho ~ 0.18-0.23). Goalkeeper ratings are worthless
# (rho ~ 0 vs SofaScore) — the APM engine's pillar collapses D+M+F and ignores G.
#
# FILE MAP
#   segments.jl    — the match clock, personnel segments, weights, competition sets, design matrix
#   shot_parser.jl — BBC commentary -> shot descriptors -> the zonal xG cell model
#   targets.jl     — the four y_* responses over those segments
#   ridge.jl       — similarity matrix, penalty matrix, the solve, `fit_ratings`
#
# The extractor that turns a rating vector into the 8-vector feature contract lives in
# src/features/extractors/plus_minus_extractors.jl.

include("segments.jl")
include("shot_parser.jl")
include("targets.jl")
include("ridge.jl")

# ==========================================
# THE PREPARED-SEGMENT CACHE
# ==========================================
"""
Everything that is a pure function of the DataStore and therefore fold-INDEPENDENT: the segment
table with all four `y_*` targets attached, and the per-match competition sets.

Rebuilding this per fold would dominate feature-building cost (segment construction plus shot
attribution over ~2k matches), and it would produce identical output every time. What IS
fold-dependent — which segments enter the ridge, and where the time decay is anchored — is applied
in the extractor.
"""
struct PMPrepared
    segments::DataFrame
    rejects::DataFrame
    comp_sets::Dict{Int, Dict{Int, Set{Int}}}
    it1_by_match::Dict{Int, Float64}
end

const PM_PREP_CACHE = Dict{Tuple{UInt, UInt, UInt}, PMPrepared}()
const PM_PREP_LOCK  = ReentrantLock()

_pm_cache_key(ds::Data.DataStore) =
    (objectid(ds.lineups), objectid(ds.incidents), objectid(ds.bbc_events))

"""
    pm_prepared(ds) -> PMPrepared

Build (or return the cached) segment table for this DataStore.

⚠ ON THE ONE THING FITTED GLOBALLY. The zonal shot-xG model (`fit_shot_xg`) is fitted over ALL
shots in the store, not per fold. It is a global lookup table of `P(goal | zone, body part,
context)` over ~40k attempts, carries no team or player identity, and it is exactly how the
research computed the `y_xg` target that the WP7 verdict was measured on — refitting it per fold
would mean the src ratings no longer reproduce the validated numbers. The *player ratings*
themselves remain strictly history-fit, which is the leak that would actually matter. This only
affects the `XGPlusMinusFeature` variant; the green-lit `ShotsPlusMinusFeature` never touches it.
"""
function pm_prepared(ds::Data.DataStore)
    key = _pm_cache_key(ds)
    lock(PM_PREP_LOCK) do
        haskey(PM_PREP_CACHE, key) && return PM_PREP_CACHE[key]

        segments, rejects = build_segments(ds)
        it1 = Dict{Int, Float64}()
        if nrow(ds.matches) > 0 && "injury_time1" in names(ds.matches)
            for r in eachrow(ds.matches)
                it1[Int(r.match_id)] = _pm_num(r.injury_time1)
            end
        end

        if nrow(segments) > 0
            shots = build_shots(ds)
            if nrow(shots) > 0
                xgm = fit_shot_xg(shots)
                shots.xg = predict_xg(xgm, shots)
            else
                shots.xg = Float64[]
            end
            add_targets!(segments, shots, it1)
        end

        prep = PMPrepared(segments, rejects,
                          nrow(segments) > 0 ? competition_sets(ds) :
                                               Dict{Int, Dict{Int, Set{Int}}}(),
                          it1)
        PM_PREP_CACHE[key] = prep
        return prep
    end
end

"""Drop the cache (use after mutating a DataStore in place)."""
pm_clear_cache!() = lock(PM_PREP_LOCK) do; empty!(PM_PREP_CACHE); end
