# src/features/extractors/bbc_extractors.jl
#
# Extractors reading the `ds.bbc` domain (BBC match-page counts; see src/Data/fetchers/sql/bbc.jl).

# --- ShotsFunnelFeature -------------------------------------------------------------------
#
# Two-layer funnel inputs: per-side total shots, aligned to `ordered_ids`.
#
# AD-safety contract (docs/turing_ad_performance_guide.md): the model never sees `missing` or
# `NaN`. Absent counts are emitted as a 0 DUMMY together with a 0.0 mask, so the masked slot is
# evaluated on valid inputs (0 events at rate λ_s weighted by 0) and contributes exactly nothing
# to the log-likelihood or its gradient. Post-hoc masking of an invalid value would give
# `-Inf * 0.0 == NaN` and poison the whole gradient.
#
# `sot_*` is deliberately NOT read: r06/Stage-2 closed the shots-on-target layer, so only the
# clean Shots -> Goals form graduated. The columns still live in `ds.bbc` for later work.
#
# Segments outside the Scottish tiers have an empty `ds.bbc`; the mask is then all-zero and the
# funnel engine degrades to a goals-only Poisson rather than erroring.
function add_feature!(F_data::Dict, ::ShotsFunnelFeature, ordered_ids, team_map::Dict,
                      ds::Data.DataStore)
    n = length(ordered_ids)
    shots_h = zeros(Int, n)
    shots_a = zeros(Int, n)
    mask_h  = zeros(Float64, n)
    mask_a  = zeros(Float64, n)

    bbc = ds.bbc
    has_bbc = nrow(bbc) > 0 && all(c -> string(c) in names(bbc), (:match_id, :shots_h, :shots_a))

    if has_bbc
        lut = Dict{Int, Tuple{Any, Any}}(
            Int(r.match_id) => (r.shots_h, r.shots_a) for r in eachrow(bbc)
        )

        # usable = present, finite and non-negative
        _ok(x) = !ismissing(x) && isfinite(float(x)) && float(x) >= 0

        for (i, id) in enumerate(ordered_ids)
            haskey(lut, Int(id)) || continue
            sh, sa = lut[Int(id)]
            if _ok(sh)
                shots_h[i] = round(Int, float(sh)); mask_h[i] = 1.0
            end
            if _ok(sa)
                shots_a[i] = round(Int, float(sa)); mask_a[i] = 1.0
            end
        end
    end

    F_data[:flat_home_shots_n]  = shots_h
    F_data[:flat_away_shots_n]  = shots_a
    F_data[:flat_funnel_mask_h] = mask_h
    F_data[:flat_funnel_mask_a] = mask_a
end
