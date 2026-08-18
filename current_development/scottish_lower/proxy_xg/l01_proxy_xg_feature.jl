# current_development/scottish_proxy_xg/l01_proxy_xg_feature.jl
#
# LOADER 1/2 — team-match PROXY xG from BBC commentary, as a FeatureSet contract.
#
# The per-shot half of this already exists in src and is NOT reimplemented here:
#   src/features/plus_minus/shot_parser.jl
#     build_shots(ds)          -> one row per attempt, with (zone, body_part, context, is_penalty)
#     fit_shot_xg(shots; k)    -> saturated cell table P(goal | zone x body x context), EB-shrunk
#     predict_xg(model, shots) -> per-shot xG
# Measured by the plus-minus research at 98.4-99.8% parse coverage, Brier 11.1% better than base
# rate, and team-level correlation 0.817 vs SofaScore xG.
#
# What is NEW here is only the last step the repo never took: SUM those per-shot xG to a
# TEAM-MATCH total and hand it to a model as a Gamma-likelihood observation. Today the commentary
# xG only reaches an engine as an RAPM player rating (`XGPlusMinusFeature`), never as a team-match
# observation of its own.
#
# WHY THIS SHOULD CARRY SIGNAL (measured on bbc.live_text, tiers 56/57, 2026-08-04):
#   * zone conversion spans 0.048 (outside box) -> 0.182 (box centre) -> 0.444 (six yard)
#     -> 0.762 (penalty). A ~9x range, so shot MIX matters a lot per shot.
#   * cross-team spread in high-value-shot share (20 teams, >=40 team-matches each):
#     observed SD 0.036 vs binomial SD 0.016 => 4.9x EXCESS VARIANCE, range 0.178.
#     Team shot quality is real, not sampling noise.
#   * but a crude 2-bucket estimate puts the axis at only ~+/-4% on the scoring rate, against the
#     funnel's +/-10.7% team-strength SD. Expect a SECOND-ORDER effect. See NOTES.md.
#
# COVERAGE (hard constraint, verified): BBC live text on 56/57 is 100% from 23/24 and ZERO before.
# 1,070 covered matches. That is the same restriction `XGPlusMinusFeature` already lives under
# (see src/features/plus_minus/targets.jl:25-31), so it is not a new limitation — but it does mean
# the pillar is masked off on any pre-23/24 history a fold pulls in.

using DataFrames
using Statistics
using Distributions

# Module aliases for the whole stream. l02 includes THIS file, so a runner includes exactly one of
# the two loaders (l01 for the data/EDA runners, l02 for the model runners) and never both.
const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Pred     = BayesianFootball.Predictions
const Data     = BayesianFootball.Data

# ==========================================
# 1. THE FEATURE CONFIG
# ==========================================
"""
    ProxyXGFeature(; k = 25.0, fit_on = :global)

Team-match proxy xG summed from BBC commentary shot events.

`k` is the empirical-Bayes pseudo-count of the zonal cell table (a cell with `k` shots is pulled
halfway to the base rate); 25.0 is the research value.

`fit_on` picks which matches the CELL TABLE may be fitted on:
  :global   (default) — every shot in the store. Matches what `pm_prepared` already does for the
                        `y_xg` target (src/features/plus_minus/plus_minus.jl:64-71) and keeps the
                        two xG routes numerically consistent. The cell table carries no team or
                        player identity — it is a league-wide `P(goal | zone, body, context)` over
                        ~19.5k attempts — so a fold's own ~25 shots move it by ~0.1%.
  :training           — refit per fold on `ordered_ids` only (history u target, exactly the
                        engine's own information set). Strictly leak-free; kept so the difference
                        can be MEASURED rather than asserted.
  :history            — the frozen history block only (needs `F_data[:history_match_ids]`).
"""
Base.@kwdef struct ProxyXGFeature <: Features.AbstractFeatureConfig
    k::Float64     = 25.0
    fit_on::Symbol = :global
end

# ==========================================
# 2. CACHES
# ==========================================
# `build_shots` parses ~19.5k commentary strings. With 40 folds x 5 cells that is 200 repeats of
# identical work, so the PARSE is cached on the events table's identity. The cell-table fit is a
# groupby and is cheap, but the fully-scored per-match table is cached too, keyed on the fit set,
# so `fit_on = :global` costs exactly one pass for a whole grid.
const PXG_SHOT_CACHE  = Dict{UInt, DataFrame}()
const PXG_TABLE_CACHE = Dict{Tuple{UInt, Float64, UInt}, Tuple{Dict{Int, NTuple{4, Float64}}, Set{Int}}}()
const PXG_LOCK        = ReentrantLock()

"""Drop both caches (use after mutating a DataStore in place, e.g. a Betfair odds swap)."""
pxg_clear_cache!() = lock(PXG_LOCK) do
    empty!(PXG_SHOT_CACHE); empty!(PXG_TABLE_CACHE)
end

function _pxg_shots(ds::Data.DataStore)
    key = objectid(ds.bbc_events)
    lock(PXG_LOCK) do
        haskey(PXG_SHOT_CACHE, key) && return PXG_SHOT_CACHE[key]
        sh = Features.build_shots(ds)
        PXG_SHOT_CACHE[key] = sh
        return sh
    end
end

# ==========================================
# 3. THE AGGREGATION
# ==========================================
"""
    proxy_xg_table(ds; k = 25.0, fit_ids = nothing) -> (lut, covered)

`lut :: Dict{match_id => (xg_h, xg_a, n_h, n_a)}` and `covered :: Set{match_id}` (matches with any
BBC shot event at all — the same notion as `segments.covered`).

`fit_ids === nothing` fits the cell table on every shot; otherwise on shots from those matches
only. Scoring always covers ALL shots, so an out-of-fold match still gets an xG — it is the cell
table, not the coverage, that `fit_ids` restricts.

Sides that resolved ZERO shots are NOT emitted as a genuine 0.0. Every one of the 2,137 team-match
sides measured on 56/57 had at least one attempt, so a zero here means the ~2.4% team-slug
attribution failure, not a team that never shot. The caller masks those sides off.
"""
function proxy_xg_table(ds::Data.DataStore; k::Float64 = 25.0,
                        fit_ids::Union{Nothing, Set{Int}} = nothing)
    ck = (objectid(ds.bbc_events), k, fit_ids === nothing ? UInt(0) : hash(fit_ids))
    lock(PXG_LOCK) do
        haskey(PXG_TABLE_CACHE, ck) && return PXG_TABLE_CACHE[ck]

        shots = _pxg_shots(ds)
        if nrow(shots) == 0
            empty = (Dict{Int, NTuple{4, Float64}}(), Set{Int}())
            PXG_TABLE_CACHE[ck] = empty
            return empty
        end

        fit_rows = if fit_ids === nothing
            shots
        else
            sub = shots[in.(Int.(shots.match_id), Ref(fit_ids)), :]
            # An empty or tiny fit set would give a degenerate table; fall back rather than emit
            # nonsense. 500 attempts ~ 25 matches.
            nrow(sub) < 500 ? shots : sub
        end

        xgm = Features.fit_shot_xg(fit_rows; k = k)
        sc  = copy(shots)
        sc.xg = Features.predict_xg(xgm, sc)

        covered = Set(Int.(sc.match_id))
        sc = sc[.!ismissing.(sc.is_home), :]          # drop the unattributable ~2.4%

        lut = Dict{Int, NTuple{4, Float64}}()
        for r in eachrow(sc)
            mid = Int(r.match_id)
            c   = get(lut, mid, (0.0, 0.0, 0.0, 0.0))
            lut[mid] = (r.is_home === true) ?
                       (c[1] + r.xg, c[2], c[3] + 1.0, c[4]) :
                       (c[1], c[2] + r.xg, c[3], c[4] + 1.0)
        end

        out = (lut, covered)
        PXG_TABLE_CACHE[ck] = out
        return out
    end
end

# ==========================================
# 4. THE EXTRACTOR
# ==========================================
"""
    add_feature!(F_data, ::ProxyXGFeature, ordered_ids, team_map, ds)

Emits, aligned to `ordered_ids`:
  :flat_home_xg_proxy / :flat_away_xg_proxy    Float64 — team-match proxy xG
  :flat_home_pxg_shots / :flat_away_pxg_shots  Int     — the EVENT shot count the xG was summed
                                                         over (Arm B conditions on this; it is NOT
                                                         the same series as ds.bbc's match-page
                                                         shotsTotal — see r00 gate 2)
  :flat_pxg_mask_h / :flat_pxg_mask_a          Float64 — 1.0 usable, 0.0 otherwise

AD-safety contract (docs/turing_ad_performance_guide.md): an unusable slot gets a DUMMY 1.0 — a
valid point in the Gamma support — together with a 0.0 mask. Never `missing`, never `NaN`, and
never a 0.0 (Gamma support is x > 0, so `logpdf(Gamma, 0) = -Inf` and `-Inf * 0.0 == NaN` would
poison the entire gradient).

Present-but-tiny values are floored to 1e-3 for the same reason, mirroring
src/.../outfield_xg_double_poisson.jl:177-186.

Degrades to an all-zero mask (never an error) on a segment with no BBC events, so the engine falls
back to goals-only rather than failing. ⚠ Do not actually run it that way — with no xG only the
product kappa*mu is identified.
"""
function Features.add_feature!(F_data::Dict, config::ProxyXGFeature, ordered_ids,
                               team_map::Dict, ds::Data.DataStore)
    n = length(ordered_ids)
    xg_h = fill(1.0, n); xg_a = fill(1.0, n)     # DUMMY, paired with a 0.0 mask
    sh_h = zeros(Int, n); sh_a = zeros(Int, n)
    mk_h = zeros(Float64, n); mk_a = zeros(Float64, n)

    fit_ids = if config.fit_on === :global
        nothing
    elseif config.fit_on === :training
        Set(Int.(ordered_ids))
    elseif config.fit_on === :history
        haskey(F_data, :history_match_ids) || error(
            "ProxyXGFeature(fit_on = :history) needs F_data[:history_match_ids]; " *
            "the builder stashes it (src/features/builder.jl), a hand-rolled F_data must too.")
        F_data[:history_match_ids]::Set{Int}
    else
        error("Unknown fit_on = $(config.fit_on); expected :global, :training or :history")
    end

    lut, covered = proxy_xg_table(ds; k = config.k, fit_ids = fit_ids)

    if !isempty(lut)
        for (i, id) in enumerate(ordered_ids)
            mid = Int(id)
            mid in covered || continue
            rec = get(lut, mid, nothing); rec === nothing && continue
            x_h, x_a, n_h, n_a = rec
            if n_h > 0
                xg_h[i] = max(x_h, 1e-3); sh_h[i] = round(Int, n_h); mk_h[i] = 1.0
            end
            if n_a > 0
                xg_a[i] = max(x_a, 1e-3); sh_a[i] = round(Int, n_a); mk_a[i] = 1.0
            end
        end
    end

    F_data[:flat_home_xg_proxy]   = xg_h
    F_data[:flat_away_xg_proxy]   = xg_a
    F_data[:flat_home_pxg_shots]  = sh_h
    F_data[:flat_away_pxg_shots]  = sh_a
    F_data[:flat_pxg_mask_h]      = mk_h
    F_data[:flat_pxg_mask_a]      = mk_a
    return nothing
end

# ==========================================
# 5. EDA CONVENIENCE — long team-match frame
# ==========================================
"""
    proxy_team_rows(ds; k = 25.0) -> DataFrame

Two rows per covered match (one per side) with everything the WP1 EDA needs:
`match_id, kickoff, season, team, opponent, is_home, goals, goals_against, shots_bbc,
shots_events, pxg, pxg_against`.

`shots_bbc` comes from `ds.bbc` (BBC match-page totals, all six seasons); `shots_events` is the
commentary event count the xG was actually summed over. Keeping BOTH is the point — r00 gate 2
reconciles them, and the two engines use different ones on purpose.
"""
function proxy_team_rows(ds::Data.DataStore; k::Float64 = 25.0)
    lut, covered = proxy_xg_table(ds; k = k)
    isempty(lut) && return DataFrame()

    mt = ds.matches
    for c in (:match_id, :home_team, :away_team, :match_date)
        string(c) in names(mt) || error("ds.matches is missing $(c); cannot build EDA rows")
    end

    bbc = ds.bbc
    bbc_lut = Dict{Int, NamedTuple}()
    if nrow(bbc) > 0
        for r in eachrow(bbc)
            bbc_lut[Int(r.match_id)] = (
                gh = r.home_score, ga = r.away_score, sh = r.shots_h, sa = r.shots_a,
            )
        end
    end

    _num(x) = (ismissing(x) || (x isa AbstractFloat && isnan(x))) ? missing : Float64(x)

    rows = NamedTuple[]
    for r in eachrow(mt)
        mid = Int(r.match_id)
        mid in covered || continue
        haskey(lut, mid)     || continue
        haskey(bbc_lut, mid) || continue
        x_h, x_a, n_h, n_a = lut[mid]
        (n_h > 0 && n_a > 0) || continue
        b = bbc_lut[mid]
        gh, ga = _num(b.gh), _num(b.ga)
        (gh === missing || ga === missing) && continue
        season = "season" in names(mt) ? String(r.season) : "?"

        push!(rows, (match_id = mid, kickoff = r.match_date, season = season,
                     team = String(r.home_team), opponent = String(r.away_team), is_home = 1.0,
                     goals = gh, goals_against = ga,
                     shots_bbc = _num(b.sh), shots_bbc_against = _num(b.sa),
                     shots_events = n_h, shots_events_against = n_a,
                     pxg = x_h, pxg_against = x_a))
        push!(rows, (match_id = mid, kickoff = r.match_date, season = season,
                     team = String(r.away_team), opponent = String(r.home_team), is_home = 0.0,
                     goals = ga, goals_against = gh,
                     shots_bbc = _num(b.sa), shots_bbc_against = _num(b.sh),
                     shots_events = n_a, shots_events_against = n_h,
                     pxg = x_a, pxg_against = x_h))
    end
    return DataFrame(rows)
end
