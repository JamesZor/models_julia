# src/MatchDay/inference.jl
#
# Stages 4 and 6: materialise features for fixtures that are in no training fold, then extract
# posterior latents for them using a trained chain.
#
# This is the part of the prototype that was cleverest and most fragile. Two things are fixed
# here.
#
# Q11 -- SPLIT SELECTION. The prototype picked its chain with
#     last_split_idx = length(experiment.training_results)
# and then indexed TODAY's freshly rebuilt feature collection with that same integer. Measured
# on src_sup40_sw40: 29 training_results, 31 boundaries when rebuilt. So it silently conditioned
# on split 29 of 31 -- discarding the two most recent windows -- and was only *correct* at all
# if the splitter appends rather than recomputes. `select_split` names the split by its
# BOUNDARY and refuses when the pairing cannot be justified.
#
# FEATURE MATERIALISATION. The prototype mutated a cached FeatureSet's :player_ratings_map in
# place, which (a) mutates a cache and (b) has no path for any feature other than ratings --
# so no market-pillar engine could ever run on match day. Materialisers dispatch per feature.

export select_split, matchday_latents, RatingsFromTracker, LineupAggregateFromRAPM,
       LeagueFromFixture, MarketPillarFromBook, MaterialiserChain, check_coverage

"""
    INJECTABLE_KEYS

`FeatureSet.data` entries that are **per-match lookup maps**, which is what
`extract_parameters` indexes for a fixture it has never seen. Everything else in a `FeatureSet`
is a flat training-time array (`:flat_home_goals`, `:flat_market_λ_home`, ...) and is not read
at inference.

Both entries here are read as `get(map, match_id, <default>)`, so a fixture missing from either
is priced silently rather than refused -- which is why every one of them must be materialised
*and* covered by `check_coverage`.

* `:player_ratings_map` -- read by every player-level engine
  (`DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel` and friends); default `Dict()`,
  i.e. zero player strength.
* `:player_lineup_ratings_map` -- read by the **builder** family's `PlayerLineupPillar`
  (`Models.PreGame`'s `engine.jl`: `get(d, :player_lineup_ratings_map, Dict{Int,PMLineupAggregate}())`,
  consumed by `predictor_oos(::PlayerLineupPillar, ...)`); default
  `_pm_empty_lineup_aggregate()`, i.e. **the lineup pillar contributes exactly zero**. This key
  was absent from this tuple until 2026-09-04, which made the omission invisible in the worst
  possible way: `m12_joint_hybrid_synergy` would have priced a full Scottish card with no lineup
  term at all, and nothing would have raised, because the fallback is a valid value rather than a
  missing one.
* `:league_lookup` -- read by every pooled multi-division engine
  (`DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel`, `...PlusMinus...`,
  `...SmileLeague...`) as `get(league_lookup, mid, 0)`; default `0`, which **zeroes the
  zero-sum δ_league offset**. On `ScottishLower [56, 57]` that offset is precisely the goal-level
  gap between League One and League Two, so an unmaterialised fixture is priced at the mean of
  the two tiers.
"""
const INJECTABLE_KEYS = (:player_ratings_map, :player_lineup_ratings_map, :league_lookup)

"""
    check_coverage(fs, fixtures, model)

Assert that every fixture is actually representable, and say precisely what is missing when it
is not.

This checks **per-fixture coverage**, not per-feature presence. `haskey(fs.data,
:player_ratings_map)` is true straight out of training and tells you nothing about today; a
fixture absent from that map is priced off `get(ratings_map, mid, Dict())`, i.e. an empty
rating vector, silently.
"""
function check_coverage(fs, fx::Vector{Fixture}, model)
    data = fs.data
    problems = String[]

    if haskey(data, :team_map)
        tm = data[:team_map]
        unknown = unique(vcat([f.home for f in fx if !haskey(tm, f.home)],
                              [f.away for f in fx if !haskey(tm, f.away)]))
        isempty(unknown) || push!(problems,
            "teams absent from team_map (would be priced as league-average): " *
            join(unknown, ", "))
    end

    if haskey(data, :player_ratings_map)
        rm = data[:player_ratings_map]
        missing_ids = [f.m_id for f in fx if !haskey(rm, f.m_id)]
        isempty(missing_ids) || push!(problems,
            "fixtures with no entry in player_ratings_map (extract_parameters would fall back " *
            "to an empty Dict and price them with zero player strength): " *
            join(string.(missing_ids), ", "))
    end

    if haskey(data, :player_lineup_ratings_map)
        lm = data[:player_lineup_ratings_map]
        missing_ids = [f.m_id for f in fx if !haskey(lm, f.m_id)]
        isempty(missing_ids) || push!(problems,
            "fixtures with no entry in player_lineup_ratings_map (PlayerLineupPillar's " *
            "predictor_oos would fall back to the neutral aggregate, contributing exactly zero " *
            "lineup effect while still producing a plausible-looking price): " *
            join(string.(missing_ids), ", "))
    end

    if haskey(data, :league_lookup)
        ll = data[:league_lookup]
        missing_ids = [f.m_id for f in fx if !haskey(ll, f.m_id)]
        isempty(missing_ids) || push!(problems,
            "fixtures with no entry in league_lookup (extract_parameters would fall back to " *
            "league index 0 and zero the δ_league offset, pricing them at the mean of the " *
            "pooled divisions): " * join(string.(missing_ids), ", "))
    end

    isempty(problems) && return true
    error("MatchDay.check_coverage: " * join(problems, " | "))
end

"""
    select_split(expr, boundaries; strict = true, exclude = nothing) -> (idx, chain, warning)

Choose which trained split to condition on.

The chain at index `i` was fitted on the boundary that had index `i` **at training time**. If the
boundary list has since grown, the correspondence survives only when the splitter appends.

There are three rules, tried in this order.

**1. `ds` + `config` + `fixture_ids` — POSITIVE IDENTIFICATION. Prefer this.**
`Data.get_next_matches(ds, boundaries[i], config)` returns the next observed block that fold `i`
was built to predict. (For pooled calendar clocks this can skip blank period labels.) So the fold
for a match day is simply the one whose *next block is this card*, and the answer is keyed on
`(target_season, time_step)` — stable facts about
the fold — rather than on its position in a list that gets recomputed. This is the same call
`Experiments.post_processing` uses to generate out-of-sample predictions, which is what makes
train and serve consistent by construction instead of by coincidence.

**2. `exclude` — NEGATIVE FALLBACK.** The most recent fold whose target window contains none of
the ids being priced. Used when rule 1 cannot answer, which is the normal LIVE case: an unplayed
fixture is not in `ds.matches` at all, so no fold's `get_next_matches` can contain it.

**3. Positional.** `min(n_trained, n_bounds)`, the original behaviour.

Why rules 1 and 2 exist, measured on `ScottishUpper` 2026-08-09:

```
fold   targets   last target date   fixtures being priced, inside
  2        10        2026-08-02       0     <- correct choice
  3        22        2026-08-09       6     <- what min(n_trained, n_bounds) picked
```

The DataStore cache had been rebuilt, so the splitter recomputed fold 3's target window and it
grew to swallow the very card being priced. Both counts were 3, so the positional rule selected
it **and the mismatch warning never fired** — the failure was completely silent, and the
FeatureSet handed to `extract_parameters` would have been built over a window containing the
results.

Returns the index used, the chain, and a warning string (empty when nothing is suspicious).
"""
function select_split(expr, boundaries; strict::Bool = true, exclude = nothing,
                      ds = nothing, config = nothing, fixture_ids = nothing)
    n_trained = length(expr.training_results)
    n_bounds  = length(boundaries)
    n_trained == 0 && error("experiment has no training results")

    idx  = min(n_trained, n_bounds)
    warn = ""

    if n_bounds != n_trained
        warn = "boundary count has changed since training ($(n_trained) trained, " *
               "$(n_bounds) rebuilt): conditioning on split $(idx) of $(n_bounds), so the " *
               "$(n_bounds - idx) most recent window(s) are NOT used. This is only correct " *
               "if the splitter appends rather than recomputes its windows."
        strict && @warn "MatchDay.select_split: $warn"
    end

    # --- rule 1: the fold whose NEXT round is this card ------------------------------------
    if ds !== nothing && config !== nothing && fixture_ids !== nothing && !isempty(fixture_ids)
        want = Set(fixture_ids)
        for i in idx:-1:1
            nxt = try
                Data.get_next_matches(ds, boundaries[i], config)
            catch
                continue          # metadata shape this splitter does not support; fall through
            end
            (isempty(nxt) || !hasproperty(nxt, :match_id)) && continue
            hit = length(intersect(want, Set(nxt.match_id)))
            hit == 0 && continue
            i == idx || (w = "split $(idx) is the most recent trained fold, but the card being " *
                             "priced is the NEXT round of split $(i) (matched $hit of " *
                             "$(length(want)) fixtures via get_next_matches); conditioning on " *
                             "$(i).";
                         warn = isempty(warn) ? w : warn * " | " * w;
                         strict && @info "MatchDay.select_split: $w")
            return (idx = i, chain = expr.training_results[i][1], warning = warn)
        end
    end

    # --- rule 2: the most recent fold that has not already seen this card -------------------
    if exclude !== nothing && !isempty(exclude)
        ex   = Set(exclude)
        safe = idx
        while safe >= 1 && !isempty(intersect(ex, Set(boundaries[safe][1].target_match_ids)))
            safe -= 1
        end
        safe == 0 && error(
            "MatchDay.select_split: EVERY fold from 1..$(idx) has at least one of the " *
            "$(length(ex)) fixtures being priced inside its TARGET window. There is no chain " *
            "here that has not already seen this card. Retrain with a target season ending " *
            "before these fixtures.")
        if safe != idx
            w = "split $(idx) contains $(length(intersect(ex, Set(boundaries[idx][1].target_match_ids)))) " *
                "of the fixtures being priced in its target window; stepping back to split " *
                "$(safe), whose window is clear of them. This normally means the DataStore " *
                "cache was rebuilt after training and the splitter regrew the last fold."
            warn = isempty(warn) ? w : warn * " | " * w
            strict && @warn "MatchDay.select_split: $w"
        end
        idx = safe
    end

    chain = expr.training_results[idx][1]
    return (idx = idx, chain = chain, warning = warn)
end

# ===================================================================
# Materialisers
# ===================================================================

"""
    RatingsFromTracker()

Rolls every player's rating history forward to `as_of` using the model's own tracker, then
aggregates the starting XI into per-side positional sums -- which is the shape the model
consumes.

Delegates the recursion to `Features.calculate_player_ratings` rather than reimplementing it.
The prototype carried its own copy of every tracker's update rule; they were consistent for
`BayesianTracker`, `EWMATracker` and `WindowAverageTracker`, but `LastValueTracker` diverged
whenever a player's most recent rating was missing, and the prototype's generic
`::AbstractRatingTracker` fallback had no counterpart in `src/features` -- so an unknown tracker
failed loudly at training and silently at serving. Calling the training-time function makes
train/serve skew unrepresentable rather than merely untested.

The "latest" rating is `calculate_player_ratings(tracker, history)` evaluated one step past the
end, which is exactly the value training would assign to the player's next match.
"""
struct RatingsFromTracker <: AbstractFeatureMaterialiser end

function materialise!(::RatingsFromTracker, ::Val{:player_ratings_map}, fs,
                      fx::Vector{Fixture}, ctx)
    ds, model = ctx.ds, ctx.model
    # DECLINE, do not throw. A builder-family `PoissonCountModel` carries its player term inside
    # `covariates` as a `PlayerLineupPillar` and has no `player_ratings_feature` at all; reading
    # the field raised a `FieldError` that took down the whole match day. Returning `false` lets
    # `MaterialiserChain` fall through to `LineupAggregateFromRAPM`, and a key that NO member
    # claims is still an error in `matchday_latents` -- so declining here cannot silently skip a
    # feature, only redirect it.
    hasproperty(model, :player_ratings_feature) || return false
    tracker = model.player_ratings_feature.tracker
    latest, fallback = latest_player_ratings(ds, tracker)

    map_ = fs.data[:player_ratings_map]
    for f in fx
        lu = get(ctx.lineups, f.m_id, nothing)
        lu === nothing && continue
        entry = Dict{Tuple{String,String},Float64}()
        for (side, players) in (("home", lu.home), ("away", lu.away))
            sums = Dict("G" => 0.0, "D" => 0.0, "M" => 0.0, "F" => 0.0)
            for p in players
                p.substitute && continue
                sums[String(p.position)] += get(latest, p.player_id, fallback)
            end
            for (pos, v) in sums
                entry[(side, pos)] = v
            end
        end
        map_[f.m_id] = entry
    end
    return true
end

materialise!(::RatingsFromTracker, ::Val, _fs, ::Vector{Fixture}, _ctx) = false

"""
    latest_player_ratings(ds, tracker) -> (Dict{Int,Float64}, Float64)

Every player's current rating, plus the global-average fallback for debutants.

One extra element is appended to each player's history before the tracker runs, so the returned
value is the state *after* the last observation -- the same quantity training assigns to the
next fixture.
"""
function latest_player_ratings(ds, tracker)
    lu = DataFrames.select(ds.lineups, :match_id, :player_id, :rating)
    md = DataFrames.select(ds.matches, :match_id, :match_date)
    df = sort!(innerjoin(lu, md, on = :match_id), :match_date)

    valid = collect(skipmissing(df.rating))
    valid = filter(!isnan, valid)
    fallback = isempty(valid) ? 6.0 : mean(valid)

    out = Dict{Int,Float64}()
    for g in groupby(df, :player_id)
        hist = collect(g.rating)
        # one step past the end: the value the next match would be given
        series = Features.calculate_player_ratings(tracker, vcat(hist, missing))
        v = isempty(series) ? NaN : last(series)
        out[Int(g.player_id[1])] = (ismissing(v) || isnan(v)) ?
            (hasproperty(tracker, :prior_mean) ? tracker.prior_mean : fallback) : Float64(v)
    end
    return out, fallback
end

"""
    LineupAggregateFromRAPM()

The plus-minus counterpart of `RatingsFromTracker`: turns tomorrow's teamsheet into the two
lineup maps the `AbstractPlusMinusFeature` extractor emits at training time.

Claims a FeatureSet only when it carries `:plus_minus_ratings` -- the fold's leak-controlled
`Dict{player_id => RAPM rating}`, fit on the history block alone. That is a property of the data
rather than a type test, which matters because the same FeatureSet shape is produced for both the
builder family (`PlayerLineupPillar`) and the legacy player-level engines.

**Which rating vector, and why this is not a leak.** The ratings are the fold's own, frozen at
training time; only the *teamsheet* is new. `plus_minus_extractors.jl` makes the same argument
where it builds the training-time map over the whole DataStore rather than the fold: applying a
history-fit rating to a future XI is precisely the pre-match quantity under test.

**Fidelity.** Both maps mirror `plus_minus_extractors.jl` exactly, including the two places where
the two maps deliberately disagree:

* `:player_lineup_ratings_map` (`pm_lineup_aggregates`) EXCLUDES goalkeepers and includes
  substitutes, in the 18-field order `PMLineupAggregate` declares.
* `:player_ratings_map` INCLUDES goalkeepers, is starters-only, and drops any player whose fitted
  rating is exactly `0.0` -- the extractor's `rt == 0.0 && continue`, which is what makes
  `flat_plus_minus_fallback` meaningful.

Expected minutes are recomputed the way the extractor's rolling state computes them: the mean of
a player's last five positive `minutes_played`, clamped at 120, defaulting to 90 for a starter
and 0 for a substitute with no usable history. Because `as_of` is after every match in `ds`, that
final state IS the pre-match value for tomorrow. Only `MinuteWeightedPlayerAggregation` reads
those two fields; `m12_joint_hybrid_synergy` uses `BenchWeightedPlayerAggregation` and does not.
They are filled anyway, because a materialiser that populated some fields of a struct and left
others at zero would be a trap for the next model.

`MatchDay.clean_position` and `Features.pm_clean_position` are separate functions that agree on
every value SofaScore emits (`G`/`GK`, `D`/`DF`, `F`/`FW`/`A`, everything else `M`). The
`Player.position` symbol is already normalised by the lineup source, so it is used directly.

RELATION TO THE REPLAY ENGINE. `current_development/match_day_inference/replay_state.jl` has
carried a local `PointInTimeLineupRatings` since the replay console was built, for exactly this
gap; it is why the replay suite's R10/R11 pass while the live path could not price m12 at all.
This is the live-path version, and it differs deliberately in two places:

* **No XI, no entry.** The replay materialiser writes the NEUTRAL aggregate for a fixture whose
  teamsheet has not dropped yet, because before the drop that IS the state being replayed. Here
  the fixture is left absent so `check_coverage` refuses it. On a live Saturday a missing XI
  means the scrape failed, and pricing a real card off a zero pillar is the outcome this whole
  key exists to prevent.
* **Minute slots 17-18.** The replay version weights starters 1.0 and substitutes 0.0, on the
  grounds that a pre-match teamsheet has no minute history. That reading is right for a debutant
  and wrong for everyone else: the extractor's `minute_history` holds the player's PREVIOUS five
  appearances, which are available pre-match, and only an empty history falls back to 90/0. This
  version reproduces the rolling mean, so a `MinuteWeightedPlayerAggregation` model would be
  served the quantity it was trained on. Neither version's choice reaches
  `m12_joint_hybrid_synergy`, whose `BenchWeightedPlayerAggregation` reads slots 1-4 only.

The two should eventually be one function; merging them changes replay behaviour that R10/R11
pin, so it is a separate change rather than a side effect of this one.
"""
struct LineupAggregateFromRAPM <: AbstractFeatureMaterialiser end

function materialise!(::LineupAggregateFromRAPM, ::Val{:player_lineup_ratings_map}, fs,
                      fx::Vector{Fixture}, ctx)
    haskey(fs.data, :plus_minus_ratings) || return false
    rating_of = fs.data[:plus_minus_ratings]
    expected  = expected_minutes(ctx.ds)
    map_ = fs.data[:player_lineup_ratings_map]
    for f in fx
        lu = get(ctx.lineups, f.m_id, nothing)
        lu === nothing && continue
        map_[f.m_id] = pm_lineup_aggregate(lu, rating_of, expected)
    end
    return true
end

function materialise!(::LineupAggregateFromRAPM, ::Val{:player_ratings_map}, fs,
                      fx::Vector{Fixture}, ctx)
    haskey(fs.data, :plus_minus_ratings) || return false
    rating_of = fs.data[:plus_minus_ratings]
    map_ = fs.data[:player_ratings_map]
    for f in fx
        lu = get(ctx.lineups, f.m_id, nothing)
        lu === nothing && continue
        entry = Dict{Tuple{String,String},Float64}()
        for (side, players) in (("home", lu.home), ("away", lu.away))
            for p in players
                p.substitute && continue                  # starters only
                rt = get(rating_of, p.player_id, 0.0)
                (isfinite(rt) && rt != 0.0) || continue   # extractor drops unrated players
                key = (side, String(p.position))
                entry[key] = get(entry, key, 0.0) + rt
            end
        end
        map_[f.m_id] = entry
    end
    return true
end

materialise!(::LineupAggregateFromRAPM, ::Val, _fs, ::Vector{Fixture}, _ctx) = false

"""
    pm_lineup_aggregate(lineup, rating_of, expected_minutes) -> PMLineupAggregate

One teamsheet collapsed into the 18 pre-match sums, in `PMLineupAggregate`'s field order.

The index arithmetic is copied from `Features.pm_lineup_aggregates` rather than re-derived, so
the two stay comparable line by line: starters land in slots 1-2 and 5-10, substitutes in 3-4 and
11-16, and every outfielder contributes to the minute-weighted slots 17-18.
"""
function pm_lineup_aggregate(lu::Lineup, rating_of::AbstractDict, expected::AbstractDict)
    values = zeros(Float64, 18)
    for (home, players) in ((true, lu.home), (false, lu.away))
        for p in players
            p.position === :G && continue                 # goalkeepers are not in this aggregate
            rating = get(rating_of, p.player_id, 0.0)
            isfinite(rating) || continue
            pos_index = p.position === :D ? 0 : p.position === :M ? 1 : 2
            if p.substitute
                values[home ? 3 : 4] += rating
                values[(home ? 11 : 14) + pos_index] += rating
            else
                values[home ? 1 : 2] += rating
                values[(home ? 5 : 8) + pos_index] += rating
            end
            minutes = get(expected, p.player_id, p.substitute ? 0.0 : 90.0)
            values[home ? 17 : 18] += rating * (minutes / 90.0)
        end
    end
    return Features.PMLineupAggregate(Tuple(values))
end

"""
    expected_minutes(ds) -> Dict{Int,Float64}

Each player's mean minutes over his last five positive appearances, clamped at 120.

Mirrors the rolling `minute_history` inside `Features.pm_lineup_aggregates`: rows are walked in
`(match_date, match_id)` order, a positive `minutes_played` is appended, and the window is capped
at five. A player with no usable history is simply absent, and the caller supplies the
starter/substitute default -- the same split the extractor makes.

Returns an empty `Dict` when the segment has no `minutes_played` column, which on tiers 56/57 is
also effectively true before 23/24; the fallback then applies to everyone, which is exactly what
training saw.
"""
function expected_minutes(ds)
    out = Dict{Int,Float64}()
    lu = ds.lineups
    cols = propertynames(lu)
    (:player_id in cols && :minutes_played in cols && :match_id in cols) || return out
    nrow(lu) == 0 && return out

    date_of = Dict{Int,Date}(Int(r.match_id) => r.match_date for r in eachrow(ds.matches))
    order = sortperm(1:nrow(lu); by = i -> begin
        mid = Int(lu.match_id[i])
        (get(date_of, mid, Date(9999, 12, 31)), mid)
    end)

    history = Dict{Int,Vector{Float64}}()
    for i in order
        ismissing(lu.player_id[i]) && continue
        ismissing(lu.minutes_played[i]) && continue
        m = Float64(lu.minutes_played[i])
        (isfinite(m) && m > 0.0) || continue
        h = get!(history, Int(lu.player_id[i]), Float64[])
        push!(h, min(m, 120.0))
        length(h) > 5 && popfirst!(h)
    end
    for (player_id, h) in history
        out[player_id] = sum(h) / length(h)
    end
    return out
end

"""
    LeagueFromFixture()

Maps each fixture's `tournament_id` onto the league index the pooled engines were trained with.

The index convention is **not** free to reinvent: `Features.add_feature!(::LeagueFeature, ...)`
builds it as `sort(unique(ds.matches.tournament_id))` enumerated from 1, keyed off the *full*
DataStore rather than the split so it is stable across folds. This reconstructs it from the same
`ds`, which is what makes the serving index identical to the training index by construction
instead of by comment.

A fixture whose tournament is absent from `ds.matches` is left unmapped, so `check_coverage`
refuses it rather than assigning it an arbitrary neighbouring league.
"""
struct LeagueFromFixture <: AbstractFeatureMaterialiser end

function materialise!(::LeagueFromFixture, ::Val{:league_lookup}, fs, fx::Vector{Fixture}, ctx)
    league_ids = sort(unique(Int.(ctx.ds.matches.tournament_id)))
    league_map = Dict(t => i for (i, t) in enumerate(league_ids))

    map_ = fs.data[:league_lookup]
    for f in fx
        idx = get(league_map, f.tournament_id, nothing)
        idx === nothing && continue
        map_[f.m_id] = idx
    end
    return true
end

materialise!(::LeagueFromFixture, ::Val, _fs, ::Vector{Fixture}, _ctx) = false

"""
    MarketPillarFromBook()

Supplies market-derived features from the **same** `odds_df` the staking layer prices against.

Engines like `src_sup40_sw40` take market odds as a model feature, so on match day inference
depends on the book that staking also depends on. That is why the pipeline builds the book
before it materialises features, and why the odds are threaded through rather than re-fetched:
if the two ever come from different reads, the diagnostics compare the model against a price it
was not given.

Declared and deliberately unimplemented -- wiring it needs the engine's exact feature key, which
should be read off `Features.required_features` for the specific engine rather than guessed.
"""
struct MarketPillarFromBook <: AbstractFeatureMaterialiser end

materialise!(::MarketPillarFromBook, ::Val{F}, _fs, ::Vector{Fixture}, _ctx) where {F} = false

"""
    MaterialiserChain(ms...)

Tries each materialiser until one claims the feature. A feature no member handles is an
**error**, not a silent carry-forward of whatever the cached FeatureSet happened to hold.
"""
struct MaterialiserChain{T<:Tuple} <: AbstractFeatureMaterialiser
    members::T
end
MaterialiserChain(ms::AbstractFeatureMaterialiser...) = MaterialiserChain(ms)

function materialise!(c::MaterialiserChain, v::Val{F}, fs, fx::Vector{Fixture}, ctx) where {F}
    for m in c.members
        materialise!(m, v, fs, fx, ctx) && return true
    end
    return false
end

# ===================================================================
# Stage 6
# ===================================================================

"""
    matchday_latents(spec, expr, ds, cards, odds_df, as_of) -> (df, diagnostics)

Posterior latents for fixtures that are in no training fold.

Copies the `FeatureSet` before materialising into it -- the prototype mutated one that came
straight out of a cache, so a second call in the same session saw the first call's fixtures.
"""
function matchday_latents(spec::MatchDaySpec, expr, ds, cards::Vector{<:FixtureCard},
                          odds_df::DataFrame, as_of::DateTime)
    model = expr.config.model
    fx    = Fixture[c.fixture for c in cards]
    isempty(fx) && return (DataFrame(), (; split = 0, warning = "no fixtures"))

    # Identify the fold POSITIVELY where we can: `get_next_matches(ds, fold, config)` is the
    # round that fold was built to predict, so the right fold for a match day is the one whose
    # next round is this card. That is the same call the OOS prediction path uses, which is what
    # keeps train and serve consistent. `exclude` is the fallback for genuinely unplayed
    # fixtures, which are in no fold's next-round because they are not in ds.matches at all.
    boundaries = Data.create_id_boundaries(ds, expr.config.splitter)
    ids = [f.m_id for f in fx]
    sel = select_split(expr, boundaries; exclude = ids,
                       ds = ds, config = expr.config.splitter, fixture_ids = ids)

    fcol = Features.create_features(boundaries, ds, model, expr.config.splitter)
    fs   = deepcopy(fcol[sel.idx][1])          # never mutate the cached FeatureSet

    lineups = Dict(c.fixture.m_id => c.lineup for c in cards if c.lineup !== nothing)
    ctx = (ds = ds, model = model, as_of = as_of, odds = odds_df, lineups = lineups)

    # What `extract_parameters` actually needs for an unseen fixture is narrow: the team_map
    # must know both teams, and any per-match lookup map must have an entry keyed by match_id.
    # The flat `:flat_*` arrays are training-time data and are NOT read at inference.
    #
    # Note `Features.required_features(model)` returns feature *config objects*, not symbols, so
    # it cannot be used to drive `Val` dispatch. Materialisers are keyed on the FeatureSet's own
    # per-match map names instead, which is what extract_parameters indexes.
    for key in INJECTABLE_KEYS
        haskey(fs.data, key) || continue
        materialise!(spec.features, Val(key), fs, fx, ctx) || error(
            "MatchDay: no materialiser in $(typeof(spec.features)) handles :$key, which " *
            "$(typeof(model)) reads per match_id. Carrying the trained map forward unchanged " *
            "would price every fixture off that feature's fallback silently.")
    end

    # Per-fixture coverage, not per-feature presence. `haskey(fs.data, :player_ratings_map)` is
    # true straight out of training and says nothing about today -- checking it would be exactly
    # the silent stale-value failure this guard exists to prevent.
    check_coverage(fs, fx, model)

    # `month_idx` is the CALENDAR month, matching Features' `:flat_months`
    # (`[Dates.month(date) for ...]`), and it must be supplied explicitly. Engines that read it
    # do so as `hasproperty(row, :month_idx) ? Int(row.month_idx) : 1` -- so omitting it applied
    # JANUARY's seasonality to every fixture. Ireland plays no matches at all in January, which
    # makes `δ_month[:, 1]` pure prior; measured over three seasons August is tournament 79's
    # lowest-scoring month (2.13 goals) and 718's highest (3.22), so the omission biased the two
    # Irish divisions' totals in opposite directions.
    #
    # `season_idx` is deliberately NOT supplied. Its fallback is `n_seasons`, the most recent
    # season in the training window, which is the correct season for an upcoming fixture. Naming
    # it here would only create a second place for the index convention to drift.
    frame = DataFrame(match_id = [f.m_id for f in fx],
                      home_team = [f.home for f in fx],
                      away_team = [f.away for f in fx],
                      match_date = [Date(f.kickoff) for f in fx],
                      month_idx = [month(f.kickoff) for f in fx],
                      match_week = fill(999, length(fx)))

    raw = Models.PreGame.extract_parameters(model, frame, fs, sel.chain)
    return (_raw_to_df(raw), (; split = sel.idx, n_splits = length(boundaries),
                                warning = sel.warning))
end

function _raw_to_df(raw)
    ids = collect(keys(raw))
    isempty(ids) && return DataFrame()
    cols = Dict{Symbol,Vector{Any}}(:match_id => ids)
    for k in keys(raw[ids[1]])
        cols[Symbol(k)] = [raw[i][k] for i in ids]
    end
    return DataFrame(cols)
end
