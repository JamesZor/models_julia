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

export select_split, matchday_latents, RatingsFromTracker, LeagueFromFixture,
       MarketPillarFromBook, MaterialiserChain, check_coverage

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
* `:league_lookup` -- read by every pooled multi-division engine
  (`DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel`, `...PlusMinus...`,
  `...SmileLeague...`) as `get(league_lookup, mid, 0)`; default `0`, which **zeroes the
  zero-sum δ_league offset**. On `ScottishLower [56, 57]` that offset is precisely the goal-level
  gap between League One and League Two, so an unmaterialised fixture is priced at the mean of
  the two tiers.
"""
const INJECTABLE_KEYS = (:player_ratings_map, :league_lookup)

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
`Data.get_next_matches(ds, boundaries[i], config)` returns the matches at `time_step + 1`: the
round fold `i` was built to predict. So the fold for a match day is simply the one whose *next
round is this card*, and the answer is keyed on `(target_season, time_step)` — stable facts about
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
    tracker   = model.player_ratings_feature.tracker
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

    fcol = Features.create_features(boundaries, ds, model)
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
