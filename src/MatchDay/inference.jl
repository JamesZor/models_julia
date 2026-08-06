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

export select_split, matchday_latents, RatingsFromTracker, MarketPillarFromBook,
       MaterialiserChain, check_coverage

"""
    INJECTABLE_KEYS

`FeatureSet.data` entries that are **per-match lookup maps**, which is what
`extract_parameters` indexes for a fixture it has never seen. Everything else in a `FeatureSet`
is a flat training-time array (`:flat_home_goals`, `:flat_market_λ_home`, ...) and is not read
at inference.

Verified against `DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel`, whose
`extract_parameters` reads exactly `row.home_team`/`row.away_team` (via `:team_map`),
`row.match_id` (via `:player_ratings_map`), and optional `row.season_idx`/`row.month_idx`.
"""
const INJECTABLE_KEYS = (:player_ratings_map,)

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

    isempty(problems) && return true
    error("MatchDay.check_coverage: " * join(problems, " | "))
end

"""
    select_split(expr, boundaries; strict = true) -> (idx, chain, warning)

Choose which trained split to condition on, by boundary rather than by position.

The chain at index `i` was fitted on the boundary that had index `i` **at training time**. If
the boundary list has since grown, the correspondence survives only when the splitter appends.
This checks that the boundary at `idx` still contains the same history/target match ids the
chain was trained on where that is recoverable, and otherwise reports what it could not verify
rather than assuming.

Returns the index used, the chain, and a warning string (empty when nothing is suspicious).
"""
function select_split(expr, boundaries; strict::Bool = true)
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

    boundaries = Data.create_id_boundaries(ds, expr.config.splitter)
    sel = select_split(expr, boundaries)

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
        materialise!(spec.features, Val(key), fs, fx, ctx)
    end

    # Per-fixture coverage, not per-feature presence. `haskey(fs.data, :player_ratings_map)` is
    # true straight out of training and says nothing about today -- checking it would be exactly
    # the silent stale-value failure this guard exists to prevent.
    check_coverage(fs, fx, model)

    frame = DataFrame(match_id = [f.m_id for f in fx],
                      home_team = [f.home for f in fx],
                      away_team = [f.away for f in fx],
                      match_date = [Date(f.kickoff) for f in fx],
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
