# current_development/orderbook_layer2/l00_corpus.jl
#
# The replayable corpus: which Ireland fixtures can actually be re-priced from the archived
# order book, how deep their book goes, and at what cadence.
#
# ---------------------------------------------------------------------------------------------
# WHY THIS FILE EXISTS
# ---------------------------------------------------------------------------------------------
#
# Every Layer-2 question in this stream ("when do you enter", "how much do you trust the model",
# "which bets do you skip") is answered by replaying the SAME pipeline at many `as_of` instants
# against a book that is now history. That is only honest if the set of fixtures being replayed
# is defined ONCE, by the SAME resolver the replay itself uses, and measured rather than assumed.
#
# So this file does three things and nothing else:
#
#   1. enumerates the fixtures,        via `MatchDay.MatchMetaCrosswalk` -- not fresh SQL
#   2. measures the feed,              per fixture: depth, cadence, liquidity at the close
#   3. freezes the results,            so grading can never drift under a re-run
#
# ---------------------------------------------------------------------------------------------
# WHAT THE IRELAND CORPUS ACTUALLY OFFERS (measured 2026-08-11, not assumed)
# ---------------------------------------------------------------------------------------------
#
#   fixtures            81 with a book  (38 in tournament 79, 43 in 718), out of 106 played
#   window              2026-05-29 -> 2026-08-09, in 12 settlement windows
#   crosswalk           81/81 present AND `is_verified` in `betfair.match_meta`
#   markets             9 usable types per fixture (see MARKET SCOPE below)
#   cadence             3.0 min, p10 = p90 = 3.0 -- a metronome, despite the table's name
#   pre-KO depth        MEDIAN first tick T-225 min. 81/81 reach T-60, 66/81 T-180,
#                       37/81 T-240, only 15/81 T-360
#   two-sided quotes    99.5% of pre-KO rows carry both a bid and an ask
#   top-of-book size    100% populated
#   market_matched      23.8% populated -- see the note in `_fixture_coverage`
#   in-play             the feed keeps running ~2h past kick-off
#
# Two of those lines are corrections of earlier readings, and both errors had the same shape --
# an average taken over a column that is only sometimes there:
#
#   * "avg first tick T-379/T-591 min" was a MEAN dragged by a 48-hour outlier. The median is
#     T-225 and the quartile that matters (how deep can a grid reach and still serve most
#     fixtures) is T-180. `recommend_grid` now derives the grid from that quantile.
#   * The matched-volume-by-time-to-kickoff profile (GBP 8 at T-24h rising to GBP 10.6k at the
#     off) was computed with `avg()`, which skips NULLs -- so it silently described only the 18
#     August fixtures that carry the column. The SPREAD half of that profile is sound, because
#     bid/ask arrays are ~100% populated across all 81.
#
# The crosswalk being complete is the single fact that makes this stream cheap. The MatchDay
# module docstring warns that `MatchMetaCrosswalk` resolves "0% after the job stopped" -- that
# was measured on a ScottishUpper slate on 2026-08-07, AFTER the job's last write (2026-08-04).
# For this corpus, which ends 2026-08-08 and was crosswalked before the job stopped, it resolves
# 100%. `LiveNameMatch` is therefore deliberately NOT in the resolver chain here: a fallback that
# never fires is a fallback you cannot audit, and if it ever DID fire on a replay it would mean
# the crosswalk was backfilled after the fact, which is a different and worse thing.
#
# ---------------------------------------------------------------------------------------------
# TWO TRAPS THIS FILE EXISTS TO AVOID
# ---------------------------------------------------------------------------------------------
#
# T1. `betfair_live.market_metadata.competition_id` is NULL on 360 of the Ireland rows -- the
#     2026-05-28..06-25 collection phase predates `core.tournament_config` being populated.
#     Filtering the corpus on `competition_id` silently drops HALF of it and looks like a data
#     gap rather than a query bug. We never touch that column; we go through `match_meta`.
#
# T2. The table is called `order_book_1m` but the cadence is a property of the collector, not of
#     the name: the 2026-08-08 ScottishLower slate measured 3 minutes. A snapshot grid finer
#     than the true cadence re-reads the same tick and multiplies run time for no information.
#     `measure_cadence` reports the real number so the grid is chosen from evidence.
#
# ---------------------------------------------------------------------------------------------
# MARKET SCOPE
# ---------------------------------------------------------------------------------------------
#
# `DOUBLE_CHANCE` and `ASIAN_HANDICAP` are excluded at the corpus boundary rather than filtered
# downstream, because both arrive MALFORMED in this feed and no amount of modelling fixes that:
#
#   DOUBLE_CHANCE   carries 2 runner symbols ("Home or Away", "draw") where the market has 3
#                   (1X, 12, X2). The third is simply absent, so the group can never be
#                   completed -- and DC additionally has a known scoring defect where
#                   `is_winner` marks 1 of 2 winners and fair probabilities are halved.
#   ASIAN_HANDICAP  does not carry its line; the join key is the selection, not the line, so two
#                   different handicaps are indistinguishable in the archive.
#
# `CORRECT_SCORE` IS kept, despite a standing prior that it is a -20% ROI drag, precisely so that
# the prior can be tested on this corpus rather than assumed (see r04_skip_rules).

using DataFrames, Dates, Statistics, Printf

# Aliases are local to each function rather than top-level `const`s: this file gets re-included
# into warm REPL sessions where `const MD = ...` collides with whatever the last runner bound.
_md() = BayesianFootball.MatchDay
_pf() = BayesianFootball.Portfolio
_dd() = BayesianFootball.Data

"""
Market types this stream will price. See the MARKET SCOPE note above for why `DOUBLE_CHANCE`
and `ASIAN_HANDICAP` are absent — they are malformed in the archive, not merely unprofitable.
"""
"""
The staleness limit the replay's readiness gate enforces, in minutes.

Kept here as well as in `replay_spec(; book_age = Minute(10))` because coverage must be measured
against the SAME limit the pipeline will apply. A corpus that reports a fixture as covered from
T-334 while the gate refuses to price it until T-120 is not a corpus, it is a trap: every entry
rule aimed at the uncovered region silently returns zero legs, which reads like "the model found
no edge early" rather than "there was no book".
"""
const MAX_BOOK_AGE_MIN = 10

const CORPUS_MARKET_TYPES = [
    "MATCH_ODDS",
    "BOTH_TEAMS_TO_SCORE",
    "OVER_UNDER_05", "OVER_UNDER_15", "OVER_UNDER_25",
    "OVER_UNDER_35", "OVER_UNDER_45", "OVER_UNDER_55",
    "CORRECT_SCORE",
]

# ===================================================================
# 1. The corpus
# ===================================================================

"""
    L2Corpus

The frozen, measured set of replayable fixtures.

`identities` is keyed by `match_id` and holds the SAME `MatchDay.Resolved` objects the replay
will use, so the corpus cannot disagree with the pipeline about which markets exist.

`results` is frozen at construction on purpose: grading must not drift if the DataStore or the
`sofascore.matches` table is updated between a replay and its tearsheet.

`excluded` is a first-class field rather than a log line — a fixture that fell out of the corpus
is a fact about the corpus, and the reason it fell out is the only way to tell a thin feed from
a broken query.
"""
struct L2Corpus
    name::String
    tournament_ids::Vector{Int}
    fixtures::Vector{Any}                    # Vector{MatchDay.Fixture}
    identities::Dict{Int,Any}                # match_id => MatchDay.Resolved
    results::Dict{Int,Tuple{Int,Int}}
    coverage::DataFrame
    excluded::DataFrame
    market_types::Vector{String}
    built_at::DateTime
end

"""
    candidate_fixtures(tournament_ids, from, to) -> (fixtures, results)

Every fixture in `tournament_ids` kicking off in `[from, to)`, as `MatchDay.Fixture` objects,
plus a frozen `match_id -> (home, away)` score map.

Reads `sofascore.events` (not `sofascore.matches`) for the fixture list because that is what the
live path reads, and joins `matches` only for the score. Note the live source additionally
filters `status_type = 'notstarted'`, which is exactly why a replay cannot use it — see
`ExplicitFixtures`. Window is in epoch seconds because `events.start_timestamp` is an integer.
"""
function candidate_fixtures(tournament_ids::Vector{Int}, from::Date, to::Date)
    MD = _md()
    lo = Int(round(datetime2unix(DateTime(from))))
    hi = Int(round(datetime2unix(DateTime(to))))

    df = MD._query("""
        SELECT e.match_id, e.tournament_id, e.home_team, e.away_team, e.start_timestamp,
               e.status_type, m.home_score, m.away_score
        FROM sofascore.events e
        LEFT JOIN sofascore.matches m USING (match_id)
        WHERE e.tournament_id = ANY(\$1)
          AND e.start_timestamp >= \$2 AND e.start_timestamp < \$3
        ORDER BY e.start_timestamp, e.match_id;
        """, (tournament_ids, lo, hi))

    isempty(df) && error("candidate_fixtures: no fixtures for $tournament_ids in [$from, $to)")

    fx = MD.Fixture[MD.Fixture(Int(r.match_id), String(r.home_team), String(r.away_team),
                               unix2datetime(r.start_timestamp), Int(r.tournament_id))
                    for r in eachrow(df)]

    results = Dict{Int,Tuple{Int,Int}}()
    for r in eachrow(df)
        (ismissing(r.home_score) || ismissing(r.away_score)) && continue
        results[Int(r.match_id)] = (Int(r.home_score), Int(r.away_score))
    end

    return fx, results
end

"""
    build_corpus(name, tournament_ids; from, to, min_markets = 6, market_types = CORPUS_MARKET_TYPES)
        -> L2Corpus

Resolve, measure and freeze.

A fixture enters the corpus only if it clears every one of:

  * `MatchMetaCrosswalk` returns `Resolved`   -- the replay could find its markets
  * at least `min_markets` of `market_types`  -- enough of a book to build a portfolio from
  * it has at least one pre-kick-off tick     -- there is something to replay
  * it has a final score                      -- it can be graded

Everything rejected lands in `corpus.excluded` with its reason, because the count of what fell
out is the only way to distinguish "the feed is thin here" from "the query is wrong".
"""
function build_corpus(name::String, tournament_ids::Vector{Int};
                      from::Date, to::Date,
                      min_markets::Int = 6,
                      market_types::Vector{String} = CORPUS_MARKET_TYPES)
    MD = _md()
    cands, results = candidate_fixtures(tournament_ids, from, to)

    keep_fx   = Any[]
    ids       = Dict{Int,Any}()
    cov_rows  = NamedTuple[]
    excl_rows = NamedTuple[]

    for f in cands
        label = "$(f.home) v $(f.away)"
        _excl(reason) = push!(excl_rows, (match_id = f.m_id, fixture = label,
                                          kickoff = f.kickoff, tournament_id = f.tournament_id,
                                          reason = reason))

        id = MD.resolve(MD.MatchMetaCrosswalk(), f)
        if !(id isa MD.Resolved)
            _excl(string(id.reason)); continue
        end

        wanted = Dict(k => v for (k, v) in id.market_ids if k in market_types)
        if length(wanted) < min_markets
            _excl("only $(length(wanted)) of $(length(market_types)) market types"); continue
        end

        cov = _fixture_coverage(f, collect(values(wanted)))
        if cov.n_snaps_preko == 0
            _excl("no pre-kickoff ticks"); continue
        end
        if !haskey(results, f.m_id)
            _excl("no final score"); continue
        end

        push!(keep_fx, f)
        ids[f.m_id] = id
        push!(cov_rows, merge((match_id = f.m_id, fixture = label, kickoff = f.kickoff,
                               tournament_id = f.tournament_id, bf_event_id = id.bf_event_id,
                               verified = id.verified, n_markets = length(wanted)), cov))
    end

    frozen = Dict(m => results[m] for m in keys(ids))

    return L2Corpus(name, tournament_ids, keep_fx, ids, frozen,
                    DataFrame(cov_rows), DataFrame(excl_rows), market_types, now())
end

# ===================================================================
# 2. Measuring the feed
# ===================================================================

"""
    _fixture_coverage(fixture, market_ids) -> NamedTuple

Depth and cadence of one fixture's replayable window.

Distinct instants are pulled and differenced in Julia rather than aggregated in SQL because the
MEDIAN gap is the number that matters and it is not the mean: the feed drops out for minutes at
a time, and a mean gap quietly reports a cadence the collector never ran at.

`lead_min` is minutes BEFORE kick-off (positive = earlier), so bigger is deeper.
"""
function _fixture_coverage(f, market_ids::Vector{String})
    MD = _md()
    df = MD._query("""
        SELECT DISTINCT ts FROM betfair_live.order_book_1m
        WHERE market_id = ANY(\$1) AND ts <= \$2 ORDER BY ts;
        """, (market_ids, f.kickoff))

    if isempty(df)
        return (first_lead_min = NaN, last_lead_min = NaN, n_snaps_preko = 0,
                cadence_min = NaN, matched_close = NaN)
    end

    ts   = sort(DateTime.(df.ts))
    lead = [Dates.value(f.kickoff - t) / 60_000 for t in ts]      # minutes before KO
    gaps = length(ts) > 1 ? diff(Dates.value.(ts)) ./ 60_000 : Float64[]

    # `market_matched` is NOT a property of the exchange, it is a property of the COLLECTOR: it
    # is NULL on every Ireland fixture before 2026-08-02 and populated after. Measured on the
    # corpus, 8,085 of 33,954 pre-KO rows (23.8%) carry it. It is therefore reported alongside a
    # `has_matched` flag rather than as a bare number, because a liquidity gate built on it would
    # silently apply to 18 of 81 fixtures. Top-of-book SIZE (`bid_volumes`) is the usable
    # liquidity signal here — that is 100% populated.
    mm = MD._query("""
        SELECT max(market_matched) AS mm,
               count(*) FILTER (WHERE market_matched IS NOT NULL) AS n_mm,
               count(*) AS n_rows
        FROM betfair_live.order_book_1m
        WHERE market_id = ANY(\$1) AND ts <= \$2;
        """, (market_ids, f.kickoff))
    matched = isempty(mm) || ismissing(mm[1, :mm]) ? NaN : Float64(mm[1, :mm]) / 10_000
    n_mm    = isempty(mm) ? 0 : Int(mm[1, :n_mm])

    # LIVE lead, not FIRST lead. `first_lead_min` says when a tick first appeared; it does NOT
    # say the book was tradeable from then on. Measured on the 2026-05-29 slate, the first tick
    # is T-334 and then the feed goes SILENT for over two hours — so `MaxBookAge(10 minutes)`
    # correctly blocks every card until ~T-120, and an entry rule aimed at T-180 returns nothing.
    # `live_lead_min` walks back from the close and stops at the first gap wider than the gate,
    # which is the only lead that answers "how early could you actually have traded".
    lim = Float64(MAX_BOOK_AGE_MIN)
    i = length(ts)
    while i > 1 && (Dates.value(ts[i]) - Dates.value(ts[i - 1])) / 60_000 <= lim
        i -= 1
    end

    return (first_lead_min = maximum(lead),
            live_lead_min  = lead[i],
            last_lead_min  = minimum(lead),
            max_gap_min    = isempty(gaps) ? NaN : maximum(gaps),
            n_snaps_preko  = length(ts),
            cadence_min    = isempty(gaps) ? NaN : median(gaps),
            matched_close  = matched,
            has_matched    = n_mm > 0)
end

"Median that ignores `NaN`. A single missing fixture must not poison a corpus-wide summary."
_nanmedian(v) = (u = filter(!isnan, collect(skipmissing(v))); isempty(u) ? NaN : median(u))

"""
    measure_cadence(corpus) -> NamedTuple

The one number that sets the snapshot grid: the corpus-wide median inter-tick gap, with the
spread across fixtures so a single well-covered match cannot speak for the rest.

Use `recommended_step` as the fine step inside the last hour. It is the median rounded UP to a
whole minute — rounding down would place grid points between ticks, which costs run time and
returns the previous tick again.
"""
function measure_cadence(c::L2Corpus)
    v = collect(skipmissing(c.coverage.cadence_min))
    v = filter(!isnan, v)
    isempty(v) && return (median_min = NaN, p10 = NaN, p90 = NaN, recommended_step = Minute(3))
    med = median(v)
    return (median_min = med,
            p10 = quantile(v, 0.10), p90 = quantile(v, 0.90),
            recommended_step = Minute(max(1, Int(ceil(med)))))
end

"""
    subset_corpus(corpus, tournament_id) -> L2Corpus

One league's slice, sharing the parent's frozen results and identities.

Replays run per league because each league has its own trained experiment, but the corpus is
built once across both so that coverage is measured — and reported — on a single consistent
pass. Subsetting after the fact keeps those two facts from drifting apart.
"""
function subset_corpus(c::L2Corpus, tournament_id::Int)
    fx  = filter(f -> f.tournament_id == tournament_id, c.fixtures)
    ids = Dict(f.m_id => c.identities[f.m_id] for f in fx)
    res = Dict(f.m_id => c.results[f.m_id] for f in fx)
    cov = isempty(c.coverage)  ? c.coverage  : filter(:tournament_id => ==(tournament_id), c.coverage)
    exc = isempty(c.excluded)  ? c.excluded  : filter(:tournament_id => ==(tournament_id), c.excluded)
    return L2Corpus("$(c.name)_$(tournament_id)", [tournament_id], fx, ids, res,
                    cov, exc, c.market_types, c.built_at)
end

"""
    recommend_grid(corpus; coverage = 0.80) -> NamedTuple

Choose the snapshot grid from the measured feed rather than from intuition.

`lookback` is the deepest reach that at least `coverage` of fixtures can actually honour. This
matters more than it sounds: the MEAN first tick on this corpus is T-379/T-591 min, but the
MEDIAN is T-225 and only 15 of 81 fixtures reach T-360. A grid set from the mean would run 4x
the snapshots to serve 18% of the corpus, and every fixture past its own first tick contributes
a `no quotes retrieved` block that reads like a broken pipeline.

The grid is deliberately ADAPTIVE. Nothing moves fast at T-3h — spread is flat from T-4h to
T-1h — so the coarse leg runs at `coarse_step` and only the final hour, where spread halves and
the book fills, runs at the collector's true cadence.
"""
function recommend_grid(c::L2Corpus; coverage::Float64 = 0.80,
                        fine_from::Period = Minute(60), coarse_step::Period = Minute(15))
    # `live_lead_min`, NOT `first_lead_min` — see `_fixture_coverage`. A grid set from the first
    # tick reaches into a region the readiness gate refuses to price.
    fl = filter(!isnan, c.coverage.live_lead_min)
    isempty(fl) && return (lookback = Minute(60), fine_step = Minute(3),
                           coarse_step = coarse_step, fine_from = fine_from, honoured = 0.0,
                           n_snaps = 0)
    lookback = Minute(Int(floor(quantile(fl, 1 - coverage))))
    fine     = measure_cadence(c).recommended_step
    return (lookback   = lookback,
            fine_step  = fine,
            coarse_step = coarse_step,
            fine_from  = fine_from,
            honoured   = count(>=(Dates.value(lookback)), fl) / length(fl),
            first_tick_lookback = Minute(Int(floor(quantile(
                filter(!isnan, c.coverage.first_lead_min), 1 - coverage)))),
            n_snaps    = (Dates.value(lookback) - Dates.value(Minute(fine_from))) ÷
                             Dates.value(Minute(coarse_step)) +
                         Dates.value(Minute(fine_from)) ÷ Dates.value(Minute(fine)) + 1)
end

"""
    corpus_slates(corpus) -> Vector{@NamedTuple{day::Date, fixtures::Vector{Any}}}

Group the corpus into settlement windows by kick-off date.

Matches `Portfolio.DailySlate` semantics deliberately: the Kelly solve must see fixtures that
settle together as ONE simultaneous book, because sizing them sequentially is the ruin bug the
portfolio module was built to fix.
"""
function corpus_slates(c::L2Corpus)
    days = sort(unique(Date.(f.kickoff) for f in c.fixtures))
    return [(day = d, fixtures = filter(f -> Date(f.kickoff) == d, c.fixtures)) for d in days]
end

# ===================================================================
# 3. Reporting
# ===================================================================

"""
    corpus_report(corpus) -> DataFrame

Per-league summary: how many fixtures, how deep the book, how liquid at the close.
The first thing to look at, and the thing to quote when reporting how thin an interval is.
"""
function corpus_report(c::L2Corpus)
    isempty(c.coverage) && return DataFrame()
    g = combine(groupby(c.coverage, :tournament_id),
                nrow => :fixtures,
                :n_markets      => mean        => :mean_markets,
                :first_lead_min => _nanmedian  => :med_first_lead_min,
                :last_lead_min  => _nanmedian  => :med_last_lead_min,
                :n_snaps_preko  => _nanmedian  => :med_snaps,
                :cadence_min    => _nanmedian  => :med_cadence_min,
                :has_matched    => count       => :n_with_matched,
                :matched_close  => _nanmedian  => :med_matched_close)
    return sort!(g, :tournament_id)
end

function Base.show(io::IO, ::MIME"text/plain", c::L2Corpus)
    println(io, "L2Corpus \"$(c.name)\"  [$(join(c.tournament_ids, ", "))]")
    println(io, "├─ fixtures   $(length(c.fixtures)) kept, $(nrow(c.excluded)) excluded")
    if !isempty(c.coverage)
        ko = extrema(c.coverage.kickoff)
        println(io, "├─ window     $(Date(ko[1])) .. $(Date(ko[2]))")
        cad = measure_cadence(c)
        @printf(io, "├─ cadence    median %.1f min (p10 %.1f, p90 %.1f) -> step %s\n",
                cad.median_min, cad.p10, cad.p90, cad.recommended_step)
        @printf(io, "├─ depth      median first tick T-%.0f min, %.0f snapshots pre-KO\n",
                median(c.coverage.first_lead_min), median(c.coverage.n_snaps_preko))
    end
    println(io, "├─ markets    $(length(c.market_types)) types: $(join(c.market_types, ", "))")
    print(io,   "└─ slates     $(length(corpus_slates(c))) settlement windows")
end

Base.show(io::IO, c::L2Corpus) =
    print(io, "L2Corpus(\"$(c.name)\", $(length(c.fixtures)) fixtures, ",
              "$(length(corpus_slates(c))) slates)")
