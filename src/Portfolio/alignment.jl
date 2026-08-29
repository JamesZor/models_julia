# src/Portfolio/alignment.jl
#
# Everything that has to line up before a book can be priced: the odds frame indexed by match,
# the fixture table, and the partition of books into settlement windows.
#
# All three used to be per-fixture full-frame scans or `DataStore`-only conveniences. Nothing here
# changes an answer -- `build_odds_index` preserves row order exactly, and `fixture_table` on a
# `DataStore` is the same body it always was.

export build_odds_index, group_slates_by_day

# ===================================================================
# 1. The odds index
# ===================================================================

"""
    build_odds_index(odds_df) -> OddsIndex

One pass over the odds frame, replacing `odds_df.match_id .== match_id` per fixture.

Requires `:match_id, :market_name, :market_line, :selection, :odds_close`. Any frame with that
schema works -- the historical Betfair summary, `ds.odds`, or a live feed.

`missing` in `:odds_close` becomes `NaN` and is rejected at admission by an explicit `isnan` test.
`missing` in `:market_line` is REFUSED here rather than downstream: the legacy predicate
`isapprox.(rows.market_line, line; atol = 1e-3)` produces `missing` for such a row and then raises
inside `view`, so no such frame has ever worked.
"""
function build_odds_index(odds_df::AbstractDataFrame)
    for c in (:match_id, :market_name, :market_line, :selection, :odds_close)
        hasproperty(odds_df, c) || error(
            "odds frame has no :$c column; the portfolio pipeline reads " *
            ":match_id, :market_name, :market_line, :selection, :odds_close. " *
            "It has $(propertynames(odds_df)).")
    end

    n = nrow(odds_df)
    src_mid  = odds_df.match_id
    src_name = odds_df.market_name
    src_line = odds_df.market_line
    src_sel  = odds_df.selection
    src_odds = odds_df.odds_close

    any(ismissing, src_line) && error(
        "odds frame carries `missing` in :market_line. The legacy market predicate raises on " *
        "such a row rather than skipping it, so this refuses it by name. Drop or impute those " *
        "rows before building books.")

    # Stable bucket by match: count, then place. `sortperm` would also work and would also be
    # stable, but this is one pass and no comparison.
    counts = Dict{Int, Int}()
    for i in 1:n
        m = Int(src_mid[i])
        counts[m] = get(counts, m, 0) + 1
    end
    ids    = sort!(collect(keys(counts)))
    rows   = Dict{Int, UnitRange{Int}}()
    cursor = Dict{Int, Int}()
    start  = 1
    for m in ids
        c = counts[m]
        rows[m]   = start:(start + c - 1)
        cursor[m] = start
        start += c
    end

    mid  = Vector{Int}(undef, n)
    name = Vector{String}(undef, n)
    line = Vector{Float64}(undef, n)
    sel  = Vector{Symbol}(undef, n)
    odds = Vector{Float64}(undef, n)

    @inbounds for i in 1:n
        m = Int(src_mid[i])
        p = cursor[m]
        cursor[m] = p + 1
        mid[p]  = m
        name[p] = String(src_name[i])
        line[p] = Float64(src_line[i])
        sel[p]  = Symbol(src_sel[i])
        o = src_odds[i]
        odds[p] = ismissing(o) ? NaN : Float64(o)
    end

    return OddsIndex(rows, mid, name, line, sel, odds, n)
end

build_odds_index(oi::OddsIndex) = oi

# ===================================================================
# 2. The fixture table
# ===================================================================
#
# `fixture_table(ds)` (book.jl) is untyped in its argument, so a bare `DataFrame` and a
# comprehension-built `Dict` both used to reach it and both used to fail three frames down. These
# two methods are additive; the `DataStore` body is unchanged and still lives in book.jl.

"""
    fixture_table(matches_df::AbstractDataFrame) -> Dict{Int,FixtureInfo}

Kick-off date and final score from a bare `matches`-shaped frame.

**A `DataStore` can only ever produce SETTLED fixtures.** `ds.matches` is the curated store of
finished matches, so an upcoming fixture is absent from it entirely and `require_result = false`
against a `DataStore` is a silent no-op. For match-day use, build the `Dict` from a fixture list
-- `MatchDay.fixture_info` does this -- and pass that.
"""
function fixture_table(matches_df::AbstractDataFrame)
    out = Dict{Int, FixtureInfo}()
    for r in eachrow(matches_df)
        sc = (ismissing(r.home_score) || ismissing(r.away_score)) ? nothing :
             (Int(r.home_score), Int(r.away_score))
        out[Int(r.match_id)] = (date = Date(r.match_date), score = sc)
    end
    return out
end

fixture_table(d::Dict{Int, FixtureInfo}) = d

# A caller building the dictionary by comprehension gets whatever element type the comprehension
# inferred -- `Dict{Int, @NamedTuple{date::Date, score::Nothing}}` when every fixture is unplayed,
# which is exactly the match-day case. Converting is a dozen entries of work and turns a
# `MethodError` three frames down into nothing at all.
fixture_table(d::AbstractDict) = Dict{Int, FixtureInfo}(
    Int(k) => (date = Date(v.date),
               score = v.score === nothing ? nothing :
                       (Int(v.score[1]), Int(v.score[2]))) for (k, v) in d)

# ===================================================================
# 3. Settlement windows
# ===================================================================

"""
    group_slates_by_day(books) -> Vector{Slate}

The daily partition, named. Exactly `group(DailySlate(), books)`; provided because "group by day"
is what a call site means and `group` alone reads as a generic.
"""
group_slates_by_day(books::Vector{MatchBook}) = group(DailySlate(), books)
