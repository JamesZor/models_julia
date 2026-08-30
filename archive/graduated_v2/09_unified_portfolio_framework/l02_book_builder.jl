# ==============================================================================
# 09 — UNIFIED PORTFOLIO & STAKING FRAMEWORK : THE BOOK BUILDER
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Stage A of the pipeline, rebuilt on `06`'s zero-allocation kernels:
#
#     typed latents + quotes  ──build_books──>  Vector{MatchBook}
#
# ------------------------------------------------------------------------------
# THE THREE RULES THIS FILE FOLLOWS
# ------------------------------------------------------------------------------
#
# RULE 1 — THE HOT PATH ALLOCATES NOTHING.
#
#   `price_fixture!(w, latents, i)` is 0 bytes under `@allocated`, for every
#   container family and every market `06` prices. It writes into `w.S` and into the
#   destination vectors of `w.slots_*`, all of which the caller allocated once.
#
#   What is NOT zero, and why each is inherent rather than an oversight:
#
#     * `p_grid`, `R`, `sels`, `a_kelly` — these are the MatchBook's own fields. A
#       book that shared them with the next fixture would be a book that changed
#       when the next fixture was priced.
#     * the `Dict{Symbol,Float64}` of quotes, one per market per fixture — kept
#       because its ITERATION ORDER fixes the selection order (see RULE 2), and a
#       reused-and-`empty!`ed Dict has a different capacity history and therefore a
#       different order. Three entries; ~2,500 of them on a 500-fixture fold, against
#       the 1.8 GB of tensor churn this file removes.
#     * `allocate` and `shrink_factor` — Optim, and 128 re-solves of it. That is the
#       dominant remaining cost of a build and it is untouched here, deliberately;
#       see `l01_types.jl` §1.
#
# RULE 2 — THE SELECTION ORDER IS PART OF THE ANSWER.
#
#   The order selections land in `book.sels` is the column order of the payoff
#   matrix `R`, which is the coordinate order the Kelly solver starts from. LBFGS on
#   a permuted problem does not return the exactly-permuted answer in floating
#   point. So this file reproduces `extract_selections`' order EXACTLY: the same
#   market loop, the same row order, the same `Dict{Symbol,Float64}`, the same
#   last-write-wins on a duplicate quote.
#
# RULE 3 — CONVERGENCE IS A GATE, NOT A FOOTNOTE.
#
#   `build_books(spec, fit, …)` reads `fit.diagnostics.passed` before it prices
#   anything. `require_converged = true` is the DEFAULT on the `Fit` method, because
#   the failure it prevents is staking real money on a chain that did not mix — and
#   an unconverged posterior is not merely noisier, it is too NARROW and biased
#   toward wherever the sampler stuck, which makes every edge in the book look
#   larger than it is. The bet sizing is monotone in that edge.
#
# ==============================================================================

include(joinpath(@__DIR__, "l01_types.jl"))


# ==============================================================================
# 1. THE ODDS INDEX
# ==============================================================================

"""
    build_odds_index(odds_df) -> OddsIndex

One pass over the odds frame, replacing `odds_df.match_id .== match_id` per fixture.

Requires `:match_id, :market_name, :market_line, :selection, :odds_close`. Any frame
with that schema works — the historical Betfair summary, `ds.odds`, or a live feed.

`missing` in `:odds_close` becomes `NaN` and is rejected at admission by an explicit
`isnan` test. `missing` in `:market_line` is REFUSED here rather than downstream: the
legacy predicate `isapprox.(rows.market_line, line; atol = 1e-3)` produces `missing`
for such a row and then raises inside `view`, so no such frame has ever worked.
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
        "odds frame carries `missing` in :market_line. The legacy market predicate " *
        "raises on such a row rather than skipping it, so this refuses it by name. " *
        "Drop or impute those rows before building books.")

    # Stable bucket by match: count, then place. `sortperm` would also work and would
    # also be stable, but this is one pass and no comparison.
    counts = Dict{Int, Int}()
    for i in 1:n
        m = Int(src_mid[i])
        counts[m] = get(counts, m, 0) + 1
    end
    ids = sort!(collect(keys(counts)))
    rows = Dict{Int, UnitRange{Int}}()
    start = 1
    cursor = Dict{Int, Int}()
    for m in ids
        c = counts[m]
        rows[m] = start:(start + c - 1)
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


# ==============================================================================
# 2. THE FIXTURE TABLE
# ==============================================================================
#
# `Portfolio.fixture_table(ds)` reads `ds.matches` and is untyped in its argument, so
# it cannot be extended to a bare `DataFrame` without changing `src`. This framework
# declares its own generic with three methods and reproduces the `DataStore` body
# VERBATIM, so `fixture_table(ds) == Portfolio.fixture_table(ds)` — asserted in
# `r01_demo.jl` §4.

"""
    fixture_table(x) -> Dict{Int, FixtureInfo}

Kick-off date for every match, plus the final score where one exists. Built once and
shared across a whole build.

Accepts a `DataStore` (the legacy shape), a bare `matches`-shaped `DataFrame` (what
the briefing's `build_books(spec, latents, odds_df, matches_df)` needs), or an
already-built `Dict`, which is returned unchanged so a caller can pass either.

**A `DataStore` can only ever produce SETTLED fixtures.** `ds.matches` is the curated
store of finished matches, so an upcoming fixture is absent from it entirely and
`require_result = false` against a `DataStore` is a silent no-op. For match-day use,
build the `Dict` from a fixture list — `MatchDay.fixture_info` does this — and pass
that. The warning is `src`'s (`book.jl:160-168`) and is repeated because the failure
mode is an empty stake sheet with no error.
"""
function fixture_table end

function fixture_table(matches_df::AbstractDataFrame)
    out = Dict{Int, FixtureInfo}()
    for r in eachrow(matches_df)
        sc = (ismissing(r.home_score) || ismissing(r.away_score)) ? nothing :
             (Int(r.home_score), Int(r.away_score))
        out[Int(r.match_id)] = (date = Date(r.match_date), score = sc)
    end
    return out
end

fixture_table(ds::UP_D.DataStore) = fixture_table(ds.matches)
fixture_table(d::Dict{Int, FixtureInfo}) = d

# A caller building the dictionary by comprehension gets whatever element type the
# comprehension inferred — `Dict{Int, @NamedTuple{date::Date, score::Nothing}}` when
# every fixture is unplayed, which is exactly the match-day case. Converting is a dozen
# entries of work and turns a `MethodError` three frames down into nothing at all.
fixture_table(d::AbstractDict) = Dict{Int, FixtureInfo}(
    Int(k) => (date = Date(v.date),
               score = v.score === nothing ? nothing :
                       (Int(v.score[1]), Int(v.score[2]))) for (k, v) in d)


# ==============================================================================
# 3. THE WORKSPACE
# ==============================================================================

"""
    BookWorkspace(spec::BookSpec, latents; max_goals = 12) -> BookWorkspace

Allocate the one grid, one `GridWorkspace` and one destination vector per market
outcome that a WHOLE FOLD reuses.

Markets are sorted into four buckets — 1X2, BTTS, Over/Under, and everything else —
so the three fast buckets each price under a statically-dispatched loop. `order`
records where each of `spec.markets.markets` went, so `extract_selections` can walk
the caller's market sequence unchanged.

The fourth bucket is markets `06` has no `price_market!` for: the Asian-handicap
ladder, correct score, double chance, draw-no-bet. They are priced through
`Predictions.compute_market_probs` against a `ScoreMatrix` view of the same shared
grid — so they still never allocate a tensor, but they do allocate a `Dict` and its
vectors per fixture. `@warn`ed once, here, rather than silently.
"""
function BookWorkspace(spec::BookSpec, l::AbstractPosteriorLatents;
                       max_goals::Integer = TPL_MAX_GOALS, quiet::Bool = false)
    mg = Int(max_goals)
    nd = n_draws(l)

    ws = GridWorkspace(mg)
    S  = alloc_score_grid(l, mg)

    s1  = MarketSlot{Market1X2, 3}[]
    sb  = MarketSlot{MarketBTTS, 2}[]
    so  = MarketSlot{MarketOverUnder, 2}[]
    sf  = FallbackSlot[]
    ord = Tuple{Symbol, Int}[]

    for m in spec.markets.markets
        g, ln = market_group(m), market_line(m)
        if m isa Market1X2
            push!(s1, MarketSlot(m, g, ln, market_keys(m), alloc_market_book(m, nd)))
            push!(ord, (:x, length(s1)))
        elseif m isa MarketBTTS
            push!(sb, MarketSlot(m, g, ln, market_keys(m), alloc_market_book(m, nd)))
            push!(ord, (:b, length(sb)))
        elseif m isa MarketOverUnder
            push!(so, MarketSlot(m, g, ln, market_keys(m), alloc_market_book(m, nd)))
            push!(ord, (:o, length(so)))
        else
            push!(sf, FallbackSlot(m, g, ln, collect(values(outcomes(m)))))
            push!(ord, (:f, length(sf)))
        end
    end

    quiet || isempty(sf) || @warn(
        "BookWorkspace: $(length(sf)) market(s) have no zero-allocation kernel in " *
        "`06_typed_posterior_latents` and will be priced through " *
        "`Predictions.compute_market_probs`, which allocates per fixture. " *
        "Add a `price_market!` method for them to move them onto the fast path.",
        markets = String[string(s.market) for s in sf])

    # The smile route. `SmileScoreGrid` is built ONCE and holds `S`, `λ_tot` and `φ`
    # by reference, so `fill_smile_buffers!` writes through it and the Over/Under
    # pricer reaches the smile method with no per-fixture object.
    if l isa SmileLatents
        buf  = alloc_smile_buffers(l)
        grid = SmileScoreGrid(S, buf.λ_tot, buf.φ, copy(l.strikes))
        return BookWorkspace{SmileScoreGrid}(ws, S, grid, buf.λ_tot, buf.φ,
                                             s1, sb, so, sf, ord,
                                             Array{Float64,3}(undef, mg, mg, 1), mg, nd)
    end
    return BookWorkspace{Array{Float64,3}}(ws, S, S, Float64[], Matrix{Float64}(undef, 0, 0),
                                           s1, sb, so, sf, ord,
                                           Array{Float64,3}(undef, mg, mg, 1), mg, nd)
end


# ==============================================================================
# 4. PRICING ONE FIXTURE — THE ZERO-ALLOCATION KERNEL
# ==============================================================================

"Copy fixture `i`'s smile curve through the workspace's `SmileScoreGrid`. No-op otherwise."
@inline _fill_extra!(::BookWorkspace{Array{Float64,3}}, ::AbstractPosteriorLatents, ::Int) = nothing
@inline _fill_extra!(w::BookWorkspace{SmileScoreGrid}, l::SmileLatents, i::Int) =
    fill_smile_buffers!(w.λ_tot, w.φ, l, i)

"Price one bucket. Concretely typed in `M` and `N`, so the dispatch is static."
@inline function _price_slots!(slots::Vector{MarketSlot{M, N}}, grid) where {M, N}
    @inbounds for s in slots
        price_market!(s.book, grid, s.market)
    end
    return nothing
end

"""
    price_fixture!(w, latents, i) -> nothing

Fill `w.S` with fixture row `i`'s score grid and every fast-bucket market book with
its prices. **0 bytes**, verified by `@allocated` in `r01_demo.jl` §5.

This is the whole of what used to be
`compute_score_matrix` + `Dict(string(m) => compute_market_probs(...))`, at
`n_draws` × 1.4 MB less per fixture.
"""
function price_fixture!(w::BookWorkspace, l::AbstractPosteriorLatents, i::Int)
    compute_score_grid!(w.S, w.ws, l, i)
    _fill_extra!(w, l, i)
    _price_slots!(w.slots_1x2,  w.grid)
    _price_slots!(w.slots_btts, w.grid)
    _price_slots!(w.slots_ou,   w.grid)
    return nothing
end

"""
    fallback_probs(w) -> Dict{String, Dict{Symbol, Vector{Float64}}}

Prices for the markets with no `06` kernel, from the SAME shared grid via a
`ScoreMatrix` that wraps it by reference. Allocating, by construction. Empty — and
therefore free — whenever the spec names only markets `06` prices.
"""
function fallback_probs(w::BookWorkspace)
    isempty(w.slots_fb) && return Dict{String, Dict{Symbol, Vector{Float64}}}()
    sm = UP_Pred.ScoreMatrix(w.S)
    return Dict{String, Dict{Symbol, Vector{Float64}}}(
        string(s.market) => UP_Pred.compute_market_probs(sm, s.market) for s in w.slots_fb)
end


# ==============================================================================
# 5. SELECTIONS
# ==============================================================================
#
# A market group is admitted only if EVERY one of its outcomes is quoted. This
# matters more than it looks: vig removal divides by the sum over whatever legs are
# present, so a group missing a leg silently manufactures edge on the survivors — up
# to 20% on a 1X2 market missing one way. On ScottishLower ~70% of O/U 0.5 groups and
# 2 of 1522 1X2 groups are partial (`src/Portfolio/book.jl:19-23`).

"Trust key for a selection: `1X2_home`, `O/U 2.5_over_25`, `BTTS_btts_yes`."
selection_family(group::AbstractString, line::Real, sel::Symbol) =
    group == "OverUnder" ? "O/U $(line)_$(sel)" : "$(group)_$(sel)"

"""
    extract_selections(w, oi, match_id, spec, fb) -> Vector{Selection}

The fast form: reads model probabilities out of the workspace's market books rather
than out of a freshly-built `Dict{String, Dict{Symbol, Vector{Float64}}}`.

Mirrors `src/Portfolio/book.jl:25-63` step for step — the market loop order, the row
order, the `Dict{Symbol,Float64}` of quotes, the last-write-wins on duplicates, the
completeness test, the overround, and `mean(prob_vector)` for `p_model`. RULE 2.
"""
function extract_selections(w::BookWorkspace, oi::OddsIndex, match_id::Integer,
                            spec::BookSpec,
                            fb::Dict{String, Dict{Symbol, Vector{Float64}}} =
                                Dict{String, Dict{Symbol, Vector{Float64}}}())
    out = Selection[]
    rng = get(oi.rows, Int(match_id), nothing)
    rng === nothing && return out

    for (kind, idx) in w.order
        if kind === :x
            _collect_market!(out, w.slots_1x2[idx], oi, rng, spec)
        elseif kind === :b
            _collect_market!(out, w.slots_btts[idx], oi, rng, spec)
        elseif kind === :o
            _collect_market!(out, w.slots_ou[idx], oi, rng, spec)
        else
            _collect_fallback!(out, w.slots_fb[idx], oi, rng, spec, fb)
        end
    end
    return out
end

"""
Quotes for one market group in one match, in the row order the legacy `view` yields.

Returns `nothing` when the group is not admissible, which is the same three tests the
legacy loop applies in the same order: the group must be present at all, complete if
`require_complete_markets`, and non-empty.
"""
@inline function _quotes(oi::OddsIndex, rng::UnitRange{Int}, grp::String, ln::Float64,
                         n_want::Int, exec::ExecutionConfig)
    quoted = Dict{Symbol, Float64}()
    seen = false
    @inbounds for p in rng
        oi.market_name[p] == grp || continue
        isapprox(oi.market_line[p], ln; atol = 1e-3) || continue
        seen = true
        o = oi.odds_close[p]
        (isnan(o) || o <= 1.0) && continue
        quoted[oi.selection[p]] = o
    end
    seen || return nothing
    (exec.require_complete_markets && length(quoted) != n_want) && return nothing
    isempty(quoted) && return nothing
    return quoted
end

function _collect_market!(out::Vector{Selection}, slot::MarketSlot{M, N},
                          oi::OddsIndex, rng::UnitRange{Int}, spec::BookSpec) where {M, N}
    quoted = _quotes(oi, rng, slot.group, slot.line, N, spec.exec)
    quoted === nothing && return out

    overround = sum(1.0 / o for o in values(quoted))
    for (sel, o) in quoted
        j = _key_index(slot.keys, sel)
        j == 0 && continue
        push!(out, Selection(selection_family(slot.group, slot.line, sel),
                             slot.group, slot.line, sel,
                             o,
                             settlement_odds(spec.price, o, overround),
                             mean(slot.book[j]),
                             (1.0 / o) / overround))
    end
    return out
end

function _collect_fallback!(out::Vector{Selection}, slot::FallbackSlot, oi::OddsIndex,
                            rng::UnitRange{Int}, spec::BookSpec,
                            fb::Dict{String, Dict{Symbol, Vector{Float64}}})
    probs = get(fb, string(slot.market), nothing)
    probs === nothing && return out
    quoted = _quotes(oi, rng, slot.group, slot.line, length(slot.keys), spec.exec)
    quoted === nothing && return out

    overround = sum(1.0 / o for o in values(quoted))
    for (sel, o) in quoted
        haskey(probs, sel) || continue
        push!(out, Selection(selection_family(slot.group, slot.line, sel),
                             slot.group, slot.line, sel,
                             o,
                             settlement_odds(spec.price, o, overround),
                             mean(probs[sel]),
                             (1.0 / o) / overround))
    end
    return out
end

"Position of `sel` in a market's outcome tuple, or `0`. Unrolled; no allocation."
@inline function _key_index(keys::NTuple{N, Symbol}, sel::Symbol) where {N}
    @inbounds for j in 1:N
        keys[j] === sel && return j
    end
    return 0
end

"""
    extract_selections(odds_df, match_id, spec, model_probs) -> Vector{Selection}

The LEGACY four-argument shape (`src/Portfolio/book.jl:25`), delegated verbatim to
`src` so a caller that already holds a `model_probs` dictionary keeps the number it
has. Kept because it is a documented, exported primitive and composing it by hand is
a supported thing to do.
"""
extract_selections(odds_df::AbstractDataFrame, match_id::Integer, spec::BookSpec,
                   model_probs::Dict) =
    UP_PF.extract_selections(DataFrame(odds_df), match_id, spec, model_probs)


# ==============================================================================
# 6. SHRINKAGE OVER A SHARED GRID
# ==============================================================================

"""
    grid_shrink_factor(shrink, S, R, p, allocator, exec; seed_offset) -> Float64

`Portfolio.shrink_factor` against the workspace's shared grid.

`BakerMcHale` reads its posterior through `Predictions.score_matrix_data`, which
takes an `AbstractScoreMatrix`. `ScoreMatrix(S)` wraps the shared array BY REFERENCE
— one 16-byte struct, no copy of the 1.4 MB — so the legacy shrinkage runs on
exactly the grid `06`'s kernel just wrote.

Delegation rather than reimplementation is the point: `BakerMcHale` re-solves the
allocator on 128 posterior draws and picks the `k` maximising expected log growth, and
a second copy of that loop in this repository would exist only to agree with the first
one. `r01_demo.jl` §6 checks `k` against the legacy builder at 0 ULP.
"""
grid_shrink_factor(s::AbstractShrinkage, S::Array{Float64,3}, R::AbstractMatrix{Float64},
                   p::AbstractVector{Float64}, alloc::AbstractAllocator,
                   exec::ExecutionConfig; seed_offset::Integer = 0) =
    shrink_factor(s, UP_Pred.ScoreMatrix(S), R, p, alloc, exec; seed_offset = seed_offset)


# ==============================================================================
# 7. BUILDING ONE BOOK
# ==============================================================================

"""
    build_book(spec, w, latents, i, oi, fixtures; require_result = true) -> MatchBook | nothing

One fixture, off the typed container's row `i`.

Returns `nothing` for anything unstakeable — unknown fixture, unplayed when a result
is required, no quotes, no complete market group. `build_books` records WHICH of those
it was; a bare `nothing` is the legacy return and is preserved for callers that
compose the primitive themselves.

Quotes are checked before the grid is computed, because the grid is the expensive
step — `src`'s ordering, kept.
"""
function build_book(spec::BookSpec, w::BookWorkspace, l::AbstractPosteriorLatents,
                    i::Int, oi::OddsIndex, fixtures::Dict{Int, FixtureInfo};
                    require_result::Bool = true)
    m_id = Int(latent_match_ids(l)[i])
    haskey(fixtures, m_id) || return nothing
    fx = fixtures[m_id]
    (require_result && fx.score === nothing) && return nothing
    haskey(oi.rows, m_id) || return nothing

    price_fixture!(w, l, i)                       # 0 bytes
    sels = extract_selections(w, oi, m_id, spec, fallback_probs(w))
    isempty(sels) && return nothing

    return _finish_book(spec, w, m_id, fx, sels)
end

"""
The part of a book that is downstream of the selections: the mean grid, the payoff
matrix, the Kelly solve, the shrinkage factor and the settlement vector.

`p_grid` is formed exactly as `book.jl:101-102` forms it — `vec(mean(S, dims = 3)[:, :, 1])`
then normalised — including the `mean` call itself rather than a hand-rolled loop.
`sum` and `mean` use pairwise summation, a hand-rolled accumulation does not, and the
two disagree in the last bit; §6 of the runner gates this at 0 ULP.
"""
function _finish_book(spec::BookSpec, w::BookWorkspace, m_id::Int, fx::FixtureInfo,
                      sels::Vector{Selection})
    max_h, max_a, _ = size(w.S)
    p_grid = vec(mean(w.S, dims = 3)[:, :, 1])
    p_grid ./= sum(p_grid)                        # absorb grid truncation

    R   = payoff_matrix(sels, max_h, max_a, spec.exec.commission)
    res = allocate(spec.allocator, p_grid, R, spec.exec)
    k   = grid_shrink_factor(spec.shrink, w.S, R, p_grid, spec.allocator, spec.exec;
                             seed_offset = m_id)

    settle = fx.score === nothing ? nothing :
             settle_vector(sels, fx.score[1], fx.score[2], spec.exec.commission)

    return MatchBook(m_id, fx.date, sels, p_grid, R, settle, res.a, k, res.kkt, res.converged)
end


# ==============================================================================
# 8. BUILDING A FOLD
# ==============================================================================

"""
    build_books_reported(spec, latents, odds, fixtures; …) -> (Vector{MatchBook}, BuildReport)

The core builder. One workspace, one grid, one odds index for the whole fold.

Books come back sorted by `(date, match_id)`. Chronological order is established
here, once, so nothing downstream has to remember to sort — path metrics computed on
an unsorted series are meaningless, and a typed container's row order is the
splitter's, which is neither chronological nor recoverable by sorting on `match_id`.

`converged` / `failed_gates` are recorded into the report when the caller supplies
them; the `Fit` method (§9) always does.
"""
function build_books_reported(spec::BookSpec, l::AbstractPosteriorLatents,
                              odds, fixtures;
                              require_result::Bool = true,
                              max_goals::Integer = TPL_MAX_GOALS,
                              converged::Union{Nothing, Bool} = nothing,
                              failed_gates::Vector{String} = String[],
                              gated::Bool = false,
                              quiet::Bool = false)
    t0  = time()
    oi  = build_odds_index(odds)
    fxs = fixture_table(fixtures)
    w   = BookWorkspace(spec, l; max_goals = max_goals, quiet = quiet)

    n = n_matches(l)
    books = MatchBook[]
    sizehint!(books, n)
    no_fixture = Int[]; unplayed = Int[]; no_quotes = Int[]; no_sels = Int[]
    errored = Pair{Int, String}[]

    ids = latent_match_ids(l)
    fb_needed = !isempty(w.slots_fb)

    for i in 1:n
        m_id = Int(ids[i])
        if !haskey(fxs, m_id)
            push!(no_fixture, m_id); continue
        end
        fx = fxs[m_id]
        if require_result && fx.score === nothing
            push!(unplayed, m_id); continue
        end
        if !haskey(oi.rows, m_id)
            push!(no_quotes, m_id); continue
        end

        # The legacy builder wraps the whole of this in a bare `catch; return nothing`
        # (book.jl:86-91), so a model whose extractor is broken produces an empty book
        # set and no message. Here the fixture is still dropped — a half-priced book is
        # not a book — but the reason is recorded and surfaced by `BuildReport`.
        local book
        try
            price_fixture!(w, l, i)
            sels = extract_selections(w, oi, m_id, spec,
                                      fb_needed ? fallback_probs(w) :
                                      Dict{String, Dict{Symbol, Vector{Float64}}}())
            if isempty(sels)
                push!(no_sels, m_id); continue
            end
            book = _finish_book(spec, w, m_id, fx, sels)
        catch e
            push!(errored, m_id => sprint(showerror, e))
            continue
        end
        push!(books, book)
    end

    sort!(books, by = b -> (b.date, b.m_id))

    report = BuildReport(n, length(books), no_fixture, unplayed, no_quotes, no_sels,
                         errored, fallback_market_names(w), converged, failed_gates,
                         gated, time() - t0)
    return (books, report)
end

"""
    build_books(spec, latents::AbstractPosteriorLatents, odds_df, fixtures) -> Vector{MatchBook}

The briefing's typed-container signature. `fixtures` may be a `DataStore`, a
`matches`-shaped `DataFrame`, or a `Dict{Int, FixtureInfo}`.

Use `build_books_reported` when you want to know what was skipped and why.
"""
build_books(spec::BookSpec, l::AbstractPosteriorLatents, odds, fixtures; kw...) =
    first(build_books_reported(spec, l, odds, fixtures; kw...))


# ==============================================================================
# 9. THE `Fit` PATH, AND THE CONVERGENCE GATE
# ==============================================================================

"""
    build_books(spec, fit::Fit, odds_df, ds; require_result = true, require_converged = true)

The briefing's `Fit` signature, and the one a runner should use.

Two things happen here that cannot happen anywhere else in the pipeline:

**The posterior is read, not recomputed.** `fit.latents` is the typed container the
inference run already extracted (`07`). No `extract_oos_predictions`, no re-derived
split boundaries, no rebuilt feature sets, no `DataStore` round trip to recover a
posterior the run had in hand.

**The chains are audited before any money is sized.** `fit.diagnostics.passed` is one
field read (`07/l02_convergence.jl`), so gating two hundred fits loaded from disk needs
no chains and no re-audit.

`require_converged` DEFAULTS TO `true` here, and to `false` nowhere. That asymmetry is
deliberate. An unconverged chain does not produce a noisier book — it produces a
posterior that is too NARROW and centred wherever the sampler stuck, so every model
probability looks more confident than the evidence supports, every `p_model - p_market`
edge looks larger, and Kelly stake size is monotone in that edge. The failure is not
"a worse backtest"; it is a larger bet on a number that is not a number.

Pass `require_converged = false` to build anyway. The books are identical — the gate
refuses, it does not change arithmetic — and `BuildReport.converged` records `false`
so the provenance travels with the result.

An UNAUDITED container counts as NOT CONVERGED, for the reason `07` abstains on an
unmeasured gate: letting a `Fit` earn a clean bill of health by recording nothing is
precisely backwards.
"""
function build_books_reported(spec::BookSpec, fit::Fit, odds, fixtures;
                              require_result::Bool = true,
                              require_converged::Bool = true,
                              max_goals::Integer = TPL_MAX_GOALS,
                              quiet::Bool = false)
    passed, gates, detail = convergence_verdict(fit)
    if require_converged && !passed
        throw(ConvergenceRefusal(fit_name(fit), gates,
              vcat(detail, ["Refusing to build a staking book on this posterior. " *
                            "Pass `require_converged = false` to build it anyway — the " *
                            "books will be identical and flagged, not trusted."])))
    end
    passed || quiet || @warn(
        "building books on a fit that did NOT converge; every model probability in " *
        "them is more confident than the evidence supports, and Kelly stake size is " *
        "monotone in that confidence.",
        fit = fit_name(fit), failed_gates = gates)

    return build_books_reported(spec, fit_latents(fit), odds, fixtures;
                                require_result = require_result, max_goals = max_goals,
                                converged = passed, failed_gates = gates,
                                gated = require_converged, quiet = quiet)
end

build_books(spec::BookSpec, fit::Fit, odds, fixtures; kw...) =
    first(build_books_reported(spec, fit, odds, fixtures; kw...))


# ==============================================================================
# 10. THE LEGACY SHAPE
# ==============================================================================
#
# `build_books(spec, latents_df, expr, odds_df, ds)` — five positional arguments, a
# raw `DataFrame` of boxed posterior samples, and an `ExperimentResults` carried
# solely to reach `expr.config.model`.
#
# Routed onto the fast path when the frame can be lifted into a typed container, and
# delegated to `src` when it cannot. The one family that cannot is recombination:
# a legacy frame carries the recombined totals but neither `q_pen` nor `og_rate`, so
# `λ_total − λ_open = q_pen·λ_pen + og_rate` is one equation in two unknowns and
# guessing would put own goals in the penalty channel, invisibly (`06/README`).

"""
    build_books(spec, latents_df::DataFrame, expr, odds_df, fixtures; require_result = true)

The legacy five-argument call, unchanged. `expr` may be an `ExperimentResults`, a
`Fit`, or anything else exposing `.config.model`.

Lifts `latents_df` into a typed container and takes the fast path when the model's
family has a legacy-frame reader; falls back to `Portfolio.build_books` when it does
not, so no caller loses a capability by switching. `route` in the returned report
says which happened.
"""
function build_books_reported(spec::BookSpec, latents_df::AbstractDataFrame, expr,
                              odds, fixtures; require_result::Bool = true,
                              max_goals::Integer = TPL_MAX_GOALS, quiet::Bool = false)
    model = _up_model_of(expr)
    typed = try
        as_typed_latents(latents_df, model)
    catch e
        quiet || @warn("no typed container for this model's family — falling back to " *
                       "`Portfolio.build_books`, which is correct but slow.",
                       model = typeof(model), reason = sprint(showerror, e))
        nothing
    end

    if typed === nothing
        fxs = fixture_table(fixtures)
        books = UP_PF.build_books(spec, DataFrame(latents_df), expr, DataFrame(odds), fxs;
                                  require_result = require_result)
        return (books, BuildReport(nrow(latents_df), length(books), Int[], Int[], Int[],
                                   Int[], Pair{Int,String}[], ["legacy route"], nothing,
                                   String[], false, 0.0))
    end
    return build_books_reported(spec, typed, odds, fixtures;
                                require_result = require_result, max_goals = max_goals,
                                quiet = quiet)
end

build_books(spec::BookSpec, latents_df::AbstractDataFrame, expr, odds, fixtures; kw...) =
    first(build_books_reported(spec, latents_df, expr, odds, fixtures; kw...))

"""
    build_book(spec, latents_row, expr, odds_df, fixtures; require_result = true)

The legacy single-fixture call (`src/Portfolio/book.jl:78`), delegated verbatim.

Kept as delegation rather than reimplemented: it takes a `DataFrameRow` of boxed
sample vectors, which is exactly the representation this framework exists to stop
using, and there is no fast path to route a single unboxed row onto. A caller that
wants speed wants `build_books` over a container.
"""
build_book(spec::BookSpec, latents_row::DataFrameRow, expr, odds_df::AbstractDataFrame,
           fixtures::Dict{Int, FixtureInfo}; require_result::Bool = true) =
    UP_PF.build_book(spec, latents_row, expr, DataFrame(odds_df), fixtures;
                     require_result = require_result)

"The model an `ExperimentResults`, a `Fit`, or a bare model was passed as."
function _up_model_of(expr)
    expr isa UP_BF.TypesInterfaces.AbstractFootballModel && return expr
    cfg = try
        getproperty(expr, :config)
    catch
        error("cannot find a model: `$(typeof(expr))` has no `.config`. Pass an " *
              "`ExperimentResults`, a `Fit`, or the model itself.")
    end
    return getproperty(cfg, :model)
end


# ==============================================================================
# 11. THE MATCH-DAY SEAM
# ==============================================================================

"Has this fixture been played and graded? `src/Portfolio/matchday.jl:14`."
const is_settled = UP_PF.is_settled

"""
    unsettled_books(books) -> Vector{MatchBook}

The books `simulate` will refuse. A one-line check that turns
"`simulate` threw an assertion" into "these four fixtures have no result yet".
"""
unsettled_books(books::Vector{MatchBook}) = MatchBook[b for b in books if !is_settled(b)]
