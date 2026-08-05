# src/Portfolio/book.jl
#
# Stage A of the pipeline: L1 posterior + market quotes -> MatchBook.
#
# Everything here is a pure function of the data and the BookSpec. Nothing in a PolicySpec can
# reach it, which is what makes `hash(BookSpec)` a sound cache key.

export extract_selections, build_book, build_books, book_cache_key, fixture_table,
       is_settled

"Trust key for a selection: `1X2_home`, `O/U 2.5_over_25`, `BTTS_btts_yes`."
selection_family(group::AbstractString, line::Real, sel::Symbol) =
    group == "OverUnder" ? "O/U $(line)_$(sel)" : "$(group)_$(sel)"

"""
    extract_selections(odds_df, match_id, spec, model_probs) -> Vector{Selection}

Pull the closing price of every configured market for one match and price it.

A market group is admitted only if **every** one of its outcomes is quoted. This matters more
than it looks: the vig-removal step divides by the sum over whatever legs are present, so a
group missing a leg silently manufactures edge on the survivors -- up to 20% on a 1X2 market
missing one way. On ScottishLower ~70% of O/U 0.5 groups and 2 of 1522 1X2 groups are partial.
"""
function extract_selections(odds_df::DataFrame, match_id::Integer, spec::BookSpec,
                            model_probs::Dict)
    # `Integer`, not `Int`: match ids arrive as Int32 from `ds.matches` and Int64 from a
    # latents frame, and a caller composing the primitives by hand should not have to know which.
    rows = view(odds_df, odds_df.match_id .== match_id, :)
    out  = Selection[]
    isempty(rows) && return out

    for m in spec.markets.markets
        m_str  = string(m)
        grp    = Data.market_group(m)
        line   = Data.market_line(m)
        n_want = length(Data.outcomes(m))
        haskey(model_probs, m_str) || continue

        sub = view(rows, (rows.market_name .== grp) .&
                          isapprox.(rows.market_line, line; atol = 1e-3), :)
        isempty(sub) && continue

        quoted = Dict{Symbol,Float64}()
        for r in eachrow(sub)
            (ismissing(r.odds_close) || r.odds_close <= 1.0) && continue
            quoted[r.selection] = r.odds_close
        end
        (spec.exec.require_complete_markets && length(quoted) != n_want) && continue
        isempty(quoted) && continue

        overround = sum(1.0 / o for o in values(quoted))
        for (sel, o) in quoted
            haskey(model_probs[m_str], sel) || continue
            push!(out, Selection(selection_family(grp, line, sel), grp, line, sel,
                                 o,
                                 settlement_odds(spec.price, o, overround),
                                 mean(model_probs[m_str][sel]),
                                 (1.0 / o) / overround))
        end
    end
    return out
end

"Date, and final score when the fixture has been played."
const FixtureInfo = @NamedTuple{date::Date, score::Union{Nothing,Tuple{Int,Int}}}

"""
    build_book(spec, latents_row, expr, odds_df, fixtures; require_result = true) -> MatchBook | nothing

Returns `nothing` for any match we cannot stake: unknown fixture, no usable quotes, or a
score-matrix failure. Quotes are checked *before* the score matrix is computed, because that is
the expensive step.

With `require_result = false` an unplayed fixture is built with `settle = nothing`. Such a book
can be staked (that is match-day use) but not simulated -- `simulate` refuses it.
"""
function build_book(spec::BookSpec, latents_row, expr, odds_df::DataFrame,
                    fixtures::Dict{Int,FixtureInfo}; require_result::Bool = true)
    m_id = latents_row.match_id
    haskey(fixtures, m_id) || return nothing
    fx = fixtures[m_id]
    (require_result && fx.score === nothing) && return nothing
    any(==(m_id), odds_df.match_id) || return nothing

    score_matrix = try
        Predictions.compute_score_matrix(expr.config.model,
                                         Predictions.extract_params(expr.config.model, latents_row))
    catch
        return nothing
    end

    model_probs = Dict(string(m) => Predictions.compute_market_probs(score_matrix, m)
                       for m in spec.markets.markets)

    sels = extract_selections(odds_df, m_id, spec, model_probs)
    isempty(sels) && return nothing

    max_h, max_a, _ = size(score_matrix.data)
    p_grid = vec(mean(score_matrix.data, dims = 3)[:, :, 1])
    p_grid ./= sum(p_grid)                       # absorb grid truncation

    R   = payoff_matrix(sels, max_h, max_a, spec.exec.commission)
    res = allocate(spec.allocator, p_grid, R, spec.exec)
    k   = shrink_factor(spec.shrink, score_matrix, R, p_grid, spec.allocator, spec.exec;
                        seed_offset = m_id)

    settle = fx.score === nothing ? nothing :
             settle_vector(sels, fx.score[1], fx.score[2], spec.exec.commission)

    return MatchBook(m_id, fx.date, sels, p_grid, R, settle, res.a, k, res.kkt, res.converged)
end

"""
    fixture_table(ds) -> Dict{Int,FixtureInfo}

Kick-off date for every match, plus the final score where one exists. Built once and shared
across the threaded book build.
"""
function fixture_table(ds)
    out = Dict{Int,FixtureInfo}()
    for r in eachrow(ds.matches)
        sc = (ismissing(r.home_score) || ismissing(r.away_score)) ? nothing :
             (Int(r.home_score), Int(r.away_score))
        out[Int(r.match_id)] = (date = Date(r.match_date), score = sc)
    end
    return out
end

"""
    build_books(spec, latents_df, expr, odds_df, ds) -> Vector{MatchBook}

`require_result = false` admits unplayed fixtures, which is what match-day staking needs.

Threaded over matches. Returns books sorted by `(date, match_id)` -- chronological order is
established here, once, so nothing downstream has to remember to sort. Path metrics computed on
an unsorted series are meaningless, and the prototype's `latents.df` order was neither
chronological nor recoverable by sorting on `match_id`.
"""
function build_books(spec::BookSpec, latents_df::DataFrame, expr, odds_df::DataFrame, ds;
                    require_result::Bool = true)
    fixtures = fixture_table(ds)

    n   = nrow(latents_df)
    buf = Vector{Union{Nothing,MatchBook}}(undef, n)
    Threads.@threads for i in 1:n
        buf[i] = build_book(spec, latents_df[i, :], expr, odds_df, fixtures;
                            require_result = require_result)
    end

    books = MatchBook[b for b in buf if b !== nothing]
    sort!(books, by = b -> (b.date, b.m_id))
    return books
end

"""
    component_hash(x, h = UInt(0)) -> UInt

Content hash of a configuration component: its type name plus its field values, recursively.

Julia's default `hash` for an immutable struct holding a non-isbits field falls back to
`objectid`, which is identity-based -- so two `BakerMcHale()` values built in the same session
hash differently. Hashing a spec directly therefore produces a key that never repeats and a
cache that never hits, silently turning every policy sweep back into a full rebuild.
"""
function component_hash(x, h::UInt = UInt(0))
    h = hash(string(nameof(typeof(x))), h)
    for f in fieldnames(typeof(x))
        v = getfield(x, f)
        h = if v isa Union{Number,Symbol,AbstractString,Bool}
                hash(v, h)
            elseif v isa AbstractArray && eltype(v) <: Union{Number,Symbol,AbstractString}
                hash(collect(v), h)     # hash(::AbstractArray) is content-based
            else
                component_hash(v, h)
            end
    end
    return h
end

"""
    book_cache_key(spec) -> UInt

Content hash of everything that can change a `MatchBook`. Use it to name a serialised cache:
a `PolicySpec` sweep must never rebuild books.

Equal specs give equal keys -- asserted in `test/portfolio_tests.jl` for a spec carrying a
`BakerMcHale`, which is the case that breaks under a naive `hash`.
"""
function book_cache_key(spec::BookSpec)
    h = component_hash(spec.price)
    h = component_hash(spec.allocator, h)
    h = component_hash(spec.shrink, h)
    h = component_hash(spec.exec, h)
    return hash(string.(spec.markets.markets), h)
end
