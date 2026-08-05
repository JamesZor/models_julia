# src/Portfolio/book.jl
#
# Stage A of the pipeline: L1 posterior + market quotes -> MatchBook.
#
# Everything here is a pure function of the data and the BookSpec. Nothing in a PolicySpec can
# reach it, which is what makes `hash(BookSpec)` a sound cache key.

export extract_selections, build_book, build_books, book_cache_key

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
function extract_selections(odds_df::DataFrame, match_id::Int, spec::BookSpec,
                            model_probs::Dict)
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

"""
    build_book(spec, latents_row, expr, odds_df, scores) -> MatchBook | nothing

Returns `nothing` for any match we cannot both stake and settle: no usable quotes, no final
score, or a score-matrix failure. Quotes are checked *before* the score matrix is computed,
because that is the expensive step.
"""
function build_book(spec::BookSpec, latents_row, expr, odds_df::DataFrame,
                    scores::Dict{Int,Tuple{Int,Int,Date}})
    m_id = latents_row.match_id
    haskey(scores, m_id) || return nothing
    any(==(m_id), odds_df.match_id) || return nothing
    h, a, dt = scores[m_id]

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

    return MatchBook(m_id, dt, sels, p_grid, R,
                     settle_vector(sels, h, a, spec.exec.commission),
                     res.a, k, res.kkt, res.converged)
end

# `shrink_factor` for the stateless shrinkages ignores the per-match seed offset.
shrink_factor(s::AbstractShrinkage, sm, R::AbstractMatrix, p::AbstractVector,
              a::AbstractAllocator, e::ExecutionConfig; seed_offset::Int = 0) =
    shrink_factor(s, sm, R, p, a, e)

"""
    build_books(spec, latents_df, expr, odds_df, ds) -> Vector{MatchBook}

Threaded over matches. Returns books sorted by `(date, match_id)` -- chronological order is
established here, once, so nothing downstream has to remember to sort. Path metrics computed on
an unsorted series are meaningless, and the prototype's `latents.df` order was neither
chronological nor recoverable by sorting on `match_id`.
"""
function build_books(spec::BookSpec, latents_df::DataFrame, expr, odds_df::DataFrame, ds)
    scores = Dict{Int,Tuple{Int,Int,Date}}()
    for r in eachrow(ds.matches)
        (ismissing(r.home_score) || ismissing(r.away_score)) && continue
        scores[r.match_id] = (Int(r.home_score), Int(r.away_score), Date(r.match_date))
    end

    n   = nrow(latents_df)
    buf = Vector{Union{Nothing,MatchBook}}(undef, n)
    Threads.@threads for i in 1:n
        buf[i] = build_book(spec, latents_df[i, :], expr, odds_df, scores)
    end

    books = MatchBook[b for b in buf if b !== nothing]
    sort!(books, by = b -> (b.date, b.m_id))
    return books
end

"""
    book_cache_key(spec) -> UInt

Stable hash of everything that can change a `MatchBook`. Use it to name a serialised cache:
a `PolicySpec` sweep must never rebuild books.
"""
book_cache_key(spec::BookSpec) = hash((typeof(spec.price), typeof(spec.allocator),
                                       spec.shrink, spec.exec,
                                       string.(spec.markets.markets)))
