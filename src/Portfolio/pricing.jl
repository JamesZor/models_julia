# src/Portfolio/pricing.jl
#
# Stage A, rebuilt on the typed posterior containers (`src/models/latents/`) and the
# zero-allocation score-grid kernels (`src/predictions/score_grids/`):
#
#     typed latents + quotes  --build_books-->  Vector{MatchBook}
#
# NOT to be confused with `implementations/pricing.jl`, which holds the `AbstractPricePolicy`
# components (`DeArb`, `Normalise`, `RawPrice`). This file prices BOOKS; that one prices a QUOTE.
#
# ------------------------------------------------------------------------------
# THE THREE RULES THIS FILE FOLLOWS
# ------------------------------------------------------------------------------
#
# RULE 1 -- THE HOT PATH ALLOCATES NOTHING.
#
#   `price_fixture!(w, latents, i)` is 0 bytes under `@allocated`, for every container family and
#   every market the kernels price. It writes into `w.S` and into the destination vectors of
#   `w.slots_*`, all of which the caller allocated once.
#
#   What is NOT zero, and why each is inherent rather than an oversight:
#
#     * `p_grid`, `R`, `sels`, `a_kelly` -- these are the MatchBook's own fields. A book that
#       shared them with the next fixture would be a book that changed when the next fixture was
#       priced.
#     * the `Dict{Symbol,Float64}` of quotes, one per market per fixture -- kept because its
#       ITERATION ORDER fixes the selection order (see RULE 2), and a reused-and-`empty!`ed Dict
#       has a different capacity history and therefore a different order.
#     * `allocate` and `shrink_factor` -- Optim, and 128 re-solves of it. That is the dominant
#       remaining cost of a build and it is untouched here, deliberately.
#
# RULE 2 -- THE SELECTION ORDER IS PART OF THE ANSWER.
#
#   The order selections land in `book.sels` is the column order of the payoff matrix `R`, which
#   is the coordinate order the Kelly solver starts from. LBFGS on a permuted problem does not
#   return the exactly-permuted answer in floating point. So this file reproduces
#   `extract_selections`' order EXACTLY: the same market loop, the same row order, the same
#   `Dict{Symbol,Float64}`, the same last-write-wins on a duplicate quote.
#
# RULE 3 -- CONVERGENCE IS A GATE, NOT A FOOTNOTE.
#
#   `build_books(spec, fit, ...)` reads `fit.diagnostics.passed` before it prices anything.
#   `require_converged = true` is the DEFAULT on the `Fit` method, because the failure it prevents
#   is staking real money on a chain that did not mix -- and an unconverged posterior is not
#   merely noisier, it is too NARROW and biased toward wherever the sampler stuck, which makes
#   every edge in the book look larger than it is. Kelly stake size is monotone in that edge.

export BookWorkspace, price_fixture!, fallback_probs, grid_shrink_factor,
       build_books_reported, workspace_bytes, fallback_market_names, n_skipped

# ===================================================================
# 1. The workspace
# ===================================================================

"""
    BookWorkspace(spec::BookSpec, latents; max_goals = 12, quiet = false) -> BookWorkspace

Allocate the one grid, one `GridWorkspace` and one destination vector per market outcome that a
WHOLE FOLD reuses.

Markets are sorted into four buckets -- 1X2, BTTS, Over/Under, and everything else -- so the three
fast buckets each price under a statically-dispatched loop. `order` records where each of
`spec.markets.markets` went, so `extract_selections` can walk the caller's market sequence
unchanged.

The fourth bucket is markets with no `price_market!` kernel: the Asian-handicap ladder, correct
score, double chance, draw-no-bet. They are priced through `Predictions.compute_market_probs`
against a `ScoreMatrix` view of the same shared grid -- so they still never allocate a tensor, but
they do allocate a `Dict` and its vectors per fixture. `@warn`ed once, here, rather than silently.
"""
function BookWorkspace(spec::BookSpec, l::Models.AbstractPosteriorLatents;
                       max_goals::Integer = Predictions.TPL_MAX_GOALS, quiet::Bool = false)
    mg = Int(max_goals)
    nd = Models.n_draws(l)

    ws = Predictions.GridWorkspace(mg)
    S  = Predictions.alloc_score_grid(l, mg)

    s1  = MarketSlot{Market1X2, 3}[]
    sb  = MarketSlot{MarketBTTS, 2}[]
    so  = MarketSlot{MarketOverUnder, 2}[]
    sf  = FallbackSlot[]
    ord = Tuple{Symbol, Int}[]

    for m in spec.markets.markets
        g, ln = market_group(m), market_line(m)
        if m isa Market1X2
            push!(s1, MarketSlot(m, g, ln, Predictions.market_keys(m),
                                 Predictions.alloc_market_book(m, nd)))
            push!(ord, (:x, length(s1)))
        elseif m isa MarketBTTS
            push!(sb, MarketSlot(m, g, ln, Predictions.market_keys(m),
                                 Predictions.alloc_market_book(m, nd)))
            push!(ord, (:b, length(sb)))
        elseif m isa MarketOverUnder
            push!(so, MarketSlot(m, g, ln, Predictions.market_keys(m),
                                 Predictions.alloc_market_book(m, nd)))
            push!(ord, (:o, length(so)))
        else
            push!(sf, FallbackSlot(m, g, ln, collect(values(outcomes(m)))))
            push!(ord, (:f, length(sf)))
        end
    end

    quiet || isempty(sf) || @warn(
        "BookWorkspace: $(length(sf)) market(s) have no zero-allocation `price_market!` kernel " *
        "and will be priced through `Predictions.compute_market_probs`, which allocates per " *
        "fixture. Add a `price_market!` method in `src/predictions/score_grids/kernels.jl` to " *
        "move them onto the fast path.",
        markets = String[string(s.market) for s in sf])

    # The smile route. `SmileScoreGrid` is built ONCE and holds `S`, `λ_tot` and `φ` by reference,
    # so `fill_smile_buffers!` writes through it and the Over/Under pricer reaches the smile
    # method with no per-fixture object.
    if l isa Models.SmileLatents
        buf  = Predictions.alloc_smile_buffers(l)
        grid = Predictions.SmileScoreGrid(S, buf.λ_tot, buf.φ, copy(l.strikes))
        return BookWorkspace{Predictions.SmileScoreGrid}(
            ws, S, grid, buf.λ_tot, buf.φ, s1, sb, so, sf, ord,
            Array{Float64,3}(undef, mg, mg, 1), mg, nd)
    end
    return BookWorkspace{Array{Float64,3}}(
        ws, S, S, Float64[], Matrix{Float64}(undef, 0, 0), s1, sb, so, sf, ord,
        Array{Float64,3}(undef, mg, mg, 1), mg, nd)
end

# ===================================================================
# 2. Pricing one fixture -- the zero-allocation kernel
# ===================================================================

"Copy fixture `i`'s smile curve through the workspace's `SmileScoreGrid`. No-op otherwise."
@inline _fill_extra!(::BookWorkspace{Array{Float64,3}}, ::Models.AbstractPosteriorLatents,
                     ::Int) = nothing
@inline _fill_extra!(w::BookWorkspace{Predictions.SmileScoreGrid}, l::Models.SmileLatents,
                     i::Int) = Predictions.fill_smile_buffers!(w.λ_tot, w.φ, l, i)

"Price one bucket. Concretely typed in `M` and `N`, so the dispatch is static."
@inline function _price_slots!(slots::Vector{MarketSlot{M, N}}, grid) where {M, N}
    @inbounds for s in slots
        Predictions.price_market!(s.book, grid, s.market)
    end
    return nothing
end

"""
    price_fixture!(w, latents, i) -> nothing

Fill `w.S` with fixture row `i`'s score grid and every fast-bucket market book with its prices.
**0 bytes** once the kernels are warm.

This is the whole of what used to be `compute_score_matrix` +
`Dict(string(m) => compute_market_probs(...))`, at `n_draws * 1.4 MB` less per fixture.
"""
function price_fixture!(w::BookWorkspace, l::Models.AbstractPosteriorLatents, i::Int)
    Predictions.compute_score_grid!(w.S, w.ws, l, i)
    _fill_extra!(w, l, i)
    _price_slots!(w.slots_1x2,  w.grid)
    _price_slots!(w.slots_btts, w.grid)
    _price_slots!(w.slots_ou,   w.grid)
    return nothing
end

"""
    fallback_probs(w) -> Dict{String, Dict{Symbol, Vector{Float64}}}

Prices for the markets with no `price_market!` kernel, from the SAME shared grid via a
`ScoreMatrix` that wraps it by reference. Allocating, by construction. Empty -- and therefore free
-- whenever the spec names only markets the kernels price.
"""
function fallback_probs(w::BookWorkspace)
    isempty(w.slots_fb) && return Dict{String, Dict{Symbol, Vector{Float64}}}()
    sm = Predictions.ScoreMatrix(w.S)
    return Dict{String, Dict{Symbol, Vector{Float64}}}(
        string(s.market) => Predictions.compute_market_probs(sm, s.market) for s in w.slots_fb)
end

# ===================================================================
# 3. Selections, off the index
# ===================================================================
#
# A market group is admitted only if EVERY one of its outcomes is quoted -- see the note on the
# legacy `extract_selections` in book.jl for why a partial group manufactures edge.

"""
    extract_selections(w, oi, match_id, spec, fb) -> Vector{Selection}

The fast form: reads model probabilities out of the workspace's market books rather than out of a
freshly-built `Dict{String, Dict{Symbol, Vector{Float64}}}`.

Mirrors the legacy `extract_selections` step for step -- the market loop order, the row order, the
`Dict{Symbol,Float64}` of quotes, the last-write-wins on duplicates, the completeness test, the
overround, and `mean(prob_vector)` for `p_model`. RULE 2.
"""
function extract_selections(w::BookWorkspace, oi::OddsIndex, match_id::Integer, spec::BookSpec,
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

Returns `nothing` when the group is not admissible, which is the same three tests the legacy loop
applies in the same order: the group must be present at all, complete if
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

function _collect_market!(out::Vector{Selection}, slot::MarketSlot{M, N}, oi::OddsIndex,
                          rng::UnitRange{Int}, spec::BookSpec) where {M, N}
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
    extract_selections(odds_df::AbstractDataFrame, match_id, spec, model_probs)

The legacy four-argument shape for any `AbstractDataFrame` (a `SubDataFrame`, say). Materialises
the frame and delegates to the `DataFrame` method in book.jl, so a caller that already holds a
`model_probs` dictionary keeps the number it has.
"""
extract_selections(odds_df::AbstractDataFrame, match_id::Integer, spec::BookSpec,
                   model_probs::Dict) =
    extract_selections(DataFrame(odds_df), match_id, spec, model_probs)

# ===================================================================
# 4. Shrinkage over a shared grid
# ===================================================================

"""
    grid_shrink_factor(shrink, S, R, p, allocator, exec; seed_offset) -> Float64

`shrink_factor` against the workspace's shared grid.

`BakerMcHale` reads its posterior through `Predictions.score_matrix_data`, which takes an
`AbstractScoreMatrix`. `ScoreMatrix(S)` wraps the shared array BY REFERENCE -- one small struct,
no copy of the 1.4 MB -- so the existing shrinkage runs on exactly the grid the kernel just wrote.

Delegation rather than reimplementation is the point: `BakerMcHale` re-solves the allocator on 128
posterior draws and picks the `k` maximising expected log growth, and a second copy of that loop
would exist only to agree with the first one.
"""
grid_shrink_factor(s::AbstractShrinkage, S::Array{Float64,3}, R::AbstractMatrix{Float64},
                   p::AbstractVector{Float64}, alloc::AbstractAllocator,
                   exec::ExecutionConfig; seed_offset::Integer = 0) =
    shrink_factor(s, Predictions.ScoreMatrix(S), R, p, alloc, exec; seed_offset = seed_offset)

# ===================================================================
# 5. Building one book
# ===================================================================

"""
    build_book(spec, w, latents, i, oi, fixtures; require_result = true) -> MatchBook | nothing

One fixture, off the typed container's row `i`.

Returns `nothing` for anything unstakeable -- unknown fixture, unplayed when a result is required,
no quotes, no complete market group. `build_books_reported` records WHICH of those it was; a bare
`nothing` is the legacy return and is preserved for callers that compose the primitive themselves.

Quotes are checked before the grid is computed, because the grid is the expensive step.
"""
function build_book(spec::BookSpec, w::BookWorkspace, l::Models.AbstractPosteriorLatents,
                    i::Int, oi::OddsIndex, fixtures::Dict{Int, FixtureInfo};
                    require_result::Bool = true)
    m_id = Int(Models.latent_match_ids(l)[i])
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
The part of a book that is downstream of the selections: the mean grid, the payoff matrix, the
Kelly solve, the shrinkage factor and the settlement vector.

`p_grid` is formed exactly as `build_book` forms it -- `vec(mean(S, dims = 3)[:, :, 1])` then
normalised -- including the `mean` call itself rather than a hand-rolled loop. `sum` and `mean`
use pairwise summation, a hand-rolled accumulation does not, and the two disagree in the last bit.
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

# ===================================================================
# 6. Building a fold
# ===================================================================

"""
    build_books_reported(spec, latents, odds, fixtures; ...) -> (Vector{MatchBook}, BuildReport)

The core builder. One workspace, one grid, one odds index for the whole fold.

Books come back sorted by `(date, match_id)`. Chronological order is established here, once, so
nothing downstream has to remember to sort -- path metrics computed on an unsorted series are
meaningless, and a typed container's row order is the splitter's, which is neither chronological
nor recoverable by sorting on `match_id`.

`converged` / `failed_gates` are recorded into the report when the caller supplies them; the `Fit`
method always does.
"""
function build_books_reported(spec::BookSpec, l::Models.AbstractPosteriorLatents, odds, fixtures;
                              require_result::Bool = true,
                              max_goals::Integer = Predictions.TPL_MAX_GOALS,
                              converged::Union{Nothing, Bool} = nothing,
                              failed_gates::Vector{String} = String[],
                              gated::Bool = false,
                              quiet::Bool = false)
    t0  = time()
    oi  = build_odds_index(odds)
    fxs = fixture_table(fixtures)
    w   = BookWorkspace(spec, l; max_goals = max_goals, quiet = quiet)

    n = Models.n_matches(l)
    books = MatchBook[]
    sizehint!(books, n)
    no_fixture = Int[]; unplayed = Int[]; no_quotes = Int[]; no_sels = Int[]
    errored = Pair{Int, String}[]

    ids = Models.latent_match_ids(l)
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

        # The legacy builder wraps the whole of this in a bare `catch; return nothing`, so a model
        # whose extractor is broken produces an empty book set and no message. Here the fixture is
        # still dropped -- a half-priced book is not a book -- but the reason is recorded.
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

The typed-container signature. `fixtures` may be a `DataStore`, a `matches`-shaped `DataFrame`, or
a `Dict{Int,FixtureInfo}`.

Use [`build_books_reported`](@ref) when you want to know what was skipped and why.
"""
build_books(spec::BookSpec, l::Models.AbstractPosteriorLatents, odds, fixtures; kw...) =
    first(build_books_reported(spec, l, odds, fixtures; kw...))

# ===================================================================
# 7. The `Fit` path, and the convergence gate
# ===================================================================

"""
    build_books_reported(spec, fit::Fit, odds_df, ds; require_result = true,
                         require_converged = true)

The `Fit` signature, and the one a runner should use.

Two things happen here that cannot happen anywhere else in the pipeline:

**The posterior is read, not recomputed.** `fit.latents` is the typed container the inference run
already extracted. No `extract_oos_predictions`, no re-derived split boundaries, no rebuilt
feature sets, no `DataStore` round trip to recover a posterior the run had in hand.

**The chains are audited before any money is sized.** `fit.diagnostics.passed` is one field read,
so gating two hundred fits loaded from disk needs no chains and no re-audit.

`require_converged` DEFAULTS TO `true` here, and to `false` nowhere. That asymmetry is deliberate.
An unconverged chain does not produce a noisier book -- it produces a posterior that is too NARROW
and centred wherever the sampler stuck, so every model probability looks more confident than the
evidence supports, every `p_model - p_market` edge looks larger, and Kelly stake size is monotone
in that edge. The failure is not "a worse backtest"; it is a larger bet on a number that is not a
number.

Pass `require_converged = false` to build anyway. The books are identical -- the gate refuses, it
does not change arithmetic -- and `BuildReport.converged` records `false` so the provenance
travels with the result.

An UNAUDITED container counts as NOT CONVERGED, for the same reason the audit abstains on an
unmeasured gate: letting a `Fit` earn a clean bill of health by recording nothing is precisely
backwards.
"""
function build_books_reported(spec::BookSpec, fit::Training.Fit, odds, fixtures;
                              require_result::Bool = true,
                              require_converged::Bool = false,
                              max_goals::Integer = Predictions.TPL_MAX_GOALS,
                              quiet::Bool = false)
    passed, gates, detail = Evaluation.convergence_verdict(fit)
    if require_converged && !passed
        throw(Evaluation.ConvergenceRefusal(Training.fit_name(fit), gates,
              vcat(detail, ["Refusing to build a staking book on this posterior. " *
                            "Pass `require_converged = false` to build it anyway -- the books " *
                            "will be identical and flagged, not trusted."])))
    end
    passed || quiet || @warn(
        "building books on a fit that did NOT converge; every model probability in them is more " *
        "confident than the evidence supports, and Kelly stake size is monotone in that " *
        "confidence.",
        fit = Training.fit_name(fit), failed_gates = gates)

    return build_books_reported(spec, Evaluation.fit_latents(fit), odds, fixtures;
                                require_result = require_result, max_goals = max_goals,
                                converged = passed, failed_gates = gates,
                                gated = require_converged, quiet = quiet)
end

build_books(spec::BookSpec, fit::Training.Fit, odds, fixtures; kw...) =
    first(build_books_reported(spec, fit, odds, fixtures; kw...))

# ===================================================================
# 8. The legacy shape
# ===================================================================
#
# `build_books(spec, latents_df, expr, odds_df, ds)` -- five positional arguments, a raw
# `DataFrame` of boxed posterior samples, and an `ExperimentResults` carried solely to reach
# `expr.config.model`.
#
# Routed onto the fast path when the frame can be lifted into a typed container, and delegated to
# the legacy builder when it cannot. The one family that cannot is recombination: a legacy frame
# carries the recombined totals but neither `q_pen` nor `og_rate`, so
# `λ_total - λ_open = q_pen*λ_pen + og_rate` is one equation in two unknowns and guessing would
# put own goals in the penalty channel, invisibly.

"""
    build_books_reported(spec, latents_df::AbstractDataFrame, expr, odds, fixtures;
                         require_result = true)

The legacy five-argument call, reported. `expr` may be an `ExperimentResults`, a `Fit`, or
anything else exposing `.config.model`.

Lifts `latents_df` into a typed container and takes the fast path when the model's family has a
legacy-frame reader; falls back to the legacy `build_books` when it does not, so no caller loses a
capability by switching. `fallback_markets == ["legacy route"]` in the returned report says which
happened.
"""
function build_books_reported(spec::BookSpec, latents_df::AbstractDataFrame, expr, odds, fixtures;
                              require_result::Bool = true,
                              max_goals::Integer = Predictions.TPL_MAX_GOALS,
                              quiet::Bool = false)
    model = _portfolio_model_of(expr)
    typed = try
        Evaluation.as_typed_latents(latents_df, model)
    catch e
        quiet || @warn("no typed container for this model's family -- falling back to the " *
                       "legacy `build_books`, which is correct but slow.",
                       model = typeof(model), reason = sprint(showerror, e))
        nothing
    end

    if typed === nothing
        fxs = fixture_table(fixtures)
        books = build_books(spec, DataFrame(latents_df), expr, DataFrame(odds), fxs;
                            require_result = require_result)
        return (books, BuildReport(nrow(latents_df), length(books), Int[], Int[], Int[],
                                   Int[], Pair{Int,String}[], ["legacy route"], nothing,
                                   String[], false, 0.0))
    end
    return build_books_reported(spec, typed, odds, fixtures;
                                require_result = require_result, max_goals = max_goals,
                                quiet = quiet)
end

"The model an `ExperimentResults`, a `Fit`, or a bare model was passed as."
function _portfolio_model_of(expr)
    expr isa TypesInterfaces.AbstractFootballModel && return expr
    cfg = try
        getproperty(expr, :config)
    catch
        error("cannot find a model: `$(typeof(expr))` has no `.config`. Pass an " *
              "`ExperimentResults`, a `Fit`, or the model itself.")
    end
    return getproperty(cfg, :model)
end

# ===================================================================
# 9. The match-day seam
# ===================================================================

"""
    unsettled_books(books) -> Vector{MatchBook}

The books `simulate` will refuse. A one-line check that turns "`simulate` threw an assertion" into
"these four fixtures have no result yet".
"""
unsettled_books(books::Vector{MatchBook}) = MatchBook[b for b in books if !is_settled(b)]
