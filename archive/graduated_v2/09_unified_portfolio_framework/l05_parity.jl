# ==============================================================================
# 09 — UNIFIED PORTFOLIO & STAKING FRAMEWORK : THE PARITY HARNESS
# ==============================================================================
#
# Loader. Definitions only, no execution. Included from INSIDE `l04_compat_bridge.jl`'s
# module body, because it holds both implementations at once — this framework's builder
# and `BayesianFootball.Portfolio`'s — and neither is reachable from the other's
# namespace. Same arrangement as `08/l05_parity.jl`.
#
# ------------------------------------------------------------------------------
# THE COMPARABILITY CONTRACT
# ------------------------------------------------------------------------------
#
# Every row below compares TWO BUILDERS OVER ONE SET OF NUMBERS. The legacy side is
# the live `src` path — the real `Predictions.extract_params`, the real
# `compute_score_matrix`, the real `compute_market_probs`, the real
# `Portfolio.extract_selections`, the real `Portfolio.allocate` and the real
# `Portfolio.simulate` — fed a `DataFrame` built from the same typed container the new
# side reads. Nothing is transcribed. If `src/Portfolio/book.jl` changes, these tables
# change with it.
#
# THE GATE IS 0 ULP, NOT 1e-12, on everything except two rows that say so.
#
# The briefing asks for `|Δ| < 1e-12`. `06_typed_posterior_latents` established why
# that is the wrong gate for a price: a one-ULP perturbation of a single λ propagates
# into a score grid as ~1e-17 absolute and 16 ULP — inside 1e-12, and unmistakable in
# ULP. A tolerance-only gate reports "pass" while the numbers have in fact moved.
#
# Everything this framework computes on the book-building path is the same arithmetic
# on the same operands in the same order, so bit-identity is achievable and anything
# less is a defect. The two exceptions are named where they occur:
#
#   * `a_kelly` and `kkt` — Optim's LBFGS is deterministic given identical inputs, so
#     these ARE gated at 0 ULP. They are listed here only because a reader will expect
#     an optimiser to be the loose row and it is worth saying that it is not: the
#     inputs `p` and `R` are bit-identical, so the iterate sequence is bit-identical.
#   * the bootstrap growth interval has no legacy counterpart at all and is therefore
#     not a parity row. It is checked for reproducibility instead.
#
# ==============================================================================


# ==============================================================================
# 1. THE LEGACY BUILDER, RUN LIVE
# ==============================================================================

"""
    LegacyExpr(model)

The minimum an `ExperimentResults` has to be for `Portfolio.build_book` to accept it.

`build_book` reads exactly one thing off its `expr` argument — `expr.config.model`
(`book.jl:87`) — and carries the whole `ExperimentResults` to get there. Constructing
a real one here would mean constructing a real training run, which would make the
parity harness depend on the inference framework being able to fit something. It does
not need to be: the legacy builder needs a model, and this hands it a model.
"""
struct LegacyExpr{M}
    config::NamedTuple{(:model,), Tuple{M}}
end
LegacyExpr(model) = LegacyExpr((; model = model))

"""
    legacy_build(spec, latents, model, odds_df, fixtures; require_result = true)
        -> Vector{MatchBook}

`BayesianFootball.Portfolio.build_books`, over the same container, through the legacy
`DataFrame` of boxed posterior samples.

`to_legacy_dataframe(latents)` is `06`'s bridge and rebuilds exactly the frame
`Experiments._latent_state_dict_to_df` would have produced, so the legacy side pays the
`Vector{Any}` unboxing and the per-fixture tensor it always paid.
"""
function legacy_build(spec::BookSpec, l::AbstractPosteriorLatents, model,
                      odds_df::AbstractDataFrame, fixtures; require_result::Bool = true)
    df  = to_legacy_dataframe(l)
    fxs = fixture_table(fixtures)
    return UP_PF.build_books(spec, df, LegacyExpr(model), DataFrame(odds_df), fxs;
                             require_result = require_result)
end


# ==============================================================================
# 2. BOOK PARITY
# ==============================================================================
#
# A book is compared in six layers, each of which can fail on its own so the report
# says WHERE two builders diverged rather than only that they did:
#
#   structure   the same fixtures, in the same order, with the same selection count
#   selections  family / group / line / selection — the categorical spine
#   prices      odds_quoted / odds_used / p_model / p_market
#   grid        p_grid, after truncation renormalisation
#   payoff      R and the settlement vector
#   allocation  a_kelly / k_shrink / kkt / converged
#
# The order is deliberate: a divergence in `selections` makes every later row
# meaningless, and a report that shows all six lets a reader see that immediately
# instead of chasing an allocation difference that is really an ordering difference.

"""
    book_structure_checks(legacy, new) -> Vector{Pair{String, Bool}}

The comparisons that are not floating-point: fixture sets, ordering, selection counts,
selection identity, and the allocator's convergence flag.

`Selection` is compared field by categorical field rather than with `==`, because
`Selection` is an immutable struct of `Float64`s and a whole-struct `==` would report
a price difference as a selection-identity failure.
"""
function book_structure_checks(legacy::Vector{MatchBook}, new::Vector{MatchBook})
    out = Pair{String, Bool}[]
    push!(out, "same number of books" => (length(legacy) == length(new)))
    length(legacy) == length(new) || return out

    push!(out, "same fixtures, in the same order" =>
          all(a.m_id == b.m_id for (a, b) in zip(legacy, new)))
    push!(out, "same kick-off dates" =>
          all(a.date == b.date for (a, b) in zip(legacy, new)))
    push!(out, "same selection counts" =>
          all(length(a.sels) == length(b.sels) for (a, b) in zip(legacy, new)))
    push!(out, "books are chronological" =>
          issorted(new, by = b -> (b.date, b.m_id)))

    ident = all(zip(legacy, new)) do (a, b)
        length(a.sels) == length(b.sels) && all(zip(a.sels, b.sels)) do (x, y)
            x.family == y.family && x.group == y.group && x.line == y.line &&
                x.selection == y.selection
        end
    end
    push!(out, "same selections, in the same order" => ident)
    push!(out, "same allocator convergence flags" =>
          all(a.converged == b.converged for (a, b) in zip(legacy, new)))
    push!(out, "same settlement status" =>
          all((a.settle === nothing) == (b.settle === nothing) for (a, b) in zip(legacy, new)))
    return out
end

"""
    book_parity_rows(legacy, new; tol, ulp_budget) -> Vector{ParityRow}

The floating-point layers, each pooled across every book and every selection so one
row covers the whole fold.

Pooling is safe here in a way it is not for a metric: these are the same numbers
computed twice, not two summaries of one dataset, so a single divergent element must
show up as a non-zero maximum. `n` on each row is the count actually compared, and
`tpl_parity_table` fails any row that compared nothing.
"""
function book_parity_rows(legacy::Vector{MatchBook}, new::Vector{MatchBook};
                          tol::Float64 = 1e-12, ulp_budget::Integer = 0)
    rows = ParityRow[]
    n = min(length(legacy), length(new))
    if length(legacy) != length(new)
        push!(rows, ParityRow("book count", 0, Inf, typemax(Int64), tol,
                              Int64(ulp_budget), false))
        return rows
    end
    L = legacy[1:n]; N = new[1:n]

    _flat(f) = reduce(vcat, [f(b) for b in L], init = Float64[]),
               reduce(vcat, [f(b) for b in N], init = Float64[])

    for (name, f) in ("p_grid (posterior-mean score grid)" => (b -> b.p_grid),
                      "odds_quoted"      => (b -> Float64[s.odds_quoted for s in b.sels]),
                      "odds_used (post price policy)" => (b -> Float64[s.odds_used for s in b.sels]),
                      "p_model"          => (b -> Float64[s.p_model for s in b.sels]),
                      "p_market (vig-removed)" => (b -> Float64[s.p_market for s in b.sels]),
                      "R (payoff matrix)" => (b -> vec(b.R)),
                      "settle vector"    => (b -> b.settle === nothing ? Float64[] : b.settle),
                      "a_kelly (Kelly allocation)" => (b -> b.a_kelly),
                      "k_shrink (Baker-McHale)" => (b -> Float64[b.k_shrink]),
                      "kkt residual"     => (b -> Float64[b.kkt]))
        a, c = _flat(f)
        push!(rows, tpl_compare(name, a, c; tol = tol, ulp_budget = ulp_budget))
    end
    return rows
end


# ==============================================================================
# 3. TRAJECTORY PARITY
# ==============================================================================

"""
    trajectory_parity_rows(legacy::Trajectory, new::Trajectory; …) -> Vector{ParityRow}

Every path series `simulate` produces, compared element by element.

The bet frame is compared on its numeric columns only; the categorical ones are
handled by `trajectory_structure_checks`, for the same reason `Selection` is.
"""
function trajectory_parity_rows(a::Trajectory, b::Trajectory;
                                tol::Float64 = 1e-12, ulp_budget::Integer = 0)
    rows = ParityRow[]
    cmp(name, x, y) = push!(rows, tpl_compare(name, x, y; tol = tol, ulp_budget = ulp_budget))

    cmp("bankroll series", a.bankroll, b.bankroll)
    cmp("slate P/L", a.slate_pl, b.slate_pl)
    cmp("k_risk per slate", a.k_risk, b.k_risk)
    cmp("exposure per slate", a.exposure, b.exposure)
    cmp("total stake / P&L", [a.total_stake, a.total_pl], [b.total_stake, b.total_pl])

    if nrow(a.bets) == nrow(b.bets) && nrow(a.bets) > 0
        for c in (:stake, :pnl, :odds, :payoff, :p_model, :p_market)
            cmp("bets.$c", a.bets[!, c], b.bets[!, c])
        end
    else
        push!(rows, ParityRow("bets frame", 0, Inf, typemax(Int64), tol,
                              Int64(ulp_budget), false))
    end
    return rows
end

"The non-numeric half of a trajectory comparison."
function trajectory_structure_checks(a::Trajectory, b::Trajectory)
    out = Pair{String, Bool}[]
    push!(out, "same slate dates" => (a.dates == b.dates))
    push!(out, "same capped-slate count" => (a.n_capped == b.n_capped))
    push!(out, "same bet count" => (nrow(a.bets) == nrow(b.bets)))
    nrow(a.bets) == nrow(b.bets) || return out
    push!(out, "same bets, in the same order" =>
          (a.bets.match_id == b.bets.match_id && a.bets.family == b.bets.family &&
           a.bets.selection == b.bets.selection && a.bets.date == b.bets.date))
    return out
end

"""
    summary_parity_rows(new_summary, legacy_path_metrics) -> Vector{ParityRow}

The overlap between `PortfolioSummary` and `Portfolio.path_metrics`, which is every
field `src` has. The six fields this framework adds — CAGR, Sharpe, annualised Sharpe,
Sortino, win rate, 1X2 ROI — have no counterpart and are checked for reproducibility
and for internal consistency in `r01_demo.jl` §7 instead.
"""
function summary_parity_rows(s::PortfolioSummary, pm::NamedTuple;
                             tol::Float64 = 1e-12, ulp_budget::Integer = 0)
    pairs = ("final wealth"       => (s.final_bankroll / s.initial_bankroll, pm.final),
             "flat ROI"           => (s.roi, pm.roi),
             "growth per slate"   => (s.growth_per_slate, pm.growth_per_slate),
             "max drawdown"       => (s.mdd, pm.mdd),
             "ulcer index"        => (s.ulcer, pm.ulcer),
             "calmar"             => (s.calmar, pm.calmar),
             "martin"             => (s.martin, pm.martin),
             "mean exposure"      => (s.mean_exposure, pm.mean_exposure),
             "max exposure"       => (s.max_exposure, pm.max_exposure),
             "worst slate"        => (s.worst_slate, pm.worst_slate),
             "mean k_risk"        => (s.mean_k_risk, pm.mean_k_risk),
             "total stake"        => (s.total_stake, 0.0))
    rows = ParityRow[]
    for (name, (x, y)) in pairs
        name == "total stake" && continue
        push!(rows, tpl_compare(name, [x], [y]; tol = tol, ulp_budget = ulp_budget))
    end
    push!(rows, tpl_compare("slate / bet counts",
                            Float64[s.n_slates, s.n_bets, s.n_capped],
                            Float64[pm.n_slates, pm.n_bets, pm.n_capped];
                            tol = tol, ulp_budget = ulp_budget))
    return rows
end


# ==============================================================================
# 4. THE ALLOCATION AUDIT
# ==============================================================================

"""
    scoring_allocations(w, latents, i; reps = 3) -> Int

Bytes allocated by `price_fixture!` — the score grid plus every fast-bucket market
book — for one fixture.

Measured after a warm-up call so the return is a steady-state figure and not the
first-call compilation. `reps` repeats and takes the MINIMUM, because `@allocated` on a
JIT-compiled call can catch a one-off from an unrelated task on a busy thread.

The claim under test is `06`'s RULE 2, one level up: not merely that
`compute_score_grid!` allocates nothing, but that a whole fixture can be priced across
every market in the spec without touching the heap.
"""
function scoring_allocations(w::BookWorkspace, l::AbstractPosteriorLatents, i::Int;
                             reps::Int = 3)
    price_fixture!(w, l, i)                       # warm up
    best = typemax(Int)
    for _ in 1:reps
        best = min(best, Int(@allocated price_fixture!(w, l, i)))
    end
    return best
end

"An empty-closure baseline. Must also read 0, or the measurement means nothing."
function baseline_allocations(; reps::Int = 3)
    f() = nothing
    f()
    best = typemax(Int)
    for _ in 1:reps
        best = min(best, Int(@allocated f()))
    end
    return best
end

"""
    AllocationRow

One `@allocated` measurement, with the claim it is testing.
"""
struct AllocationRow
    what::String
    bytes::Int
    budget::Int
end

pass(r::AllocationRow) = r.bytes <= r.budget

function allocation_table(rows::Vector{AllocationRow}; title::AbstractString = "ALLOCATION")
    width = maximum(length(r.what) for r in rows; init = 24)
    rule  = "-"^(width + 34)
    println()
    println("  ", title)
    println("  ", rule)
    @printf("  %-*s %12s %10s  %s\n", width, "measurement", "bytes", "budget", "verdict")
    println("  ", rule)
    ok = true
    for r in rows
        pass(r) || (ok = false)
        @printf("  %-*s %12d %10d  %s\n", width, r.what, r.bytes, r.budget,
                pass(r) ? "pass" : "FAIL")
    end
    println("  ", rule)
    return ok
end


# ==============================================================================
# 5. COST
# ==============================================================================

"""
    measure_build_cost(what, spec, latents, model, odds_df, fixtures) -> CostRow

Time and heap traffic of both builders over the same fold.

Bytes are the WHOLE build on each side, under `@allocated`, not a modelled figure. That
is the honest measurement here: the legacy path's cost is not one line item but the sum
of a `(12 × 12 × n_draws)` tensor, a `Dict{String, Dict{Symbol, Vector{Float64}}}` and a
full-frame `BitVector` per fixture, and quoting any one of them would understate it.

Both sides pay the same `allocate` and `shrink_factor`, which dominate the TIME on a
small fold. The speedup column therefore understates what the container change does to
a build that is not re-solving Baker-McHale 128 times per fixture; the byte column is
the one that isolates it.
"""
function measure_build_cost(what::AbstractString, spec::BookSpec,
                            l::AbstractPosteriorLatents, model,
                            odds_df::AbstractDataFrame, fixtures)
    fxs = fixture_table(fixtures)
    df  = to_legacy_dataframe(l)
    expr = LegacyExpr(model)

    UP_PF.build_books(spec, df, expr, DataFrame(odds_df), fxs)          # warm up
    build_books(spec, l, odds_df, fxs; quiet = true)

    t_legacy = @elapsed UP_PF.build_books(spec, df, expr, DataFrame(odds_df), fxs)
    b_legacy = Int(@allocated UP_PF.build_books(spec, df, expr, DataFrame(odds_df), fxs))

    t_new = @elapsed build_books(spec, l, odds_df, fxs; quiet = true)
    b_new = Int(@allocated build_books(spec, l, odds_df, fxs; quiet = true))

    return CostRow(String(what), t_legacy, t_new, b_legacy, b_new)
end

"""
    measure_pricing_cost(what, spec, latents, model) -> CostRow

The same measurement with the CONVEX SOLVE REMOVED — one pass over every fixture doing
nothing but building a score grid and pricing the spec's markets from it.

This row is the one that isolates what this framework changed, and it needs to exist
because `measure_build_cost` cannot say so. `BakerMcHale` re-solves the allocator on
128 posterior draws per fixture and allocates on every one of them; against that, the
container change is a rounding error in the total, and a whole-build comparison would
report a real 100× improvement in the pricing stage as "1.03×".

Legacy side, per fixture, verbatim from `book.jl:86-94`:
`extract_params` (unboxing a `Vector{Any}` row) → `compute_score_matrix` (a fresh
`12 × 12 × n_draws` tensor) → one `compute_market_probs` `Dict` per market.

New side, per fixture: `price_fixture!` into the shared workspace.

THE NEW SIDE'S BYTE FIGURE IS THE WORKSPACE, NOT THE LOOP. The loop measures exactly
zero, and a `shrink` column dividing by zero would read `Inf` — technically true and
useless. So the honest denominator is what the new path costs at all: one
`BookWorkspace`, allocated once and reused by every fixture in the fold. The ratio
therefore grows linearly with fold size, which is the property worth seeing: the
legacy figure is per-fixture work, this one is not.
"""
function measure_pricing_cost(what::AbstractString, spec::BookSpec,
                              l::AbstractPosteriorLatents, model)
    df = to_legacy_dataframe(l)
    ms = spec.markets.markets
    n  = n_matches(l)

    function legacy_pass()
        acc = 0.0
        for i in 1:n
            sm = UP_Pred.compute_score_matrix(model, UP_Pred.extract_params(model, df[i, :]))
            for m in ms
                p = UP_Pred.compute_market_probs(sm, m)
                acc += first(first(values(p)))
            end
        end
        return acc
    end

    w = BookWorkspace(spec, l; quiet = true)
    function new_pass()
        acc = 0.0
        for i in 1:n
            price_fixture!(w, l, i)
            for s in w.slots_1x2;  acc += s.book[1][1]; end
            for s in w.slots_btts; acc += s.book[1][1]; end
            for s in w.slots_ou;   acc += s.book[1][1]; end
        end
        return acc
    end

    legacy_pass(); new_pass()                                  # warm up
    t_legacy = @elapsed legacy_pass()
    b_legacy = Int(@allocated legacy_pass())
    t_new = @elapsed new_pass()
    loop_bytes = Int(@allocated new_pass())

    loop_bytes == 0 || @warn("the pricing loop allocated $(loop_bytes) bytes; the " *
                             "zero-allocation claim in §5 of the runner is the one to " *
                             "believe, and this row's byte column is now a mixture.")
    return CostRow(String(what), t_legacy, t_new, b_legacy,
                   workspace_bytes(w) + loop_bytes)
end


# ==============================================================================
# 6. DETERMINISTIC FIXTURES FOR THE RUNNER
# ==============================================================================
#
# `r01_demo.jl` needs a `DataStore` whose fixtures CLUSTER INTO SLATES and whose quotes
# exercise the price policy. `08/l05_parity.jl`'s `synthetic_datastore` gives one
# fixture per date, which would make every slate a single match and quietly turn the
# whole simultaneous-settlement story into a sequence of independent bets — the exact
# failure `SingleMatchSlate`'s docstring warns about.
#
# So the store is assembled here from `08`'s parts: its `simulate_scores` and its
# `synthetic_odds` (which runs the real `Data.Markets._enrich_market_data!`), with a
# matches frame built to a slate calendar and one deliberate distortion of the quotes.
#
# WHAT IS SYNTHETIC AND WHAT IS NOT. The posteriors are prior draws with a fixed seed
# (`06/l04_parity.jl` §9). The odds are the model's OWN mean prices, perturbed in log
# space and vigged. Everything else is the real code path: real market types, real
# enrichment, real `grade_selection`, real allocator, real drawdown solver.
#
# WHAT THAT DOES AND DOES NOT ESTABLISH. It establishes that two builders agree, that
# the gate fires, and that the legacy surface still runs. It establishes NOTHING about
# whether any of these numbers is good: a market built from the model's own prices is
# one the model beats by construction, so every positive ROI in the transcript is an
# artefact of the fixture and not evidence of edge.

"""
    portfolio_matches(latents, scores; per_slate = 8, first_date, gap_days = 7) -> DataFrame

A `ds.matches`-shaped frame whose fixtures arrive `per_slate` at a time, `gap_days`
apart — a league programme, not a sequence of isolated matches.

That shape is load-bearing for everything downstream of `group`: the drawdown budget is
solved across all fixtures settling together, the exposure cap bounds their total, and
both are trivial when a slate holds one match.
"""
function portfolio_matches(l::AbstractPosteriorLatents, scores::Vector{Tuple{Int, Int}};
                           per_slate::Int = 8, first_date::Date = Date(2025, 1, 4),
                           gap_days::Int = 7,
                           teams::Union{Nothing, AbstractDataFrame} = nothing)
    ids = latent_match_ids(l)
    n = length(ids)
    dates = [first_date + Day(gap_days * ((i - 1) ÷ per_slate)) for i in 1:n]
    home = teams === nothing ? ["TEAM_$(lpad(1 + (i - 1) % 8, 2, '0'))" for i in 1:n] :
                               String.(teams.home_team)
    away = teams === nothing ? ["TEAM_$(lpad(1 + i % 8, 2, '0'))" for i in 1:n] :
                               String.(teams.away_team)
    return DataFrame(
        match_id      = Int.(ids),
        match_date    = dates,
        match_month   = [Dates.month(d) for d in dates],
        home_score    = [s[1] for s in scores],
        away_score    = [s[2] for s in scores],
        tournament_id = fill(1, n),
        season        = fill("24/25", n),
        home_team     = home,
        away_team     = away,
        match_week    = [(i - 1) ÷ per_slate + 1 for i in 1:n],
    )
end

"""
    thin_quotes!(odds_df, ids; widen = 1.10) -> DataFrame

Widen every quote of the named matches so their market groups price at an overround
BELOW one, and re-run the real enrichment.

This is not decoration. `synthetic_odds` applies a flat 5% vig, so `DeArb`'s
`d * min(overround, 1)` is the identity on every row and the price policy — the seam
that exists to stop the Kelly solver levering into a recording artefact — would never
be exercised at all.

The distortion is the one the real data has. A closing "price" is a time-weighted
average of trades that happened at different moments, and on ScottishLower the median
O/U and BTTS group has ONE trade in the 20-minute window, so ~45% of O/U groups come out
at overround < 1. Left alone the solver reads that as a risk-free arbitrage: measured 97
full-cover positions and a mean stake of 29.2% of bankroll, against 18.6% with `DeArb`
(`src/Portfolio/implementations/pricing.jl:12-19`).
"""
function thin_quotes!(odds_df::DataFrame, ids; widen::Float64 = 1.10)
    idset = Set(Int.(ids))
    for i in 1:nrow(odds_df)
        Int(odds_df.match_id[i]) in idset || continue
        ismissing(odds_df.odds_close[i]) || (odds_df.odds_close[i] *= widen)
        ismissing(odds_df.odds_open[i])  || (odds_df.odds_open[i]  *= widen)
    end
    for c in names(odds_df)
        c in ("match_id", "market_name", "market_line", "selection",
              "odds_open", "odds_close", "is_winner") || select!(odds_df, Not(c))
    end
    UP_D.Markets._enrich_market_data!(odds_df)
    return odds_df
end

"""
    portfolio_datastore(latents, markets; per_slate, thin_every, …) -> (ds, scores)

A `Data.DataStore` carrying `matches` and `odds` and nothing else — the two domains the
portfolio pipeline reads — with fixtures grouped into slates and every `thin_every`-th
match quoted at an overround below one.

The segment is a real `Data.ScottishLower()` because `DataStore`'s field is typed to
`DataTournemantSegment`. Nothing here queries it.
"""
function portfolio_datastore(l::AbstractPosteriorLatents, markets::AbstractVector;
                             seed::Int = 909090, vig::Float64 = 0.05,
                             noise::Float64 = 0.12, per_slate::Int = 8,
                             first_date::Date = Date(2025, 1, 4), gap_days::Int = 7,
                             thin_every::Int = 4,
                             fixtures::Union{Nothing, AbstractDataFrame} = nothing)
    scores  = simulate_scores(l; seed = seed)
    matches = portfolio_matches(l, scores; per_slate = per_slate, first_date = first_date,
                                gap_days = gap_days, teams = fixtures)
    odds    = synthetic_odds(l, markets, scores; seed = seed + 1, vig = vig, noise = noise)

    if thin_every > 0
        ids = [Int(matches.match_id[i]) for i in 1:nrow(matches) if i % thin_every == 0]
        thin_quotes!(odds, ids)
    end

    ds = UP_D.DataStore(UP_D.ScottishLower(), matches, DataFrame(), odds,
                        DataFrame(), DataFrame(), DataFrame(), DataFrame(), DataFrame())
    return (ds, scores)
end

"""
    drop_market_leg!(odds_df, match_id, group, line, selection) -> DataFrame

Remove one leg of one market group, in place.

Used by the runner to show that `require_complete_markets` bites. A group missing a leg
is not a smaller book: vig removal divides by the sum over the legs PRESENT, so the
survivors' `p_market` falls and the model's apparent edge over them rises — by up to 20
points on a 1X2 group missing one way. The completeness test is the only thing between
that and a stake.
"""
function drop_market_leg!(odds_df::DataFrame, match_id::Integer, group::AbstractString,
                          line::Real, selection::Symbol)
    keep_row = trues(nrow(odds_df))
    for i in 1:nrow(odds_df)
        if Int(odds_df.match_id[i]) == Int(match_id) &&
           String(odds_df.market_name[i]) == String(group) &&
           isapprox(Float64(odds_df.market_line[i]), Float64(line); atol = 1e-3) &&
           Symbol(odds_df.selection[i]) === selection
            keep_row[i] = false
        end
    end
    return deleteat!(odds_df, findall(!, keep_row))
end
