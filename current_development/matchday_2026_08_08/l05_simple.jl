# current_development/matchday_2026_08_08/l05_simple.jl
#
# ═══════════════════════════════════════════════════════════════════════════════════════════
#  A THIN FAÇADE OVER THE MATCH-DAY PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# WHY THIS EXISTS. Pricing one segment used to read like this:
#
#     ds   = load_datastore_cached(seg)
#     expr = load_experiment("./data/matchday_wknd_0808/scot_upper_poisson_outfield_20260807_011126")
#     fx, results = slate_from_db(tournament_ids(seg), day)
#     bounds = create_id_boundaries(ds, expr.config.splitter)
#     idx    = min(length(expr.training_results), length(bounds))
#     leak   = length(intersect(Set(f.m_id for f in fx), Set(bounds[idx][1].target_match_ids)))
#     spec   = MatchDaySpec(fixtures = ExplicitFixtures(fx), identity = MatchMetaCrosswalk(),
#                           lineups = SourceChain(ProvisionalDB(), LastHistorical(ds)),
#                           gate = GateChain(IdentityResolved(), MaxBookAge(Minute(10))))
#     ...
#
# Thirty lines, of which maybe five are decisions and the rest is ceremony you have to remember
# correctly every time or get a silent wrong answer. That is a bad interface.
#
# Everything above is now:
#
#     ctx = matchday(ScottishUpper(), Date(2026, 8, 8))
#
# and the health checks that used to be your job — which fold, is it leak-free, does the book
# reach back far enough — run automatically and print themselves.
#
# ───────────────────────────────────────────────────────────────────────────────────────────
# THE WHOLE API
# ───────────────────────────────────────────────────────────────────────────────────────────
#
#   ctx = matchday(segment, day)                 load model + book + slate, run the safety checks
#   show_book(ctx; as_of)                        the market book: back/lay prices AND sizes
#   pol = policy(; trust, lambda, cap, filter)   a PolicySpec without the ceremony
#   returns(ctx, policies; times)                ROI per policy per time    ← the answer table
#   stake_columns(ctx, policies; as_of)          one column of stakes per policy, side by side
#   sheet_for(ctx, pol; as_of)                   one full graded sheet, if you want the detail
#
# `times` are MINUTES BEFORE KICKOFF: `times = [60, 30, 0]` means T−60, T−30, and the close.
#
# ───────────────────────────────────────────────────────────────────────────────────────────
# WHAT IT DOES NOT HIDE
# ───────────────────────────────────────────────────────────────────────────────────────────
#
# Two things stay visible on purpose, because hiding them is how you get a confident wrong
# number rather than a loud one:
#
#   * the leak check (which fold the chain is paired with, and whether that fold was fitted on
#     the matches you are pricing) — printed by `matchday`, and it ERRORS rather than warns
#   * whether the fixtures can be graded at all — printed by `matchday`, and `returns` reports
#     P&L as `missing` rather than as zero when they cannot
#
# Requires l02_slate_replay.jl (slate_from_db, book_coverage, grade!, family_trust).

using DataFrames, Dates, Statistics, Printf

# Where r01_train_weekend.jl puts its output. Change this if you train somewhere else.
const EXPERIMENT_ROOT = "./data/matchday_wknd_0808"

# Segment → the directory-name prefix r01 used. Add a row when you add a segment.
const EXPERIMENT_PREFIX = Dict("ScottishUpper" => "scot_upper",
                               "ScottishLower" => "scot_lower",
                               "IrelandAll"    => "ire_pooled")

"""
    find_experiment(segment; root = EXPERIMENT_ROOT) -> String

The newest trained experiment for a segment.

Exists because the two directories are easy to confuse and the failure is unhelpful:
`current_development/matchday_2026_08_08/` is the CODE, `data/matchday_wknd_0808/` is the
MODELS, and `list_experiments` joins `data_dir` onto `dir` so a wrong guess just prints
"Directory not found" and hands back an empty vector.
"""
function find_experiment(segment; root::String = EXPERIMENT_ROOT)
    key = string(nameof(typeof(segment)))
    pre = get(EXPERIMENT_PREFIX, key, nothing)
    pre === nothing && error(
        "no experiment prefix registered for $key. Add one to EXPERIMENT_PREFIX in l05_simple.jl, " *
        "or pass experiment = \"<path>\" explicitly.")

    isdir(root) || error("experiment root $root does not exist. Trained models live under " *
                         "data/, NOT under current_development/.")
    cands = filter(d -> startswith(basename(d), pre), filter(isdir, readdir(root, join = true)))
    isempty(cands) && error(
        "no experiment under $root starting with \"$pre\". Available: " *
        join(basename.(filter(isdir, readdir(root, join = true))), ", "))
    return sort(cands, by = mtime, rev = true)[1]
end

"Everything one segment-day needs, loaded and checked once."
struct MatchDayContext
    segment; day::Date; ds; expr
    fixtures::Vector; results::Dict{Int,Tuple{Int,Int}}
    spec; book::Any; kickoff::DateTime; gradeable::Bool
end

function Base.show(io::IO, c::MatchDayContext)
    print(io, "MatchDayContext($(nameof(typeof(c.segment))), $(c.day), ",
              "$(length(c.fixtures)) fixtures, KO $(Dates.format(c.kickoff, "HH:MM")), ",
              c.gradeable ? "gradeable)" : "NOT gradeable)")
end

"""
    matchday(segment, day; experiment = nothing, markets = nothing, book_age = Minute(10))

Load the model, the slate and the book for one segment on one day, run every safety check, and
return a context the rest of the API takes.

Prints, in order: which experiment, which fold and whether it is leak-free, how many fixtures,
how deep the order book reaches, and whether results exist yet.

**Errors rather than warns** if the chain is conditioned on a fold whose target window contains
the fixtures being priced. That is the one failure mode that produces plausible numbers from a
model that has already seen the answers, so it is not a warning.

The lineup chain is always `ProvisionalDB → LastHistorical(ds)`. An engine that reads no lineup
simply ignores it; an engine that needs one would otherwise hit the default `LastHistorical()`
built with no DataStore, which returns `nothing` for everything and takes out the whole segment
via `check_coverage`.
"""
function matchday(segment, day::Date; experiment = nothing, markets = nothing,
                  book_age::Period = Minute(10))
    MD, DDx, EXPx = _md(), _dd(), BayesianFootball.Experiments

    path = experiment === nothing ? find_experiment(segment) : experiment
    println("\n", "="^92)
    @printf("  %-14s %s\n", "segment", nameof(typeof(segment)))
    @printf("  %-14s %s\n", "experiment", basename(path))

    ds   = DDx.load_datastore_cached(segment)
    expr = EXPx.load_experiment(path)
    @printf("  %-14s %s, %d folds\n", "model",
            nameof(typeof(expr.config.model)), length(expr.training_results))
    length(expr.training_results) > 1 || @warn """folds = 1: this experiment trained on HISTORY
        ONLY and never saw the target season. Anything it prices is off a model that has not
        seen this campaign. Retrain before trusting it."""

    fixtures, results = slate_from_db(DDx.tournament_ids(segment), day)
    kickoff = minimum(f.kickoff for f in fixtures)

    # ---- which fold ------------------------------------------------------------------------
    #
    # Called with exactly the arguments `matchday_latents` uses, so what is printed here IS the
    # fold inference will condition on — not a second opinion that could disagree.
    #
    # `get_next_matches(ds, fold, config)` is the round a fold was built to predict, so the fold
    # for a match day is the one whose next round is this card. `exclude` is the fallback for
    # genuinely unplayed fixtures, which are in no fold's next round.
    bounds = DDx.create_id_boundaries(ds, expr.config.splitter)
    ids    = [f.m_id for f in fixtures]
    naive  = min(length(expr.training_results), length(bounds))
    sel    = MD.select_split(expr, bounds; strict = false, exclude = ids,
                             ds = ds, config = expr.config.splitter, fixture_ids = ids)
    @printf("  %-14s split %d of %d rebuilt  (target sizes %s)\n", "conditioning",
            sel.idx, length(bounds), string([length(b[1].target_match_ids) for b in bounds]))
    if sel.idx == naive
        @printf("  %-14s most recent fold, and clear of this card\n", "")
    else
        nxt = try nrow(DDx.get_next_matches(ds, bounds[sel.idx], expr.config.splitter)) catch; 0 end
        held = length(intersect(Set(ids), Set(bounds[naive][1].target_match_ids)))
        @printf("  %-14s NOT the most recent (%d): that fold already holds %d of this slate in\n",
                "", naive, held)
        @printf("  %-14s its target window. Split %d is the one whose next round IS this card\n",
                "", sel.idx)
        @printf("  %-14s (%d matches). Cache rebuilt after training — retrain to catch up.\n", "", nxt)
    end

    # ---- coverage ------------------------------------------------------------------------
    cov  = book_coverage(fixtures)
    have = count(!isnothing, cov.first_tick)
    @printf("  %-14s %d fixtures, kickoff %s\n", "slate", length(fixtures), kickoff)
    if have > 0
        earliest = maximum(skipmissing(Dates.value(kickoff - t) ÷ 60_000
                                       for t in cov.first_tick if t !== nothing))
        @printf("  %-14s %d/%d fixtures priced, book reaches back T−%d min\n",
                "order book", have, length(fixtures), earliest)
    else
        @printf("  %-14s NO ORDER BOOK for any fixture\n", "order book")
    end

    gradeable = length(results) == length(fixtures)
    @printf("  %-14s %s\n", "results",
            gradeable ? "all $(length(results)) present — P&L available" :
            "$(length(results))/$(length(fixtures)) present — P&L will be `missing`, not zero")
    println("="^92)

    spec = MD.MatchDaySpec(
        fixtures = MD.ExplicitFixtures(fixtures),
        identity = MD.MatchMetaCrosswalk(),
        lineups  = MD.SourceChain(MD.ProvisionalDB(), MD.LastHistorical(ds)),
        markets  = markets === nothing ? MD.MatchDaySpec().markets : markets,
        gate     = MD.GateChain(MD.IdentityResolved(), MD.MaxBookAge(book_age)))

    return MatchDayContext(segment, day, ds, expr, fixtures, results, spec,
                           MD.ArchivedOrderBook(), kickoff, gradeable)
end

"""
    policy(; trust = 0.5, lambda = 23.0, cap = 0.25, filter = nothing, mode = :sequential)

A `PolicySpec` without the ceremony. Everything here is a pure multiplier on an already-built
book, so switching between these is milliseconds.

* `trust`  — how much you believe the model over the market. A `Float64`, or a trust MODEL
             (e.g. `family_trust()` to zero out 1X2).
             ⚠ A flat number is very often a NO-OP: `risk_factor` is homogeneous of degree 0, so
             once the drawdown constraint binds, trust 0.25 and 0.5 give bit-identical books.
             It only does work when it DIFFERS between selections.
* `lambda` — the drawdown budget, and **the dial that actually moves exposure**. Higher = tighter
             = smaller book. 23 ≈ a 20% drawdown at 1% probability.
* `cap`    — hard ceiling on simultaneous exposure, as a fraction of bankroll.
* `filter` — curation. Applied LAST, after the cap, so it can only remove exposure and never
             redistributes it. Prefer a per-family `trust` if you want the freed capacity reused.
"""
function policy(; trust = 0.5, lambda::Real = 23.0, cap::Real = 0.25,
                filter = nothing, mode::Symbol = :sequential)
    PF = _pf()
    t = trust isa Real ? PF.FlatTrust(Float64(trust)) : trust
    return PF.PolicySpec(trust = t,
                         risk = PF.SlateDrawdown(lambda = Float64(lambda), mode = mode),
                         cap = PF.FixedCap(Float64(cap)),
                         filter = filter === nothing ? PF.KeepAll() : filter)
end

"""
    show_book(ctx; as_of = ctx.kickoff, match_id = nothing) -> DataFrame

The market book: every quoted selection with its best back and lay price **and the size
available at each**.

Read the size columns, not just the prices. Nothing in `src` does — `BestAvailable` takes the
price and discards the depth — so this is the only place the binding constraint on a thin league
is visible.
"""
function show_book(ctx::MatchDayContext; as_of::DateTime = ctx.kickoff, match_id = nothing)
    MD = _md()
    rows = NamedTuple[]
    for f in ctx.fixtures
        match_id === nothing || f.m_id == match_id || continue
        id = MD.resolve(ctx.spec.identity, f)
        id isa MD.Resolved || continue
        for (k, b) in MD.quotes(ctx.book, id, as_of)
            push!(rows, (match_id = f.m_id, fixture = "$(f.home) v $(f.away)",
                         market = k.group * (k.line == 0.0 ? "" : " $(k.line)"),
                         selection = k.selection,
                         back = MD.best_back(b),
                         back_size = isempty(b.back_size) ? 0.0 : b.back_size[1],
                         lay = MD.best_lay(b),
                         lay_size = isempty(b.lay_size) ? 0.0 : b.lay_size[1],
                         matched = b.matched))
        end
    end
    isempty(rows) && return DataFrame()
    return sort!(DataFrame(rows), [:match_id, :market, :selection])
end

# ───────────────────────────────────────────────────────────────────────────────────────────
# Internals: one expensive snapshot per instant, then policies are free
# ───────────────────────────────────────────────────────────────────────────────────────────
#
# `match_day` recomputes the posterior latents on every call, and that is the expensive part
# (~9s). But latents do not depend on the POLICY — only on `as_of`. So we build one snapshot per
# instant and re-stake it under every policy, which is exactly the BookSpec/PolicySpec split the
# Portfolio module is designed around.

function _snapshot(ctx::MatchDayContext, as_of::DateTime)
    MD = _md()
    cards = MD.build_cards(ctx.spec, ctx.segment, as_of)
    isempty(cards) && return nothing
    odds, insts = MD.price_cards(ctx.spec, cards, as_of)
    for c in cards
        c.readiness = MD.ready(ctx.spec.gate, c)
    end
    passed = [c for c in cards if MD.is_ready(c.readiness)]
    isempty(passed) && return nothing
    latents, _ = MD.matchday_latents(ctx.spec, ctx.expr, ctx.ds, passed, odds, as_of)
    isempty(latents) && return nothing
    return (latents = latents, odds = odds, insts = insts,
            fixtures = MD.fixture_info(passed), n_passed = length(passed),
            n_blocked = length(cards) - length(passed))
end

function _stake(ctx::MatchDayContext, snap, pol, bankroll::Float64)
    MD, PF = _md(), _pf()
    sys = PF.PortfolioSystem(PF.BookSpec(markets = ctx.spec.markets), pol)
    s = PF.stake_sheet(sys, snap.latents, ctx.expr, snap.odds, snap.fixtures; bankroll = bankroll)
    isempty(s) && return s
    MD._attach_instruments!(s, snap.insts, ctx.spec.rounding)
    isempty(s) && return s
    grade!(s, ctx.results, sys)
    return s
end

_as_of(ctx, mins) = ctx.kickoff - Minute(mins)

"""
    sheet_for(ctx, pol; as_of = ctx.kickoff, bankroll = 1000.0) -> DataFrame

One full graded stake sheet. Use when you want the per-leg detail rather than the summary.

Columns worth knowing: `selection` is the position, `venue_selection` is the runner the ORDER
touches (they differ on every lay), `odds` are EFFECTIVE odds, `risk` is what is at stake, and
`venue_stake` is what you place at the exchange.
"""
function sheet_for(ctx::MatchDayContext, pol; as_of::DateTime = ctx.kickoff,
                   bankroll::Real = 1000.0)
    snap = _snapshot(ctx, as_of)
    snap === nothing && return DataFrame()
    return _stake(ctx, snap, pol, Float64(bankroll))
end

"""
    returns(ctx, policies; times = [0], bankroll = 1000.0) -> DataFrame

**The answer table.** One row per (policy, time): how many legs, how much was at risk, what it
returned, and the growth.

`policies` is a `Vector{Pair{String,PolicySpec}}` — label first, so the output reads.
`times` are minutes before kickoff.

Costs one posterior rebuild per TIME, not per (policy, time): policies are pure multipliers on a
book that has already been built.

**Judge on `growth`, not `roi`.** ROI is P/L over stake, so any uniform rescaling of the book
cancels out of it exactly — every flat-trust setting reports the same ROI for very different
bankroll outcomes. `pnl` and `roi` come back `missing` when the fixtures have no result yet,
which is not the same as zero.
"""
function returns(ctx::MatchDayContext, policies::Vector{<:Pair};
                 times = [0], bankroll::Real = 1000.0)
    B = Float64(bankroll)
    rows = NamedTuple[]
    for m in times
        as_of = _as_of(ctx, m)
        snap  = _snapshot(ctx, as_of)
        if snap === nothing
            @warn "no priceable book at T−$m — skipping"
            continue
        end
        for (label, pol) in policies
            s = _stake(ctx, snap, pol, B)
            if isempty(s)
                push!(rows, (policy = label, mins_to_ko = m, fixtures = 0, legs = 0,
                             staked = 0.0, exposure = 0.0, pnl = missing, roi = missing,
                             growth = missing, k_risk = NaN, capped = false))
                continue
            end
            staked = sum(s.risk)
            pnl    = ctx.gradeable ? sum(s.pnl) : missing
            push!(rows, (policy = label, mins_to_ko = m,
                         fixtures = length(unique(s.match_id)), legs = nrow(s),
                         staked = round(staked, digits = 2),
                         exposure = round(staked / B, digits = 4),
                         pnl = pnl === missing ? missing : round(pnl, digits = 2),
                         roi = pnl === missing ? missing : round(pnl / staked, digits = 4),
                         growth = pnl === missing ? missing : round(log1p(pnl / B), digits = 4),
                         k_risk = round(first(s.k_risk), digits = 4),
                         capped = first(s.capped)))
        end
    end
    return DataFrame(rows)
end

"""
    stake_columns(ctx, policies; as_of = ctx.kickoff, bankroll = 1000.0) -> DataFrame

One row per leg, **one stake column per policy**, side by side — so you can see which policy
took which position and at what size.

A blank (`0.0`) means that policy did not take the leg at all. The identifying columns
(`odds`, `p_model`, `p_market`, `edge`) come from the first policy that priced the leg; they are
properties of the BOOK and the MODEL, so they are identical across policies by construction.
"""
function stake_columns(ctx::MatchDayContext, policies::Vector{<:Pair};
                       as_of::DateTime = ctx.kickoff, bankroll::Real = 1000.0)
    snap = _snapshot(ctx, as_of)
    snap === nothing && return DataFrame()

    base, seen = nothing, Dict{Tuple,NamedTuple}()
    cols = Pair{String,Dict{Tuple,Float64}}[]
    for (label, pol) in policies
        s = _stake(ctx, snap, pol, Float64(bankroll))
        d = Dict{Tuple,Float64}()
        for r in eachrow(s)
            k = (r.match_id, r.group, r.line, r.selection)
            d[k] = r.risk
            haskey(seen, k) || (seen[k] = (odds = r.odds, side = r.side,
                                           p_model = r.p_model, p_market = r.p_market,
                                           edge = r.edge, graded = r.graded))
        end
        push!(cols, label => d)
        base === nothing && (base = s)
    end
    isempty(seen) && return DataFrame()

    ks  = sort(collect(keys(seen)), by = k -> -maximum(get(d, k, 0.0) for (_, d) in cols))
    out = DataFrame(match_id = [k[1] for k in ks], market = [k[2] for k in ks],
                    line = [k[3] for k in ks], selection = [k[4] for k in ks],
                    side = [seen[k].side for k in ks],
                    odds = [round(seen[k].odds, digits = 3) for k in ks],
                    edge = [round(seen[k].edge, digits = 3) for k in ks],
                    won = [seen[k].graded for k in ks])
    for (label, d) in cols
        out[!, Symbol(label)] = [round(get(d, k, 0.0), digits = 2) for k in ks]
    end
    return out
end
