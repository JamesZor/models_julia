# current_development/orderbook_layer2/l04_corpus_replay.jl
#
# WP3. One slate replayed becomes a season replayed, and the expensive half is cached.
#
# ---------------------------------------------------------------------------------------------
# WHAT THIS ADDS TO `matchday_2026_08_08/l02_slate_replay.jl`
# ---------------------------------------------------------------------------------------------
#
# That file is well built and is reused wholesale where it fits (`replay_spec`, `grade!`, the
# gate/blocked plumbing). It has two limits this stream cannot live with:
#
# 1. **One settlement window.** `snapshot_grid` anchors on the earliest kick-off of a single
#    slate and `replay` loops one slate. The Ireland corpus is 12 windows.
# 2. **`match_day` per snapshot.** That call does everything — cards, book, gates, latents,
#    staking — so re-staking under a different trust model means re-reading the database and
#    re-extracting posteriors. At ~28 snapshots x 12 slates x 2 leagues that is ~670 calls per
#    policy, which turns a `groupby`-sized question into a day of compute.
#
# So this file **decomposes `match_day` along its own seam**:
#
#     TIER 1  build_cards -> price_cards -> gates -> matchday_latents      EXPENSIVE, cached
#     TIER 2  Portfolio.stake_sheet -> _attach_instruments!                CHEAP, re-run freely
#
# The decomposition is the same one `l05_simple.jl`'s facade makes, and `r03`'s gate asserts it
# reproduces `match_day` row for row. That assertion is the entire licence for doing this: an
# optimised replica that drifts from the pipeline measures the replica.
#
# ---------------------------------------------------------------------------------------------
# THE FROZEN / LIVE ARMS — MEASURED TO BE THE SAME ARM (2026-08-12)
# ---------------------------------------------------------------------------------------------
#
# This section originally argued that `src_sup40_sw40`, being player-level, must have latents
# that move with `as_of` via `RatingsFromTracker` and the announced XI — so a single-arm trace
# would confound model drift with market drift.
#
# **That argument was wrong, and WP3 measured it wrong.** On the 2026-05-29 slate, comparing
# latents at T-120 against latents at kick-off:
#
#     every column (true_xg_h/a, θ_1..3, λ_h, λ_a, λ_tot, ρ, φ), every fixture,
#     all 3,200 posterior draws:  max |Δ| = 0.0    (bit-identical)
#
# Serving-time latents for this engine are a pure function of `(fixture, split)`. `as_of` and the
# book do not enter: the market pillar was a TRAINING-time regulariser, and `replay_spec` wires
# `lineups = SourceChain()` with no sources, so no XI is ever fetched. Player-level in the
# posterior does not imply clock-dependent at serve time.
#
# Consequences, all of which the rest of this stream depends on:
#
#   * `:live` and `:frozen` produce IDENTICAL ledgers. `:live` is pure cost.
#   * therefore 100% of movement in a replay IS the book — the clean reading the funnel harness
#     had, which this header previously claimed we could not have.
#   * `live - frozen` is identically zero, so it cannot measure the value of team news. Any such
#     study needs a point-in-time lineup source that this archive does not contain.
#
# The arm parameter is KEPT rather than deleted. It is correct machinery, it costs nothing while
# `:frozen` is the default, and it is the thing to reach for the moment an engine or a spec does
# consume `as_of`. `latent_delta` below is how you re-check that, cheaply, per engine.
#
# ⚠️ Do NOT verify this with `latents_invariant` (`matchday_2026_08_08/l02_slate_replay.jl:229`).
# It filters columns on `eltype(col) <: Number`, but latent columns hold POSTERIOR DRAWS —
# `Vector{Float64}` cells in an `Any`-eltype column — so the filter matches nothing, the loop
# body never runs, and it returns `(true, 0.0, :none)` no matter what the latents did. It is
# vacuous for any engine of this shape, and it reported a pass here on frames it never compared.
#
# ---------------------------------------------------------------------------------------------
# TWO THINGS THAT WILL BITE
# ---------------------------------------------------------------------------------------------
#
# T1. `_CARD_META` (`src/MatchDay/implementations/gates.jl:340`) is a module-level `IdDict` keyed
#     on mutable `FixtureCard`s and is NEVER cleared. Over ~670 snapshots it grows without bound.
#     `clear_card_meta!` is called between slates.
#
# T2. The corpus's DataStore must be the PINNED one from WP2. `matchday_latents` calls
#     `select_split`, which rebuilds boundaries from whatever store it is handed; a store that
#     grew since training makes folds mis-pair silently.

using DataFrames, Dates, Statistics, Printf

# ===================================================================
# 1. The grid
# ===================================================================

"""
    adaptive_grid(fixtures; lookback, fine_from, fine_step, coarse_step) -> Vector{DateTime}

Decision instants for one slate: coarse far out, at the feed's true cadence near the off.

Measured justification (WP0, see NOTES.md): spread is FLAT from T-240 to T-60 — MATCH_ODDS moves
2.86% -> 2.80% -> 2.86% across those buckets — and only tightens inside the last hour. Sampling
the flat region at 3-minute resolution buys nothing and costs 40 extra `match_day` calls per
slate. The feed's real cadence is 3 minutes despite the table being named `order_book_1m`, so
`fine_step` finer than that just re-reads the same tick.

Anchored on the EARLIEST kick-off in the slate, because `ExplicitFixtures` silently drops a
fixture once `as_of > kickoff` — an anchor on the latest would shrink the slate mid-trace and
make exposure incomparable across snapshots.
"""
function adaptive_grid(fixtures;
                       lookback::Period   = Minute(180),
                       fine_from::Period  = Minute(60),
                       fine_step::Period   = Minute(3),
                       coarse_step::Period = Minute(15))
    ko    = minimum(f.kickoff for f in fixtures)
    out   = DateTime[]
    t     = ko - lookback
    fine0 = ko - fine_from
    while t < fine0
        push!(out, t); t += coarse_step
    end
    t = fine0
    while t <= ko
        push!(out, t); t += fine_step
    end
    return unique!(sort!(out))
end

"""
    latent_delta(spec, expr, ds, fixtures, t1, t2) -> NamedTuple

Do this engine's serving latents actually move between two instants?

The replacement for `latents_invariant`, which cannot answer the question: it selects columns
with `eltype(col) <: Number`, and a latents frame stores POSTERIOR DRAWS — each cell a
`Vector{Float64}` (or a `Matrix` for `φ`) inside an `Any`-eltype column. Nothing matches, the
comparison loop never executes, and it returns `(true, 0.0, :none)` unconditionally. A test that
cannot fail is not evidence, and it reported a pass here on frames it never looked at.

This version compares cell CONTENTS, so a per-draw difference of 1e-12 is still caught, and it
gates on the cards the readiness gate actually passes — comparing a populated frame against an
empty one is not a measurement either.

Returns the worst absolute difference and the column carrying it; `moved = false` with
`n_compared > 0` is a real invariance result rather than a vacuous one.
"""
function latent_delta(spec, expr, ds, t1::DateTime, t2::DateTime)
    MD = _md()
    function at(t)
        cards = MD.build_cards(spec, nothing, t)
        odds, _ = MD.price_cards(spec, cards, t)
        for c in cards
            c.readiness = MD.ready(spec.gate, c)
        end
        passed = filter(c -> MD.is_ready(c.readiness), cards)
        isempty(passed) && return nothing
        l, _ = MD.matchday_latents(spec, expr, ds, passed, odds, t)
        return isempty(l) ? nothing : sort(l, :match_id)
    end

    a, b = at(t1), at(t2)
    (a === nothing || b === nothing) &&
        return (moved = false, worst = NaN, col = :no_passing_cards, n_compared = 0)
    a.match_id == b.match_id ||
        return (moved = true, worst = NaN, col = :match_id_mismatch, n_compared = 0)

    worst, wcol, n = 0.0, :none, 0
    for c in names(a)
        c == "match_id" && continue
        for i in 1:nrow(a)
            d = maximum(abs.(a[i, c] .- b[i, c]))
            n += 1
            d > worst && (worst = d; wcol = Symbol(c))
        end
    end
    return (moved = worst > 1e-10, worst = worst, col = wcol, n_compared = n)
end

"Drop the module-level card-metadata sidecar between slates. See trap T1."
function clear_card_meta!()
    MD = _md()
    try
        empty!(getfield(MD, :_CARD_META))
    catch
        # name is internal; if it moves, a growing IdDict is a leak, not a wrong answer
        @debug "clear_card_meta!: _CARD_META not reachable"
    end
    return nothing
end

# ===================================================================
# 2. Tier 1 — the expensive cache
# ===================================================================

"""
    L2Snapshot

Everything `match_day` computes for one slate at one instant, BEFORE staking.

Holding `latents` and `odds` together is the point: an engine like `src_sup40_sw40` takes market
odds as a model feature, so the posterior and the price it will be compared against must come
from the same read. If they ever come from different reads, every diagnostic compares the model
against a price it was not given.
"""
struct L2Snapshot
    slate_day::Date
    as_of::DateTime
    odds::DataFrame
    latents::DataFrame
    # MUST stay `Dict{Int,Portfolio.FixtureInfo}`, NOT widened to `Dict{Int,Any}`. `stake_sheet`
    # dispatches on exactly this type to reach its live-fixture method
    # (`src/Portfolio/matchday.jl:63`); anything else falls through to the DataStore method,
    # which calls `fixture_table(ds)` and whose own docstring warns it "returns an empty sheet
    # for any fixture that has not been played". Widening the field routed every snapshot to the
    # wrong method. It happened to throw (a Dict has no `.matches`) — had it not, the replay
    # would have silently returned empty sheets for exactly the unplayed fixtures we are here
    # to price.
    fixtures::Dict{Int,BayesianFootball.Portfolio.FixtureInfo}
    instruments::Dict{Any,Any}
    depth::DataFrame
    blocked::DataFrame
    n_passed::Int
    split_warning::String
end

"""
    L2Snapshots

A whole league's replay, one `L2Snapshot` per (slate, instant), plus the frozen results.

This is Tier 1. Build it once; stake it as many times as you like.
"""
struct L2Snapshots
    corpus_name::String
    tournament_id::Int
    arm::Symbol
    snaps::Vector{L2Snapshot}
    results::Dict{Int,Tuple{Int,Int}}
    grid::NamedTuple
    built_at::DateTime
end

Base.length(s::L2Snapshots) = length(s.snaps)
Base.isempty(s::L2Snapshots) = isempty(s.snaps)

function Base.show(io::IO, ::MIME"text/plain", s::L2Snapshots)
    println(io, "L2Snapshots \"$(s.corpus_name)\"  tournament $(s.tournament_id)  arm :$(s.arm)")
    println(io, "├─ snapshots  $(length(s.snaps))")
    if !isempty(s.snaps)
        days = unique(x.slate_day for x in s.snaps)
        println(io, "├─ slates     $(length(days))  ($(minimum(days)) .. $(maximum(days)))")
        println(io, "├─ priced     $(sum(nrow(x.odds) for x in s.snaps)) quote-rows")
        nw = count(x -> !isempty(x.split_warning), s.snaps)
        nw == 0 || println(io, "├─ WARNINGS   $nw snapshots carry a split warning")
    end
    print(io, "└─ grid       $(s.grid)")
end

"""
    build_snapshots(corpus, expr, ds; arm, grid_kw..., bankroll) -> L2Snapshots

Replay one league's whole corpus, caching everything up to (but not including) staking.

`arm = :frozen` computes latents once per slate at its earliest instant and reuses them at every
later instant; `:live` recomputes per instant. See the header for why this engine needs both.

`ds` **must** be the pinned DataStore from WP2 — see trap T2.
"""
function build_snapshots(corpus, expr, ds;
                         arm::Symbol = :frozen,
                         lookback::Period = Minute(180),
                         fine_from::Period = Minute(60),
                         fine_step::Period = Minute(3),
                         coarse_step::Period = Minute(15),
                         verbose::Bool = true)
    arm in (:frozen, :live) || error("build_snapshots: arm must be :frozen or :live, got :$arm")
    MD = _md()
    snaps = L2Snapshot[]
    slates = corpus_slates(corpus)

    for (si, sl) in enumerate(slates)
        clear_card_meta!()
        grid = adaptive_grid(sl.fixtures; lookback = lookback, fine_from = fine_from,
                             fine_step = fine_step, coarse_step = coarse_step)
        spec = replay_spec(sl.fixtures)

        frozen_lat, frozen_warn = nothing, ""
        verbose && @printf("[%2d/%2d] %s  %2d fixtures  %2d instants (%s)\n",
                           si, length(slates), sl.day, length(sl.fixtures), length(grid), arm)

        for t in grid
            cards = MD.build_cards(spec, nothing, t)
            isempty(cards) && continue
            odds, insts = MD.price_cards(spec, cards, t)
            for c in cards
                c.readiness = MD.ready(spec.gate, c)
            end
            passed  = [c for c in cards if MD.is_ready(c.readiness)]
            blocked = [c for c in cards if !MD.is_ready(c.readiness)]
            isempty(passed) && continue

            lat, warn = if arm === :live
                l, d = MD.matchday_latents(spec, expr, ds, passed, odds, t)
                (l, d.warning)
            else
                if frozen_lat === nothing
                    l, d = MD.matchday_latents(spec, expr, ds, passed, odds, t)
                    frozen_lat, frozen_warn = l, d.warning
                end
                (frozen_lat, frozen_warn)
            end
            isempty(lat) && continue

            push!(snaps, L2Snapshot(sl.day, t, odds, lat, MD.fixture_info(passed), insts,
                                    _depth_all_levels(spec, cards, t),
                                    _blocked_frame(blocked, t), length(passed), warn))
        end
    end

    return L2Snapshots(corpus.name, corpus.tournament_ids[1], arm, snaps, corpus.results,
                       (; lookback, fine_from, fine_step, coarse_step), now())
end

"""
    _depth_all_levels(spec, cards, t) -> DataFrame

Top-of-book AND the full ladder, per quoted selection.

`l02_slate_replay.jl`'s `_depth_snapshot` keeps only level 1. `FillCost` walks the ladder to
price a stake the top cannot absorb, so the whole `BookLevels` vector is carried. `back`/`lay`
are the vectors; `best_back`/`best_lay` are the scalars, so a caller that only wants the top
never has to index.
"""
function _depth_all_levels(spec, cards, t::DateTime)
    MD = _md()
    rows = NamedTuple[]
    for c in cards
        MD.resolved(c) || continue
        book = try
            MD.quotes(spec.book, c.identity, t)
        catch
            continue
        end
        for (k, b) in book
            push!(rows, (as_of = t, match_id = c.fixture.m_id, group = k.group, line = k.line,
                         selection = k.selection,
                         back = copy(b.back), back_size = copy(b.back_size),
                         lay = copy(b.lay),  lay_size = copy(b.lay_size),
                         best_back = MD.best_back(b), best_lay = MD.best_lay(b),
                         matched = b.matched, tick = b.ts))
        end
    end
    return isempty(rows) ? _empty_depth() : DataFrame(rows)
end

_empty_depth() = DataFrame(as_of = DateTime[], match_id = Int[], group = String[],
                           line = Float64[], selection = Symbol[],
                           back = Vector{Float64}[], back_size = Vector{Float64}[],
                           lay = Vector{Float64}[], lay_size = Vector{Float64}[],
                           best_back = Float64[], best_lay = Float64[],
                           matched = Float64[], tick = DateTime[])

function _blocked_frame(blocked, t::DateTime)
    rows = NamedTuple[]
    for c in blocked, (k, v) in c.readiness.reasons
        push!(rows, (as_of = t, match_id = c.fixture.m_id, gate = k, reason = v))
    end
    return isempty(rows) ?
        DataFrame(as_of = DateTime[], match_id = Int[], gate = Symbol[], reason = String[]) :
        DataFrame(rows)
end

# ===================================================================
# 3. Tier 2 — staking, cheap and repeatable
# ===================================================================

"""
    stake_snapshots(snapshots, sys, expr; bankroll, rounding) -> Layer2Ledger

Stake every cached snapshot under one `PortfolioSystem`, grade it, and stamp the closing price.

This is the whole of Tier 2. Re-running it under a different trust model or filter costs one
pass over an in-memory cache — no database, no posterior extraction — which is what makes WP5
and WP6 affordable.

The closing stamp (`:odds_close_final`, `:fair_close`) is applied HERE, once, from the last
pre-kickoff snapshot of each leg, rather than being re-derived inside each metric. Two metrics
that each derive their own close can silently disagree about what "the close" was.
"""
function stake_snapshots(snaps::L2Snapshots, sys, expr;
                         bankroll::Real = 1.0, rounding = nothing,
                         policy_name::String = "default")
    MD, PF = _md(), _pf()
    isempty(snaps) && return Layer2Ledger(DataFrame())
    rnd = rounding === nothing ? MD.NoMinimum() : rounding

    # Assert the dispatch, do not assume it. `stake_sheet`'s live method is selected by the
    # EXACT type of this argument; every other type reaches the DataStore method, which returns
    # an empty sheet for unplayed fixtures. That failure mode is silent by construction, so it
    # is checked once here rather than left to be noticed in a ROI column.
    let ft = eltype(values(first(snaps.snaps).fixtures))
        ft === PF.FixtureInfo || error(
            "stake_snapshots: fixtures dict has value type $ft, not Portfolio.FixtureInfo — " *
            "stake_sheet would silently take its DataStore method and drop unplayed fixtures")
    end

    parts = DataFrame[]
    for s in snaps.snaps
        sheet = PF.stake_sheet(sys, s.latents, expr, s.odds, s.fixtures; bankroll = bankroll)
        isempty(sheet) && continue
        MD._attach_instruments!(sheet, s.instruments, rnd)
        isempty(sheet) && continue

        sheet.as_of      = fill(s.as_of, nrow(sheet))
        sheet.slate_day  = fill(s.slate_day, nrow(sheet))
        push!(parts, sheet)
    end
    isempty(parts) && return Layer2Ledger(DataFrame())

    led = reduce(vcat, parts; cols = :union)

    # minutes to kickoff, from the corpus's own kickoff times
    ko = Dict{Int,DateTime}()
    for s in snaps.snaps, (mid, fi) in s.fixtures
        haskey(ko, mid) || (ko[mid] = _fixture_kickoff(fi, s))
    end
    led.mins_to_ko = [Dates.value(ko[m] - a) / 60_000 for (m, a) in zip(led.match_id, led.as_of)]
    add_entry_buckets!(led)

    grade!(led, snaps.results, sys)
    _stamp_close!(led, snaps)
    _attach_depth!(led, snaps)

    led.policy_name   = fill(policy_name, nrow(led))
    led.arm           = fill(snaps.arm, nrow(led))
    led.tournament_id = fill(snaps.tournament_id, nrow(led))
    led.is_winner     = [ismissing(g) ? missing : g for g in led.graded]
    led.payoff        = led.unit_payoff
    led.stake_cash    = led.risk
    led.pnl_cash      = led.pnl
    # Units: Portfolio's `frac` is the bankroll fraction that compounds. `risk` is post-rounding
    # currency. `l2_curve` compounds `:pnl`, so `:stake`/`:pnl` MUST be the fractional pair.
    led.stake         = led.frac
    led.pnl           = led.frac .* led.unit_payoff
    led.slate         = led.slate_day

    return Layer2Ledger(led)
end

_fixture_kickoff(fi, s) = hasproperty(fi, :kickoff) ? fi.kickoff : DateTime(s.slate_day)

"""
    _stamp_close!(ledger, snapshots)

Broadcast the LAST pre-kickoff quote of each leg onto every earlier row.

`:odds_close_final` is the effective odds at the final snapshot; `:fair_close` is the de-vigged
closing probability of the same selection, taken from the market group's overround at that
instant. `PriceDrift` and `ClosingLineValue` both read these, so stamping once is what keeps
them from disagreeing.
"""
function _stamp_close!(led::DataFrame, snaps::L2Snapshots)
    last_odds = Dict{Tuple{Int,String,Float64,Symbol},Float64}()
    last_seen = Dict{Tuple{Int,String,Float64,Symbol},DateTime}()
    grp_sum   = Dict{Tuple{Int,String,Float64},Float64}()
    grp_when  = Dict{Tuple{Int,String,Float64},DateTime}()

    for s in snaps.snaps, r in eachrow(s.odds)
        k = (r.match_id, String(r.market_name), Float64(r.market_line), Symbol(r.selection))
        if !haskey(last_seen, k) || s.as_of > last_seen[k]
            last_seen[k] = s.as_of
            last_odds[k] = Float64(r.odds_close)
        end
    end
    # overround per market group at its own final instant
    for s in snaps.snaps
        for sub in groupby(s.odds, [:match_id, :market_name, :market_line])
            g = (sub.match_id[1], String(sub.market_name[1]), Float64(sub.market_line[1]))
            if !haskey(grp_when, g) || s.as_of > grp_when[g]
                grp_when[g] = s.as_of
                grp_sum[g]  = sum(1.0 ./ Float64.(sub.odds_close))
            end
        end
    end

    n = nrow(led)
    oc = Vector{Float64}(undef, n); fc = Vector{Float64}(undef, n)
    for i in 1:n
        k = (led.match_id[i], led.group[i], led.line[i], led.selection[i])
        g = (led.match_id[i], led.group[i], led.line[i])
        o = get(last_odds, k, NaN)
        oc[i] = o
        ov    = get(grp_sum, g, NaN)
        fc[i] = (isfinite(o) && isfinite(ov) && ov > 0) ? (1.0 / o) / ov : NaN
    end
    led.odds_close_final = oc
    led.fair_close       = fc
    return led
end

"""
    _attach_depth!(ledger, snapshots)

Join the ladder for the exact `(as_of, leg)` each row was staked at, so `FillCost` sees the book
the order would actually have hit — not a nearby snapshot's.
"""
function _attach_depth!(led::DataFrame, snaps::L2Snapshots)
    idx = Dict{Tuple{DateTime,Int,String,Float64,Symbol},Int}()
    all_depth = isempty(snaps.snaps) ? _empty_depth() :
                reduce(vcat, [s.depth for s in snaps.snaps]; cols = :union)
    for (i, r) in enumerate(eachrow(all_depth))
        idx[(r.as_of, r.match_id, r.group, r.line, r.selection)] = i
    end

    n = nrow(led)
    bk = Vector{Vector{Float64}}(undef, n); bs = Vector{Vector{Float64}}(undef, n)
    ly = Vector{Vector{Float64}}(undef, n); ls = Vector{Vector{Float64}}(undef, n)
    mt = Vector{Float64}(undef, n);         sp = Vector{Float64}(undef, n)

    for i in 1:n
        j = get(idx, (led.as_of[i], led.match_id[i], led.group[i], led.line[i],
                      led.selection[i]), 0)
        if j == 0
            bk[i] = Float64[]; bs[i] = Float64[]; ly[i] = Float64[]; ls[i] = Float64[]
            mt[i] = NaN;       sp[i] = NaN
            continue
        end
        r = all_depth[j, :]
        bk[i] = r.back; bs[i] = r.back_size; ly[i] = r.lay; ls[i] = r.lay_size
        mt[i] = r.matched
        sp[i] = (isfinite(r.best_back) && isfinite(r.best_lay) && r.best_back > 0) ?
                (r.best_lay - r.best_back) / r.best_back : NaN
    end
    led.back = bk; led.back_size = bs; led.lay = ly; led.lay_size = ls
    led.matched = mt; led.rel_spread = sp
    return led
end

# ===================================================================
# 4. Running an L2Config end to end
# ===================================================================


"""
    run_l2_experiment(task) -> L2Results

Tier 2 + Tier 3: stake the cached snapshots, then apply the config's entry rule.

`recap_slates!` runs after the entry rule because `FirstQualifying` and `BestPrice` assemble legs
from different instants, where the per-snapshot exposure cap no longer holds — see
`l01_l2_experiment.jl`'s header. `:recapped` reports when it bound; a single-instant rule must
never trip it, which is the check that the repair leaves the baselines alone.
"""
function run_l2_experiment(task::L2Task; bankroll::Real = 1.0, expr = nothing)
    cfg = task.config
    e   = expr === nothing ? task.corpus : expr
    full = stake_snapshots(task.snapshots, cfg.sys, e;
                           bankroll = bankroll, policy_name = cfg.name)
    picked = apply_entry(cfg.entry, full.df)
    recap_slates!(picked, cap_fraction(cfg.sys.policy.cap))
    picked.entry_name = fill(entry_name(cfg.entry), nrow(picked))

    led = Layer2Ledger(picked)
    diag = Dict{Symbol,Any}(
        :n_snapshots => length(task.snapshots),
        :legs_all_instants => nrow(full.df),
        :legs_after_entry  => nrow(picked),
        :recapped => hasproperty(picked, :recapped) ? count(picked.recapped) : 0)

    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    return L2Results(cfg, picked, nothing, diag,
                     joinpath(cfg.save_dir, "$(cfg.name)_$(stamp)"))
end
