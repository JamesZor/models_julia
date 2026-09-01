# src/MatchDay/pipeline.jl
#
# The single parameterised entry point that replaces "copy last week's runner and edit the date".
#
# Stage order is deliberately NOT numerical: the book is built before features, because
# market-pillar engines consume odds as a model feature. Any diagram of this as a straight line
# is wrong.

export match_day, fixture_info, build_cards, price_cards, quote_slate,
       relative_spread, order_ticket

"""
    fixture_info(cards) -> Dict{Int,Portfolio.FixtureInfo}

The fixture table Portfolio needs, built from live fixtures rather than from `ds.matches`.

This is the fix for the seam that made `Portfolio.stake_sheet` unreachable: `fixture_table(ds)`
is derived from the curated store of *finished* matches, so it contains no entry with a
`nothing` score and an upcoming fixture is absent from it entirely -- `build_book` then returned
`nothing` for every one and the sheet came back empty, silently.
"""
fixture_info(cards::Vector{<:FixtureCard}) =
    Dict{Int,Portfolio.FixtureInfo}(
        c.fixture.m_id => (date = Date(c.fixture.kickoff), score = nothing) for c in cards)

"""
    build_cards(spec, segment, as_of) -> Vector{FixtureCard}

Stages 1-3: fixtures, identity, lineups. Every fixture gets a card, including unresolved ones.
"""
function build_cards(spec::MatchDaySpec, segment, as_of::DateTime)
    fx = fixtures(spec.fixtures, segment, as_of)
    cards = FixtureCard[]
    for f in fx
        id = resolve(spec.identity, f)
        c  = FixtureCard(f, id, as_of)
        c.lineup = lineup(spec.lineups, f, as_of)
        push!(cards, c)
    end
    return cards
end

"""
    price_cards(spec, cards, as_of) -> (odds_df, instruments)

Stage 5': quote every resolved card, choose an instrument per selection, and emit the DataFrame
Portfolio expects (`:match_id, :market_name, :market_line, :selection, :odds_close`).

`odds_close` carries the **effective** odds of the chosen instrument. For a lay that is
`d/(d-1)`, denominated so a unit of stake is a unit of risk -- which is what lets every
downstream component stay unaware that lays exist.

Also stamps book age and matched volume onto each card for the gates.
"""
price_cards(spec::MatchDaySpec, cards::Vector{<:FixtureCard}, as_of::DateTime) =
    let q = quote_slate(spec, cards, as_of)
        (q.odds, q.instruments)
    end

"""
    quote_slate(spec, cards, as_of) -> (; odds, instruments, books)

`price_cards` plus the **depth it collapsed**.

`price_cards` returns only what `Portfolio` needs, which is one scalar price per selection. The
ledger's fill model and the console's ladder both need the levels that price came from, and
re-reading them is a second database round trip against a book that has moved. `books` is keyed
`(match_id, SelectionKey)` so a sheet row reaches its own depth without a second lookup table.

Stamps four pieces of card metadata for the gates: `:book_age`, `:max_matched`,
`:spread_median` and `:spread_max`, all as of this quote.
"""
function quote_slate(spec::MatchDaySpec, cards::Vector{<:FixtureCard}, as_of::DateTime)
    rows  = NamedTuple[]
    insts = Dict{Tuple{Int,SelectionKey},Instrument}()
    books = Dict{Tuple{Int,SelectionKey},BookLevels}()

    for c in cards
        resolved(c) || continue
        book = quotes(spec.book, c.identity, as_of)
        if isempty(book)
            _set_card_meta!(c, :book_age, nothing)
            continue
        end
        _set_card_meta!(c, :book_age, as_of - maximum(b.ts for b in values(book)))
        matched = [b.matched for b in values(book) if !isnan(b.matched)]
        _set_card_meta!(c, :max_matched, isempty(matched) ? NaN : maximum(matched))

        spreads = [s for s in (relative_spread(b) for b in values(book)) if !isnan(s)]
        _set_card_meta!(c, :spread_median, isempty(spreads) ? NaN : median(spreads))
        _set_card_meta!(c, :spread_max,    isempty(spreads) ? NaN : maximum(spreads))

        ks = collect(keys(book))
        for key in ks
            books[(c.fixture.m_id, key)] = book[key]
            comp = complement_of(key, ks)
            inst = instrument(spec.instrument, key, comp, book, spec.quote_rule)
            inst === nothing && continue
            insts[(c.fixture.m_id, key)] = inst
            push!(rows, (match_id = c.fixture.m_id, market_name = key.group,
                         market_line = key.line, selection = key.selection,
                         odds_close = inst.odds))
        end
    end

    odds = isempty(rows) ? _empty_odds() : DataFrame(rows)
    return (; odds, instruments = insts, books)
end

"""
    relative_spread(b::BookLevels) -> Float64

`(lay - back) / mid` at the touch. `NaN` when either side is empty or the book is crossed --
callers treat `NaN` as "not measurable" rather than as zero, because a one-sided book is the
widest book there is, not the tightest.
"""
function relative_spread(b::BookLevels)
    bb, bl = best_back(b), best_lay(b)
    (isnan(bb) || isnan(bl) || bl <= bb) && return NaN
    return (bl - bb) / ((bl + bb) / 2)
end

_empty_odds() = DataFrame(match_id = Int[], market_name = String[], market_line = Float64[],
                          selection = Symbol[], odds_close = Float64[])

"""
    match_day(spec, sys, segment, expr, ds; as_of = now(), bankroll = 1.0) -> MatchDayResult

Run the whole pipeline.

`as_of` defaults **at the call site only**; no stage reads the clock internally, which is what
makes a past match day replayable and a decision auditable.

A refusal is a value: `result.blocked` carries every card the gate stopped and why, so "no bets
today" and "the pipeline is broken" are never the same empty DataFrame.
"""
function match_day(spec::MatchDaySpec, sys::Portfolio.PortfolioSystem, segment, expr, ds;
                   as_of::DateTime = now(), bankroll::Real = 1.0)
    cards = build_cards(spec, segment, as_of)
    isempty(cards) && return MatchDayResult(_empty_sheet(), FixtureCard[], FixtureCard[],
                                            _empty_odds(), Dict(), as_of)

    odds, insts = price_cards(spec, cards, as_of)

    for c in cards
        c.readiness = ready(spec.gate, c)
    end
    passed  = FixtureCard[c for c in cards if is_ready(c.readiness)]
    blocked = FixtureCard[c for c in cards if !is_ready(c.readiness)]

    isempty(passed) && return MatchDayResult(_empty_sheet(), cards, blocked, odds, insts, as_of)

    latents, diag = matchday_latents(spec, expr, ds, passed, odds, as_of)
    isempty(diag.warning) || @warn "MatchDay: $(diag.warning)"
    isempty(latents) && return MatchDayResult(_empty_sheet(), cards, blocked, odds, insts, as_of)

    sheet = Portfolio.stake_sheet(sys, latents, expr, odds, fixture_info(passed);
                                  bankroll = bankroll)

    if !isempty(sheet)
        _attach_instruments!(sheet, insts, spec.rounding)
    end
    return MatchDayResult(sheet, cards, blocked, odds, insts, as_of)
end

"""
Attach the execution columns and apply the exchange minimum.

`stake` from Portfolio is risk. `venue_stake` is what is actually placed, which for a lay is
`risk/(d-1)` -- so a lay at a short price clears a £1 minimum with far less than £1 at risk.

`venue_selection` is the runner the order touches, which for a synthetic is NOT `selection`.
It is carried as its own column rather than re-derived at ticket time so that a saved sheet is
executable on its own, without needing the `Instrument` dictionary that produced it.
"""
function _attach_instruments!(sheet::DataFrame, insts, rounding::AbstractStakeRounding)
    n = nrow(sheet)
    side  = Vector{Symbol}(undef, n); venue = Vector{Float64}(undef, n)
    vstk  = Vector{Float64}(undef, n); risk = Vector{Float64}(undef, n)
    vsel  = Vector{Symbol}(undef, n)

    for i in 1:n
        key = (group = sheet.group[i], line = sheet.line[i], selection = sheet.selection[i])
        inst = get(insts, (sheet.match_id[i], key), nothing)
        if inst === nothing
            side[i], venue[i], vstk[i], risk[i] = :back, sheet.odds[i], sheet.stake[i], sheet.stake[i]
            vsel[i] = sheet.selection[i]
            continue
        end
        r = round_stake(rounding, sheet.stake[i], inst)
        side[i]  = inst.side
        venue[i] = inst.venue_odds
        risk[i]  = r
        vstk[i]  = r <= 0 ? 0.0 : venue_stake(inst, r)
        vsel[i]  = inst.venue_key.selection
    end

    sheet.side = side
    sheet.venue_odds = venue
    sheet.venue_selection = vsel
    sheet.risk = risk
    sheet.venue_stake = vstk
    filter!(:risk => >(0.0), sheet)
    return sheet
end

_empty_sheet() = DataFrame(slate = Date[], match_id = Int[], family = String[], group = String[],
                           line = Float64[], selection = Symbol[], odds_quoted = Float64[],
                           odds = Float64[], p_model = Float64[], p_market = Float64[],
                           edge = Float64[], frac = Float64[], stake = Float64[],
                           k_risk = Float64[], slate_exposure = Float64[], capped = Bool[],
                           settled = Bool[], side = Symbol[], venue_odds = Float64[],
                           venue_selection = Symbol[], risk = Float64[], venue_stake = Float64[])

"""
    order_ticket(row) -> NamedTuple

What to actually place on the exchange for one sheet row. The last step, and the only place the
back/lay distinction becomes visible again.

`selection` is `venue_selection` -- **the runner the order touches** -- and NOT the model's
`selection`, which on a synthetic is its complement. The two are both reported, because they
answer different questions: `selection` is what you type into the exchange, `model_selection`
is the position it expresses and the key you grade against.

This distinction is the whole reason `Instrument` carries a `venue_key`. Emitting the model's
selection alongside the complement's side and price -- which this function used to do -- is an
instruction to place the OPPOSITE bet at a price belonging to the other runner.

Note `market` and `line` are unchanged by the morphism: a synthetic trades the other runner of
the SAME market, never a different one.
"""
order_ticket(row) = (match_id = row.match_id, market = row.group, line = row.line,
                     selection = row.venue_selection, side = row.side,
                     price = row.venue_odds, stake = row.venue_stake,
                     liability = row.side === :lay ? row.risk : row.venue_stake,
                     model_selection = row.selection)

"""
    blocked_report(result) -> DataFrame

Why each fixture was refused. Read this before concluding there were no bets.
"""
function blocked_report(r::MatchDayResult)
    rows = NamedTuple[]
    for c in r.blocked, (k, v) in c.readiness.reasons
        push!(rows, (match_id = c.fixture.m_id, fixture = "$(c.fixture.home) v $(c.fixture.away)",
                     kickoff = c.fixture.kickoff, gate = k, reason = v))
    end
    return isempty(rows) ? DataFrame(match_id = Int[], fixture = String[], kickoff = DateTime[],
                                     gate = Symbol[], reason = String[]) : DataFrame(rows)
end
