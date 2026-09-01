# src/MatchDay/console/snapshot.jl
#
# The console's read model.
#
# THE DASHBOARD OWNS NO STATE. It renders a `PricedSlate` plus a `PaperAccount`, both of which the
# match-day process is already holding or the ledger is already storing. That is what keeps it
# ~600 lines rather than a second system: there is no store to keep in sync, no reducer, no
# reconciliation between what the page believes and what the account holds.
#
# The payload has three parts and they answer three different questions, in this order:
#
#   account  -- can we afford anything?
#   batch    -- is THIS VECTOR safe to commit?   <- the header, and the reason it is read first
#   cards    -- which legs, and how good are they?
#
# Exposure before bets, always. A sheet is a list of attractive-looking prices; the only number
# that says whether it is safe to commit is what fraction of the bankroll settles at once and
# whether `FixedCap` had to bind to get it there.

export slate_snapshot, card_payload, batch_payload, account_payload

"""
    account_payload(a::PaperAccount) -> NamedTuple

Balance, reserved, equity. `free` is `balance`, named for what it is used for: the ceiling on the
next reservation.
"""
account_payload(a::PaperAccount) = (
    account_id = a.account_id, currency = a.currency,
    balance = round(a.balance, digits = 2), reserved = round(a.reserved, digits = 2),
    equity = round(equity(a), digits = 2), free = round(a.balance, digits = 2),
    max_slate_exposure = a.max_slate_exposure, is_live = a.is_live)

"""
    batch_payload(s::PricedSlate, status) -> NamedTuple

The header. Every field here is a slate-wide fact that cannot be recovered from the legs.

`k_risk` is the single drawdown factor `SlateDrawdown` applied to every leg; `capped` says
whether `FixedCap` had to bind. `capped == true` on most slates means `λ` is set too loose --
move `λ`, because `risk_factor` is homogeneous of degree 0 and rescaling the stakes is a no-op.
"""
batch_payload(s::PricedSlate, status::BatchState = PRICED) = (
    slate_id = string(s.slate_id), window = string(s.window), as_of = string(s.as_of),
    status = string(status),
    bankroll = round(s.bankroll, digits = 2),
    total_risk = round(s.total_risk, digits = 2),
    slate_exposure = round(s.slate_exposure, digits = 6),
    exposure_cap = s.exposure_cap,
    exposure_pct = round(100 * s.slate_exposure, digits = 2),
    cap_pct = round(100 * s.exposure_cap, digits = 2),
    k_risk = round(s.k_risk, digits = 6),
    risk_lambda = s.risk_lambda,
    capped = s.capped,
    n_fixtures = n_fixtures(s), n_legs = n_legs(s), n_blocked = length(s.blocked),
    fold_idx = s.fold_idx, warning = s.warning,
    n_low_confidence = count(==(:low), s.sheet.fill_confidence))

"""
    card_payload(s::PricedSlate) -> Vector{NamedTuple}

One card per fixture, sorted by expected value descending -- the console's default sort key and
the order an operator scans in.

Per leg the card carries BOTH the probability pair and the odds pair, because they answer
different questions and the page shows both:

* `p_model` / `p_market` are drawn as two bars on **one shared 0-1 scale**, so the visible
  overhang of the model bar past the market bar *is* the edge. No arithmetic, no colour
  convention to learn, and 21 cards' worth of it comparable at a glance.
* `fair_odds = 1 / p_model` against `venue_odds` is the same comparison in the denomination that
  gets typed into an exchange.

`ev_pct` is `edge / p_market` -- the edge as a fraction of the market's own price, which is the
scale-free version and the only one that is comparable between a 1.2 shot and a 12.0 one.
"""
function card_payload(s::PricedSlate)
    sheet = s.sheet
    byfix = Dict{Int,Vector{Int}}()
    for i in 1:nrow(sheet)
        push!(get!(byfix, sheet.match_id[i], Int[]), i)
    end
    cards = NamedTuple[]
    for (mid, idx) in byfix
        ci = findfirst(c -> c.fixture.m_id == mid, s.cards)
        f  = ci === nothing ? nothing : s.cards[ci].fixture
        lu = ci === nothing ? nothing : s.cards[ci].lineup
        legs = [(
            selection       = String(sheet.selection[i]),
            market          = sheet.group[i],
            line            = sheet.line[i],
            venue_selection = String(sheet.venue_selection[i]),
            side            = String(sheet.side[i]),
            venue_odds      = round(sheet.venue_odds[i], digits = 3),
            effective_odds  = round(sheet.odds[i], digits = 3),
            fair_odds       = sheet.p_model[i] > 0 ?
                              round(1 / sheet.p_model[i], digits = 3) : 0.0,
            p_model         = round(sheet.p_model[i], digits = 4),
            p_market        = round(sheet.p_market[i], digits = 4),
            edge            = round(sheet.edge[i], digits = 4),
            edge_pp         = round(100 * sheet.edge[i], digits = 2),
            ev_pct          = sheet.p_market[i] > 0 ?
                              round(100 * sheet.edge[i] / sheet.p_market[i], digits = 2) : 0.0,
            risk            = round(sheet.risk[i], digits = 2),
            venue_stake     = round(sheet.venue_stake[i], digits = 2),
            depth_touch     = round(sheet.depth_touch[i], digits = 2),
            depth_book      = round(sheet.depth_book[i], digits = 2),
            slippage_pct    = isnan(sheet.expected_slippage[i]) ? nothing :
                              round(100 * sheet.expected_slippage[i], digits = 3),
            fillable        = sheet.fillable[i],
            confidence      = String(sheet.fill_confidence[i]),
        ) for i in idx]
        risk = sum(l -> l.risk, legs)
        push!(cards, (
            match_id = mid,
            home = f === nothing ? "?" : f.home,
            away = f === nothing ? "?" : f.away,
            kickoff = f === nothing ? "" : string(f.kickoff),
            tournament_id = f === nothing ? 0 : f.tournament_id,
            minutes_to_kickoff = f === nothing ? 0 :
                Int(round(Dates.value(f.kickoff - s.as_of) / 60_000)),
            lineup_source = lu === nothing ? "none" : String(lu.source),
            lineup_confirmed = lu === nothing ? false : lu.confirmed,
            lineup_lead_min = (lu === nothing || f === nothing) ? nothing :
                Int(round(Dates.value(f.kickoff - lu.scraped_at) / 60_000)),
            risk = round(risk, digits = 2),
            n_legs = length(legs),
            # The sort key. Risk-weighted so a card is ranked by the EV it actually carries,
            # not by its most flattering leg -- a £2 leg at +9% must not outrank a £26 one at +5%.
            ev_pct = risk > 0 ? round(sum(l -> l.ev_pct * l.risk, legs) / risk, digits = 2) : 0.0,
            legs = legs,
        ))
    end
    sort!(cards, by = c -> -c.ev_pct)
    return cards
end

"""
    slate_snapshot(s::PricedSlate, a::PaperAccount; status) -> NamedTuple

The whole payload, ready for `JSON3.write`. One object, one WebSocket frame, one `x-for` on the
client.

`blocked` is included and is not optional. `MatchDayResult.blocked` exists so that "no bets
today" and "the pipeline is broken" are distinguishable; a console that dropped it would put that
distinction back out of reach at exactly the moment it matters.
"""
slate_snapshot(s::PricedSlate, a::PaperAccount; status::BatchState = PRICED) = (
    at      = string(now()),
    account = account_payload(a),
    batch   = batch_payload(s, status),
    cards   = card_payload(s),
    blocked = [(match_id = c.fixture.m_id,
                fixture  = c.fixture.home * " v " * c.fixture.away,
                kickoff  = string(c.fixture.kickoff),
                reasons  = [string(k) * ": " * v for (k, v) in c.readiness.reasons])
               for c in s.blocked if c.readiness isa Blocked],
)
