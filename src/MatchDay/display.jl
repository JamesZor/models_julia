# src/MatchDay/display.jl
#
# Pretty printing. Follows the house style (src/features/display.jl, src/Data/display.jl):
# a rich `MIME"text/plain"` tree for the REPL, and a compact one-liner for arrays and logging.
#
# The MatchDaySpec method is the one worth caring about: it prints the spec AS THE PIPELINE, in
# execution order with stage numbers, so `spec` at the REPL doubles as the diagram. The stage
# numbers are deliberately out of order -- the book is built before features.

# ==============================================================================
# 0. Helpers
# ==============================================================================

"Members of a `…Chain`, whichever field name it uses."
_chain_members(c) = getfield(c, first(fieldnames(typeof(c))))

_is_chain(x) = x isa SourceChain || x isa GateChain || x isa ResolverChain ||
               x isa MaterialiserChain

"One-line description of a component: type name plus its fields, or its members if a chain."
function _component_line(x)
    name = string(nameof(typeof(x)))
    _is_chain(x) &&
        return name * "(" * join(string.(nameof.(typeof.(_chain_members(x)))), " → ") * ")"
    props = propertynames(x)
    isempty(props) && return name
    inner = join(("$p=$(_fmt(getproperty(x, p)))" for p in props), ", ")
    return "$name($inner)"
end

_fmt(v::Period)  = string(v)
_fmt(v::Float64) = string(round(v, digits = 4))
_fmt(v::AbstractString) = "\"$v\""
_fmt(v) = string(v)

"Print one `├──`/`└──` row."
function _leaf(io::IO, label, value; last::Bool = false, indent = "  ",
               label_color = :white, value_color = :cyan)
    printstyled(io, indent, last ? "└── " : "├── ", color = :light_black)
    printstyled(io, rpad(label, 14), color = label_color)
    printstyled(io, string(value), color = value_color)
    println(io)
end

# ==============================================================================
# 1. MatchDaySpec -- prints as the pipeline
# ==============================================================================

function Base.show(io::IO, ::MIME"text/plain", s::MatchDaySpec)
    printstyled(io, "MatchDaySpec", color = :cyan, bold = true)
    printstyled(io, "  (fixture list → stake sheet)\n", color = :light_black)
    println(io, "="^70)

    stages = ((1, "fixtures",   :fixtures,   "what is on"),
              (2, "identity",   :identity,   "who is who on the exchange"),
              (3, "lineups",    :lineups,    "which XI"),
              (5, "book",       :book,       "where prices come from"),
              (5, "quote_rule", :quote_rule, "which number in the book"),
              (5, "instrument", :instrument, "back it, or lay the complement"),
              (4, "features",   :features,   "materialise for unseen fixtures"),
              (7, "gate",       :gate,       "whether to bet at all"),
              (8, "rounding",   :rounding,   "the exchange minimum"))

    for (n, field, sym, blurb) in stages
        printstyled(io, " ", color = :light_black)
        printstyled(io, "S$n ", color = :yellow, bold = true)
        printstyled(io, rpad(field, 11), color = :magenta, bold = true)
        printstyled(io, _component_line(getproperty(s, sym)), color = :cyan)
        println(io)
        printstyled(io, "      ", rpad("", 11), blurb, "\n", color = :light_black)
    end

    println(io, "="^70)
    printstyled(io, " markets  ", color = :magenta, bold = true)
    printstyled(io, "$(length(s.markets.markets)) configured: ", color = :cyan)
    printstyled(io, join(unique(string.(Data.market_group.(s.markets.markets))), ", "),
                "\n", color = :light_black)
    printstyled(io, " stage order is NOT numerical: the book is built before features.\n",
                color = :light_black, italic = true)
end

Base.show(io::IO, s::MatchDaySpec) = print(io,
    "MatchDaySpec(", nameof(typeof(s.fixtures)), ", ", nameof(typeof(s.book)), ", ",
    nameof(typeof(s.instrument)), ", ", nameof(typeof(s.rounding)), ")")

# ==============================================================================
# 2. MatchDayResult
# ==============================================================================

function Base.show(io::IO, ::MIME"text/plain", r::MatchDayResult)
    printstyled(io, "MatchDayResult", color = :cyan, bold = true)
    printstyled(io, "  as_of $(r.as_of)\n", color = :yellow)
    println(io, "="^70)

    n_priced = length(r.cards) - length(r.blocked)
    _leaf(io, "fixtures", length(r.cards))
    _leaf(io, "priced", n_priced)
    printstyled(io, "  ├── ", color = :light_black)
    printstyled(io, rpad("blocked", 14), color = :white)
    printstyled(io, length(r.blocked),
                color = isempty(r.blocked) ? :cyan : :red)
    println(io)
    _leaf(io, "bets", nrow(r.sheet))
    _leaf(io, "quotes", nrow(r.odds), last = true)

    if !isempty(r.sheet)
        n_lay = count(==(:lay), r.sheet.side)
        println(io)
        printstyled(io, " staking\n", color = :magenta, bold = true)
        _leaf(io, "total risk", round(sum(r.sheet.risk), digits = 2))
        _leaf(io, "to place", round(sum(r.sheet.venue_stake), digits = 2))
        _leaf(io, "via lay", "$n_lay of $(nrow(r.sheet))", last = true)
    end

    if !isempty(r.blocked)
        println(io)
        printstyled(io, " refusals", color = :magenta, bold = true)
        printstyled(io, "  (blocked_report(r) for detail)\n", color = :light_black)
        counts = Dict{Symbol,Int}()
        for c in r.blocked, (k, _) in c.readiness.reasons
            counts[k] = get(counts, k, 0) + 1
        end
        ks = sort(collect(keys(counts)))
        for (i, k) in enumerate(ks)
            _leaf(io, string(k), "$(counts[k]) fixture(s)";
                  last = i == length(ks), value_color = :red)
        end
    elseif isempty(r.sheet)
        println(io)
        printstyled(io, " nothing blocked and no bets: the book qualified nobody.\n",
                    color = :light_black, italic = true)
    end
end

Base.show(io::IO, r::MatchDayResult) = print(io,
    "MatchDayResult($(nrow(r.sheet)) bets, $(length(r.cards) - length(r.blocked)) priced, ",
    "$(length(r.blocked)) blocked, as_of $(r.as_of))")

# ==============================================================================
# 3. Fixture / identity / lineup / card
# ==============================================================================

function Base.show(io::IO, ::MIME"text/plain", f::Fixture)
    printstyled(io, "Fixture ", color = :cyan, bold = true)
    printstyled(io, "$(f.home) v $(f.away)\n", color = :white, bold = true)
    _leaf(io, "match_id", f.m_id)
    _leaf(io, "kickoff", f.kickoff)
    _leaf(io, "tournament", f.tournament_id, last = true)
end

Base.show(io::IO, f::Fixture) =
    print(io, "Fixture($(f.m_id), $(f.home) v $(f.away), $(f.kickoff))")

function Base.show(io::IO, ::MIME"text/plain", r::Resolved)
    printstyled(io, "Resolved", color = :green, bold = true)
    printstyled(io, r.verified ? "  ✓ verified\n" : "  (unverified)\n",
                color = r.verified ? :green : :yellow)
    _leaf(io, "fixture", "$(r.fixture.home) v $(r.fixture.away)")
    _leaf(io, "bf_event", r.bf_event_id)
    _leaf(io, "markets", "$(length(r.market_ids)): " *
                         join(sort(collect(keys(r.market_ids))), ", "), last = true)
end

Base.show(io::IO, r::Resolved) =
    print(io, "Resolved($(r.fixture.m_id) → $(r.bf_event_id), $(length(r.market_ids)) markets)")

function Base.show(io::IO, ::MIME"text/plain", u::Unresolved)
    printstyled(io, "Unresolved", color = :red, bold = true)
    printstyled(io, "  $(u.reason)\n", color = :red)
    _leaf(io, "fixture", "$(u.fixture.home) v $(u.fixture.away)")
    _leaf(io, "match_id", u.fixture.m_id, last = true)
    if u.reason === :absent_from_crosswalk
        printstyled(io, "\n  No betfair.match_meta row. The resolution job has not seen this\n",
                    color = :light_black)
        printstyled(io, "  fixture -- it resolves 100% when it runs, so this is a stopped job,\n",
                    color = :light_black)
        printstyled(io, "  not a matching problem.\n", color = :light_black)
    end
end

Base.show(io::IO, u::Unresolved) = print(io, "Unresolved($(u.fixture.m_id), $(u.reason))")

function Base.show(io::IO, ::MIME"text/plain", l::Lineup)
    printstyled(io, "Lineup ", color = :cyan, bold = true)
    printstyled(io, "$(l.source)\n", color = :yellow)
    _leaf(io, "starters", "$(count(p -> !p.substitute, l.home)) home / " *
                          "$(count(p -> !p.substitute, l.away)) away")
    _leaf(io, "squad", "$(length(l.home)) / $(length(l.away))")
    printstyled(io, "  ├── ", color = :light_black)
    printstyled(io, rpad("confirmed", 14), color = :white)
    printstyled(io, l.confirmed, color = l.confirmed ? :green : :yellow)
    l.confirmed || printstyled(io, "   (predicted XI -- `confirmed` has never yet been true)",
                               color = :light_black)
    println(io)
    _leaf(io, "scraped_at", l.scraped_at, last = true)
end

Base.show(io::IO, l::Lineup) =
    print(io, "Lineup($(l.source), $(length(l.home))+$(length(l.away)), ",
              l.confirmed ? "confirmed" : "predicted", ")")

function Base.show(io::IO, ::MIME"text/plain", c::FixtureCard)
    printstyled(io, "FixtureCard ", color = :cyan, bold = true)
    printstyled(io, "$(c.fixture.home) v $(c.fixture.away)\n", color = :white, bold = true)
    _leaf(io, "kickoff", "$(c.fixture.kickoff)  (as_of $(c.as_of))")
    printstyled(io, "  ├── ", color = :light_black)
    printstyled(io, rpad("identity", 14), color = :white)
    if resolved(c)
        printstyled(io, "resolved → $(c.identity.bf_event_id) ",
                    "($(length(c.identity.market_ids)) markets)", color = :green)
    else
        printstyled(io, "UNRESOLVED ($(c.identity.reason))", color = :red)
    end
    println(io)
    _leaf(io, "lineup", c.lineup === nothing ? "none" :
          "$(c.lineup.source), $(round(Dates.value(c.fixture.kickoff - c.lineup.scraped_at)/3.6e6, digits=1))h before KO";
          value_color = c.lineup === nothing ? :red : :cyan)
    if c.readiness === nothing
        _leaf(io, "readiness", "not evaluated", last = true, value_color = :light_black)
    elseif is_ready(c.readiness)
        _leaf(io, "readiness", "READY", last = true, value_color = :green)
    else
        printstyled(io, "  └── ", color = :light_black)
        printstyled(io, rpad("readiness", 14), color = :white)
        printstyled(io, "BLOCKED\n", color = :red, bold = true)
        for (k, v) in c.readiness.reasons
            printstyled(io, "        • $k: ", color = :red)
            printstyled(io, v, "\n", color = :light_black)
        end
    end
end

Base.show(io::IO, c::FixtureCard) = print(io,
    "FixtureCard($(c.fixture.m_id), $(c.fixture.home) v $(c.fixture.away), ",
    resolved(c) ? "resolved" : "UNRESOLVED($(c.identity.reason))",
    c.readiness === nothing ? "" : ", $(is_ready(c.readiness) ? "ready" : "blocked")", ")")

# ==============================================================================
# 4. Readiness
# ==============================================================================

Base.show(io::IO, ::Ready) = print(io, "Ready()")
Base.show(io::IO, b::Blocked) =
    print(io, "Blocked(", join(("$k: $v" for (k, v) in b.reasons), "; "), ")")

function Base.show(io::IO, ::MIME"text/plain", b::Blocked)
    printstyled(io, "Blocked", color = :red, bold = true)
    printstyled(io, "  $(length(b.reasons)) reason(s)\n", color = :light_black)
    for (i, (k, v)) in enumerate(b.reasons)
        printstyled(io, "  ", i == length(b.reasons) ? "└── " : "├── ", color = :light_black)
        printstyled(io, "$k: ", color = :red, bold = true)
        printstyled(io, v, "\n", color = :light_black)
    end
end

# ==============================================================================
# 5. Book and instruments
# ==============================================================================

function Base.show(io::IO, ::MIME"text/plain", b::BookLevels)
    printstyled(io, "BookLevels ", color = :cyan, bold = true)
    printstyled(io, "@ $(b.ts)\n", color = :yellow)
    printstyled(io, "        back (bid)          lay (ask)\n", color = :light_black)
    n = max(length(b.back), length(b.lay))
    for i in 1:n
        bp = i <= length(b.back) ? @sprintf("%6.2f", b.back[i]) : "     -"
        bs = i <= length(b.back_size) ? @sprintf("£%-8.0f", b.back_size[i]) : "         "
        lp = i <= length(b.lay) ? @sprintf("%6.2f", b.lay[i]) : "     -"
        ls = i <= length(b.lay_size) ? @sprintf("£%-8.0f", b.lay_size[i]) : ""
        printstyled(io, "  L$i  ", color = :light_black)
        printstyled(io, bp, color = :green); printstyled(io, " $bs", color = :light_black)
        printstyled(io, lp, color = :red);   printstyled(io, " $ls\n", color = :light_black)
    end
    isnan(b.matched) || printstyled(io, "  matched £$(round(b.matched, digits=0))\n",
                                    color = :light_black)
    printstyled(io, "  bid = available to BACK, ask = available to LAY\n",
                color = :light_black, italic = true)
end

Base.show(io::IO, b::BookLevels) = print(io,
    "BookLevels(back ", isempty(b.back) ? "-" : string(b.back[1]),
    " / lay ", isempty(b.lay) ? "-" : string(b.lay[1]), " @ $(b.ts))")

function Base.show(io::IO, ::MIME"text/plain", i::Instrument)
    printstyled(io, "Instrument ", color = :cyan, bold = true)
    printstyled(io, uppercase(string(i.side)), color = i.side === :lay ? :magenta : :green,
                bold = true)
    println(io)
    _leaf(io, "selection", "$(i.key.group)$(i.key.line == 0.0 ? "" : " $(i.key.line)") " *
                           "$(i.key.selection)")
    _leaf(io, "venue odds", round(i.venue_odds, digits = 3))
    _leaf(io, "effective", round(i.odds, digits = 4))
    _leaf(io, "leverage", "$(round(i.leverage, digits = 3))x", last = true)
    if i.side === :lay
        printstyled(io, "\n  laying $(round(i.venue_odds, digits=2)) = backing ",
                    "$(round(i.odds, digits=3)) in risk units.\n", color = :light_black)
        printstyled(io, "  post £$(round(i.leverage, digits=2)) with the backer per £1 at risk.\n",
                    color = :light_black)
    end
end

Base.show(io::IO, i::Instrument) = print(io,
    "Instrument($(i.key.selection), ", uppercase(string(i.side)),
    " @ $(round(i.venue_odds, digits=3)) → eff $(round(i.odds, digits=3)))")

# ==============================================================================
# 6. Components -- generic tree, mirroring src/features/display.jl
# ==============================================================================

const MatchDayComponent = Union{AbstractFixtureSource, AbstractIdentityResolver,
                                AbstractLineupSource, AbstractBookSource, AbstractQuoteRule,
                                AbstractInstrumentRule, AbstractStakeRounding,
                                AbstractFeatureMaterialiser, AbstractReadinessGate}

function Base.show(io::IO, ::MIME"text/plain", c::MatchDayComponent)
    name = string(nameof(typeof(c)))
    printstyled(io, name, color = :green, bold = true)

    if _is_chain(c)
        members = _chain_members(c)
        conj = c isa GateChain
        printstyled(io, conj ? "  (conjunctive: ALL must pass, reasons collected)\n" :
                              "  (first success wins)\n", color = :light_black)
        for (i, m) in enumerate(members)
            printstyled(io, "  ", i == length(members) ? "└── " : "├── ", color = :light_black)
            printstyled(io, _component_line(m), "\n", color = :cyan)
        end
        return
    end

    props = propertynames(c)
    if isempty(props)
        println(io, "()")
        return
    end
    println(io)
    for (i, p) in enumerate(props)
        _leaf(io, string(p), _fmt(getproperty(c, p)); last = i == length(props))
    end
end

Base.show(io::IO, c::MatchDayComponent) = print(io, _component_line(c))
